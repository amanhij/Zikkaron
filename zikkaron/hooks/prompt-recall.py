#!/usr/bin/env python3
"""Zikkaron auto-recall — UserPromptSubmit hook handler.

Automatically retrieves relevant memories for every user prompt and
outputs JSON with additionalContext so Claude receives them without
needing to call any tool.

Uses FTS5 keyword search + optional sqlite-vec vector search for hybrid retrieval.
Each invocation is a fresh process; vector search is skipped if model load exceeds 150ms.

Target latency: <100ms (FTS5-only). Falls back to FTS5-only if vector search is slow.
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# Maximum memories to inject per turn
MAX_RESULTS = 5
# Minimum heat to surface (very low — let ranking handle quality)
MIN_HEAT = 0.0
# Maximum total characters to inject (keep context budget reasonable)
MAX_CONTEXT_CHARS = 3000
# Time budget in seconds — if we exceed this, return what we have
TIME_BUDGET = 0.5


def _extract_query(data: dict) -> str:
    """Extract the user's prompt text from hook input."""
    # UserPromptSubmit provides the prompt in 'prompt' field
    prompt = data.get("prompt", "")
    if not prompt:
        prompt = data.get("user_prompt", "")
    return str(prompt).strip()


def _preprocess_fts(query: str) -> str:
    """Convert user prompt into an FTS5 query (OR-joined terms)."""
    # Strip punctuation, split into words, filter short ones
    words = []
    for word in query.split():
        cleaned = "".join(c for c in word if c.isalnum() or c == "_")
        if len(cleaned) >= 2:
            words.append(cleaned)
    if not words:
        return ""
    # Use OR to match any relevant term
    return " OR ".join(words[:15])  # Cap at 15 terms


def _fts_search(conn, query: str, directory: str) -> list[dict]:
    """Fast FTS5 keyword search, scoped to current project first."""
    fts_query = _preprocess_fts(query)
    if not fts_query:
        return []
    try:
        # Search current project first, then global fallback
        rows = conn.execute(
            "SELECT m.id, m.content, m.heat, m.directory_context, "
            "-bm25(memories_fts) as score "
            "FROM memories m "
            "JOIN memories_fts fts ON m.id = fts.rowid "
            "WHERE memories_fts MATCH ? AND m.heat >= ? AND m.is_stale = 0 "
            "AND m.directory_context = ? "
            "ORDER BY score DESC LIMIT ?",
            (fts_query, MIN_HEAT, directory, MAX_RESULTS * 3),
        ).fetchall()
        # If not enough project-local results, supplement with global
        if len(rows) < MAX_RESULTS and directory:
            global_rows = conn.execute(
                "SELECT m.id, m.content, m.heat, m.directory_context, "
                "-bm25(memories_fts) as score "
                "FROM memories m "
                "JOIN memories_fts fts ON m.id = fts.rowid "
                "WHERE memories_fts MATCH ? AND m.heat >= ? AND m.is_stale = 0 "
                "AND m.directory_context != ? "
                "ORDER BY score DESC LIMIT ?",
                (fts_query, MIN_HEAT, directory, MAX_RESULTS * 2),
            ).fetchall()
            rows = list(rows) + list(global_rows)

        results = []
        for r in rows:
            results.append({
                "id": r[0], "content": r[1], "heat": r[2],
                "directory": r[3], "score": r[4], "source": "fts",
            })
        return results
    except Exception:
        return []


def _vector_search(conn, query: str, directory: str) -> list[dict]:
    """Vector similarity search using sqlite-vec + sentence-transformers.

    Only used if model loads within 200ms (already cached/warm).
    Falls back gracefully — FTS5 alone provides good keyword coverage.
    """
    try:
        import importlib
        # Quick check: skip if sentence_transformers isn't importable
        spec = importlib.util.find_spec("sentence_transformers")
        if spec is None:
            return []
        st = importlib.import_module("sentence_transformers")
        t0 = time.monotonic()
        model = st.SentenceTransformer(
            os.environ.get("ZIKKARON_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
        if time.monotonic() - t0 > 0.15:
            # Model loading took too long, skip vector search
            return []
        embedding = model.encode(query, normalize_embeddings=True).tobytes()

        rows = conn.execute(
            "SELECT v.rowid, v.distance, m.content, m.heat, m.directory_context "
            "FROM memory_vectors v "
            "JOIN memories m ON m.id = v.rowid "
            "WHERE v.embedding MATCH ? AND k = ? "
            "AND m.is_stale = 0 "
            "ORDER BY v.distance",
            (embedding, MAX_RESULTS * 3),
        ).fetchall()

        results = []
        for r in rows:
            if r[3] >= MIN_HEAT and r[4] != "":  # heat filter + has directory
                results.append({
                    "id": r[0], "content": r[2], "heat": r[3],
                    "directory": r[4], "score": 1.0 / (1.0 + r[1]),  # distance to similarity
                    "source": "vec",
                })
        return results
    except Exception:
        return []


def _merge_and_rank(fts_results: list, vec_results: list, directory: str) -> list[dict]:
    """Merge FTS and vector results, deduplicate, boost project matches."""
    seen = {}

    for r in fts_results:
        mid = r["id"]
        if mid not in seen:
            seen[mid] = r
            seen[mid]["combined"] = r["score"]
        else:
            seen[mid]["combined"] += r["score"] * 0.5

    for r in vec_results:
        mid = r["id"]
        if mid not in seen:
            seen[mid] = r
            seen[mid]["combined"] = r["score"]
        else:
            seen[mid]["combined"] += r["score"]

    results = list(seen.values())

    for r in results:
        # Boost memories from the current project directory
        if r["directory"] == directory:
            r["combined"] *= 1.5
        # Boost semantic/manual memories over action stream
        content = r.get("content", "")
        if not content.startswith("Session activity"):
            r["combined"] *= 2.0
        # Boost by heat (hotter = more relevant)
        r["combined"] *= (1.0 + r.get("heat", 0))

    results.sort(key=lambda x: -x["combined"])
    return results[:MAX_RESULTS]


def _format_context(memories: list, directory: str) -> str:
    """Format memories as concise context for injection."""
    if not memories:
        return ""

    lines = []
    lines.append("# Zikkaron — Auto-Recall\n")
    total_chars = 0
    for m in memories:
        content = m["content"]
        if total_chars + len(content) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 50:
                content = content[:remaining] + "..."
            else:
                break
        proj = ""
        if m["directory"] and m["directory"] != directory:
            proj = f" [{Path(m['directory']).name}]"
        lines.append(f"- {content}{proj}")
        total_chars += len(content)

    lines.append(f"\n*{len(memories)} memories surfaced for: {directory}*")
    return "\n".join(lines)


def main():
    start = time.monotonic()

    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return

    query = _extract_query(data)
    if not query or len(query) < 2:
        # Too short to meaningfully search
        return

    directory = data.get("cwd", "")

    db_path = Path(os.environ.get("ZIKKARON_DB_PATH", "~/.zikkaron/memory.db")).expanduser()
    if not db_path.exists():
        return

    try:
        conn = sqlite3.connect(str(db_path), timeout=1)
        conn.row_factory = None  # Use tuples for speed
        # Enable sqlite-vec extension
        conn.enable_load_extension(True)
        try:
            import sqlite_vec
            sqlite_vec.load(conn)
        except Exception:
            pass
    except Exception:
        return

    # Phase 1: FTS5 search (always fast, <50ms)
    fts_results = _fts_search(conn, query, directory)

    # Phase 2: Vector search (if time budget allows)
    vec_results = []
    elapsed = time.monotonic() - start
    if elapsed < TIME_BUDGET * 0.6:
        vec_results = _vector_search(conn, query, directory)

    conn.close()

    # Merge and rank
    merged = _merge_and_rank(fts_results, vec_results, directory)

    if not merged:
        return

    context = _format_context(merged, directory)
    if not context:
        return

    # Output JSON with hookSpecificOutput.additionalContext per Claude Code hook contract
    output = {"hookSpecificOutput": {"additionalContext": context}}
    print(json.dumps(output))


if __name__ == "__main__":
    main()
