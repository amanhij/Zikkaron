"""Entry point for python -m zikkaron."""

import argparse
import sys
from pathlib import Path

from zikkaron import __version__
from zikkaron.server import main

VALID_TRANSPORTS = ("stdio", "sse", "streamable-http")

STARTUP_BANNER = f"""\
=== Zikkaron v{__version__} ===
Biologically-inspired persistent memory engine for Claude Code

Active modules:
  * StorageEngine         (SQLite WAL + FTS5 + sqlite-vec)
  * EmbeddingEngine       (sentence-transformers)
  * SensoryBuffer         (episode capture)
  * MemoryThermodynamics  (surprise, importance, valence, decay)
  * KnowledgeGraph        (typed relationships, causal detection)
  * HippoRetriever        (PPR + vector + FTS5 + spreading activation + fractal)
  * MemoryCurator         (merge/link/create, contradiction, memify)
  * AstrocyteEngine       (background consolidation daemon)
  * AstrocytePool         (domain-aware processes: code/decisions/errors/deps)
  * SleepComputeEngine    (dream replay, compression, community detection)
  * FractalMemoryTree     (hierarchical multi-scale retrieval)
  * ProspectiveMemory     (future-oriented triggers)
  * NarrativeEngine       (autobiographical project stories)
  * StalenessDetector     (file-change watchdog)

MCP Tools: remember, recall, forget, validate_memory, get_project_context,
           consolidate_now, memory_stats, rate_memory, recall_hierarchical,
           drill_down, create_trigger, get_project_story, seed_project,
           checkpoint, restore, anchor, navigate_memory, assess_coverage,
           detect_gaps, install_hooks, sync_instructions

MCP Resources: memory://stats, memory://hot, memory://stale,
               memory://processes, memory://narrative/{{directory}}
"""


def _init_replay_lightweight(db_path=None):
    """Initialize only the engines needed for drain/restore (no daemons, no server)."""
    import logging
    # Suppress all library logging — hooks must only output data to stdout
    logging.disable(logging.CRITICAL)

    from zikkaron.config import Settings
    from zikkaron.storage import StorageEngine
    from zikkaron.embeddings import EmbeddingEngine
    from zikkaron.cognitive_map import CognitiveMap
    from zikkaron.metacognition import MetaCognition
    from zikkaron.knowledge_graph import KnowledgeGraph
    from zikkaron.retrieval import HippoRetriever
    from zikkaron.restoration import HippocampalReplay

    settings = Settings()
    storage = StorageEngine(db_path or settings.DB_PATH)
    embeddings = EmbeddingEngine(settings.EMBEDDING_MODEL)
    kg = KnowledgeGraph(storage, settings)
    cognitive_map = CognitiveMap(storage, settings)
    retriever = HippoRetriever(storage, embeddings, kg, settings)
    retriever.set_cognitive_map(cognitive_map)
    metacognition = MetaCognition(storage, embeddings, kg, settings)

    replay = HippocampalReplay(
        storage=storage,
        embeddings=embeddings,
        retriever=retriever,
        cognitive_map=cognitive_map,
        metacognition=metacognition,
        settings=settings,
    )
    return storage, replay


def cmd_drain(args):
    """Pre-compaction drain: save context to DB before Claude compacts."""
    import json
    directory = args.directory
    storage, replay = _init_replay_lightweight(args.db_path)
    try:
        result = replay.pre_compact_drain(directory)
        # Output JSON to stdout so hook can parse it if needed
        print(json.dumps(result))
    finally:
        storage.close()


def cmd_restore(args):
    """Post-compaction restore: reconstruct context and print markdown to stdout."""
    import json
    directory = args.directory
    storage, replay = _init_replay_lightweight(args.db_path)
    try:
        result = replay.restore(directory)
        formatted = result.get("formatted", "")
        if formatted:
            print(formatted)
    finally:
        storage.close()


def cmd_capture(args):
    """Lightweight action capture — writes directly to SQLite without ML models.

    Used by PostToolCall hooks and manual capture. Only imports sqlite3.
    """
    import sqlite3
    from datetime import datetime, timezone
    from zikkaron.config import Settings

    settings = Settings()
    db_path = Path(args.db_path or settings.DB_PATH).expanduser()
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path), timeout=1)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS action_log("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "tool_name TEXT NOT NULL,"
        "tool_input_summary TEXT DEFAULT '',"
        "directory TEXT DEFAULT '',"
        "session_id TEXT DEFAULT '',"
        "timestamp TEXT NOT NULL,"
        "processed INTEGER DEFAULT 0)"
    )
    conn.execute(
        "INSERT INTO action_log (tool_name, tool_input_summary, directory, session_id, timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            args.tool_name,
            args.summary or "",
            args.directory or "",
            args.session or "",
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def cmd_context(args):
    """Lightweight context query — reads hot memories without loading ML models.

    Used by SessionStart hooks to inject context on every session.
    """
    import json
    import sqlite3
    from zikkaron.config import Settings

    settings = Settings()
    db_path = Path(args.db_path or settings.DB_PATH).expanduser()
    if not db_path.exists():
        return

    directory = args.directory
    conn = sqlite3.connect(str(db_path), timeout=2)
    conn.row_factory = sqlite3.Row

    hot = conn.execute(
        "SELECT content, heat FROM memories "
        "WHERE directory_context = ? AND heat > 0.5 "
        "ORDER BY heat DESC LIMIT 6",
        (directory,),
    ).fetchall()

    anchored = conn.execute(
        "SELECT content FROM memories "
        "WHERE is_protected = 1 AND heat > 0 AND tags LIKE '%_anchor%' "
        "ORDER BY created_at DESC LIMIT 4"
    ).fetchall()

    conn.close()

    if not hot and not anchored:
        return

    print("# Zikkaron — Session Context\n")
    if anchored:
        print("## Critical Facts")
        for row in anchored:
            print(f"- {row['content'][:200]}")
        print()
    if hot:
        print("## Project Context")
        for row in hot:
            content = row["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"- [{row['heat']:.1f}] {content}")
        print()
    print(f"*Context for: {directory}*")


def cmd_stats(args):
    """Show detailed memory statistics."""
    import json
    import sqlite3
    from datetime import datetime, timezone
    from zikkaron.config import Settings

    settings = Settings()
    db_path = Path(args.db_path or settings.DB_PATH).expanduser()
    if not db_path.exists():
        print("No database found. Run zikkaron and store some memories first.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path), timeout=2)
    conn.row_factory = sqlite3.Row
    project = str(Path(args.project).resolve()) if args.project else None

    where = "WHERE directory_context = ?" if project else ""
    params: tuple = (project,) if project else ()

    # ── Core counts ──
    total = conn.execute(f"SELECT COUNT(*) FROM memories {where}", params).fetchone()[0]
    if total == 0:
        label = f"project {project}" if project else "database"
        print(f"No memories in {label}.", file=sys.stderr)
        conn.close()
        sys.exit(0)

    active = conn.execute(
        f"SELECT COUNT(*) FROM memories {where + ' AND ' if where else 'WHERE '}is_stale = 0 AND heat >= 0.05",
        params,
    ).fetchone()[0]
    stale = conn.execute(
        f"SELECT COUNT(*) FROM memories {where + ' AND ' if where else 'WHERE '}is_stale = 1",
        params,
    ).fetchone()[0]
    archived = conn.execute(
        f"SELECT COUNT(*) FROM memories {where + ' AND ' if where else 'WHERE '}heat < 0.05",
        params,
    ).fetchone()[0]
    try:
        protected = conn.execute(
            f"SELECT COUNT(*) FROM memories {where + ' AND ' if where else 'WHERE '}is_protected = 1",
            params,
        ).fetchone()[0]
    except Exception:
        protected = 0  # pre-migration DB

    # ── Type breakdown ──
    episodic = conn.execute(
        f"SELECT COUNT(*) FROM memories {where + ' AND ' if where else 'WHERE '}store_type = 'episodic'",
        params,
    ).fetchone()[0]
    semantic = conn.execute(
        f"SELECT COUNT(*) FROM memories {where + ' AND ' if where else 'WHERE '}store_type = 'semantic'",
        params,
    ).fetchone()[0]

    # ── Compression levels ──
    comp_0 = conn.execute(
        f"SELECT COUNT(*) FROM memories {where + ' AND ' if where else 'WHERE '}compression_level = 0",
        params,
    ).fetchone()[0]
    comp_1 = conn.execute(
        f"SELECT COUNT(*) FROM memories {where + ' AND ' if where else 'WHERE '}compression_level = 1",
        params,
    ).fetchone()[0]
    comp_2 = conn.execute(
        f"SELECT COUNT(*) FROM memories {where + ' AND ' if where else 'WHERE '}compression_level = 2",
        params,
    ).fetchone()[0]

    # ── Heat stats ──
    heat_row = conn.execute(
        f"SELECT MIN(heat), AVG(heat), MAX(heat) FROM memories {where}", params
    ).fetchone()
    heat_min, heat_avg, heat_max = heat_row[0] or 0, heat_row[1] or 0, heat_row[2] or 0

    heat_buckets = []
    for lo, hi, label in [(0, 0.01, "cold (<0.01)"), (0.01, 0.1, "cool (0.01-0.1)"),
                           (0.1, 0.5, "warm (0.1-0.5)"), (0.5, 0.9, "hot (0.5-0.9)"),
                           (0.9, 999, "burning (0.9+)")]:
        c = conn.execute(
            f"SELECT COUNT(*) FROM memories {where + ' AND ' if where else 'WHERE '}heat >= ? AND heat < ?",
            params + (lo, hi),
        ).fetchone()[0]
        heat_buckets.append((label, c))

    # ── Access stats ──
    access_row = conn.execute(
        f"SELECT SUM(access_count), AVG(access_count), MAX(access_count) FROM memories {where}",
        params,
    ).fetchone()
    total_accesses = access_row[0] or 0
    avg_accesses = access_row[1] or 0
    max_accesses = access_row[2] or 0

    useful_row = conn.execute(
        f"SELECT SUM(useful_count) FROM memories {where}", params
    ).fetchone()
    total_useful = useful_row[0] or 0

    never_accessed = conn.execute(
        f"SELECT COUNT(*) FROM memories {where + ' AND ' if where else 'WHERE '}access_count = 0",
        params,
    ).fetchone()[0]

    # ── Temporal stats ──
    oldest = conn.execute(
        f"SELECT MIN(created_at) FROM memories {where}", params
    ).fetchone()[0]
    newest = conn.execute(
        f"SELECT MAX(created_at) FROM memories {where}", params
    ).fetchone()[0]
    last_accessed = conn.execute(
        f"SELECT MAX(last_accessed) FROM memories {where}", params
    ).fetchone()[0]

    now = datetime.now(timezone.utc)
    age_days = None
    if oldest:
        try:
            oldest_dt = datetime.fromisoformat(oldest.replace("Z", "+00:00"))
            age_days = (now - oldest_dt).days
        except Exception:
            pass

    # ── Per-project breakdown (only when no --project filter) ──
    project_rows = []
    if not project:
        project_rows = conn.execute(
            "SELECT directory_context, COUNT(*) as cnt, AVG(heat) as avg_h, "
            "MAX(created_at) as last_created "
            "FROM memories WHERE directory_context != '' "
            "GROUP BY directory_context ORDER BY cnt DESC LIMIT 15"
        ).fetchall()

    # ── Consolidation history ──
    total_consolidations = conn.execute(
        "SELECT COUNT(*) FROM consolidation_log"
    ).fetchone()[0]
    last_consol = conn.execute(
        "SELECT timestamp, duration_ms, memories_added, memories_archived "
        "FROM consolidation_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    avg_duration = conn.execute(
        "SELECT AVG(duration_ms) FROM consolidation_log"
    ).fetchone()[0] or 0

    # ── Knowledge graph ──
    try:
        entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        rel_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
    except Exception:
        entity_count = rel_count = 0

    try:
        causal_edges = conn.execute("SELECT COUNT(*) FROM causal_dag_edges").fetchone()[0]
    except Exception:
        causal_edges = 0

    # ── Action log ──
    try:
        action_total = conn.execute("SELECT COUNT(*) FROM action_log").fetchone()[0]
        action_unprocessed = conn.execute(
            "SELECT COUNT(*) FROM action_log WHERE processed = 0"
        ).fetchone()[0]
    except Exception:
        action_total = action_unprocessed = 0

    # ── Clusters ──
    try:
        cluster_count = conn.execute("SELECT COUNT(*) FROM memory_clusters").fetchone()[0]
    except Exception:
        cluster_count = 0

    # ── Narrative entries ──
    try:
        narrative_count = conn.execute("SELECT COUNT(*) FROM narrative_entries").fetchone()[0]
    except Exception:
        narrative_count = 0

    # ── Prospective memories ──
    try:
        triggers_active = conn.execute(
            "SELECT COUNT(*) FROM prospective_memories WHERE is_active = 1"
        ).fetchone()[0]
        triggers_fired = conn.execute(
            "SELECT SUM(triggered_count) FROM prospective_memories"
        ).fetchone()[0] or 0
    except Exception:
        triggers_active = triggers_fired = 0

    # ── Top tags ──
    tag_counts: dict[str, int] = {}
    for row in conn.execute(f"SELECT tags FROM memories {where}", params).fetchall():
        try:
            for tag in json.loads(row[0] or "[]"):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        except Exception:
            pass
    top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:10]

    conn.close()

    # ── Output ──
    if args.format == "json":
        data = {
            "total": total, "active": active, "stale": stale, "archived": archived,
            "protected": protected, "episodic": episodic, "semantic": semantic,
            "compression": {"raw": comp_0, "gist": comp_1, "tag": comp_2},
            "heat": {"min": heat_min, "avg": heat_avg, "max": heat_max,
                     "buckets": {b[0]: b[1] for b in heat_buckets}},
            "access": {"total": total_accesses, "avg": avg_accesses,
                       "max": max_accesses, "useful": total_useful,
                       "never_accessed": never_accessed},
            "temporal": {"oldest": oldest, "newest": newest,
                         "last_accessed": last_accessed, "age_days": age_days},
            "consolidation": {"total": total_consolidations,
                              "avg_duration_ms": avg_duration},
            "knowledge_graph": {"entities": entity_count, "relationships": rel_count,
                                "causal_edges": causal_edges},
            "action_log": {"total": action_total, "unprocessed": action_unprocessed},
            "clusters": cluster_count, "narratives": narrative_count,
            "triggers": {"active": triggers_active, "fired": triggers_fired},
            "top_tags": dict(top_tags),
        }
        if project_rows:
            data["projects"] = [
                {"directory": r[0], "count": r[1], "avg_heat": round(r[2], 4),
                 "last_created": r[3]}
                for r in project_rows
            ]
        print(json.dumps(data, indent=2))
        return

    # Human-readable table output
    header = f"=== Zikkaron Stats{f' — {project}' if project else ''} ==="
    print(header)
    print()

    print("MEMORIES")
    print(f"  Total:     {total}")
    print(f"  Active:    {active}")
    print(f"  Stale:     {stale}")
    print(f"  Archived:  {archived}")
    print(f"  Protected: {protected}")
    print()

    print("TYPES")
    print(f"  Episodic:  {episodic}")
    print(f"  Semantic:  {semantic}")
    print(f"  Raw:       {comp_0}  |  Gist: {comp_1}  |  Tag: {comp_2}")
    print()

    print("HEAT")
    print(f"  Min: {heat_min:.4f}  |  Avg: {heat_avg:.4f}  |  Max: {heat_max:.4f}")
    for label, count in heat_buckets:
        bar = "#" * min(count, 40)
        print(f"  {label:20s} {count:5d}  {bar}")
    print()

    print("ACCESS")
    print(f"  Total recalls:   {total_accesses}")
    print(f"  Avg per memory:  {avg_accesses:.1f}")
    print(f"  Max on a single: {max_accesses}")
    print(f"  Rated useful:    {total_useful}")
    print(f"  Never accessed:  {never_accessed}")
    print()

    print("TEMPORAL")
    if age_days is not None:
        print(f"  Memory span:     {age_days} days")
    print(f"  Oldest:          {oldest or 'n/a'}")
    print(f"  Newest:          {newest or 'n/a'}")
    print(f"  Last accessed:   {last_accessed or 'n/a'}")
    print()

    if project_rows:
        print("PROJECTS (top 15)")
        for r in project_rows:
            print(f"  {r[1]:5d} memories  heat={r[2]:.4f}  {r[0]}")
        print()

    print("CONSOLIDATION")
    print(f"  Total cycles:    {total_consolidations}")
    print(f"  Avg duration:    {avg_duration:.0f}ms")
    if last_consol:
        print(f"  Last run:        {last_consol['timestamp']}")
        print(f"    Added: {last_consol['memories_added']}  Archived: {last_consol['memories_archived']}  Duration: {last_consol['duration_ms']}ms")
    print()

    print("KNOWLEDGE GRAPH")
    print(f"  Entities:        {entity_count}")
    print(f"  Relationships:   {rel_count}")
    print(f"  Causal edges:    {causal_edges}")
    print()

    print("SUBSYSTEMS")
    print(f"  Clusters:        {cluster_count}")
    print(f"  Narratives:      {narrative_count}")
    print(f"  Active triggers: {triggers_active}  (fired {triggers_fired} times)")
    print(f"  Action log:      {action_total} total, {action_unprocessed} unprocessed")
    print()

    if top_tags:
        print("TOP TAGS")
        for tag, count in top_tags:
            print(f"  {count:5d}  {tag}")
        print()


def cmd_seed(args):
    """Bootstrap memory for an existing project by scanning its structure."""
    import json
    from zikkaron.seed import seed_project

    directory = str(Path(args.directory).resolve())
    print(f"Seeding project: {directory}", file=sys.stderr)

    result = seed_project(
        directory=directory,
        db_path=args.db_path,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(f"\n[DRY RUN] Would create {result['memories_generated']} memories for {result['project']}\n", file=sys.stderr)
        for mem in result.get("memories", []):
            tags = ", ".join(mem["tags"])
            print(f"  [{tags}] {mem['content'][:120]}...", file=sys.stderr)
    else:
        replaced_msg = f", replaced {result['replaced']} old" if result.get('replaced') else ""
        print(
            f"\nSeeded {result['project']}: "
            f"{result['created']} created{replaced_msg} "
            f"(from {result['memories_generated']} total)",
            file=sys.stderr,
        )

    print(json.dumps(result))


def cli():
    parser = argparse.ArgumentParser(description="Zikkaron memory engine MCP server")
    subparsers = parser.add_subparsers(dest="command")

    # Default server mode (no subcommand)
    parser.add_argument("--port", type=int, default=None, help="Server port (default: 8742)")
    parser.add_argument("--db-path", type=str, default=None, help="SQLite database path")
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=VALID_TRANSPORTS,
        help="MCP transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress startup banner",
    )

    # drain subcommand
    drain_parser = subparsers.add_parser("drain", help="Pre-compaction context drain")
    drain_parser.add_argument("directory", help="Project directory")
    drain_parser.add_argument("--db-path", type=str, default=None, help="SQLite database path")

    # restore subcommand
    restore_parser = subparsers.add_parser("restore", help="Post-compaction context restore")
    restore_parser.add_argument("directory", help="Project directory")
    restore_parser.add_argument("--db-path", type=str, default=None, help="SQLite database path")

    # capture subcommand (used by PostToolCall hooks)
    capture_parser = subparsers.add_parser("capture", help="Lightweight action capture")
    capture_parser.add_argument("--tool", dest="tool_name", required=True, help="Tool name")
    capture_parser.add_argument("--summary", type=str, default="", help="Tool input summary")
    capture_parser.add_argument("--directory", type=str, default="", help="Working directory")
    capture_parser.add_argument("--session", type=str, default="", help="Session ID")
    capture_parser.add_argument("--db-path", type=str, default=None, help="SQLite database path")

    # context subcommand (used by SessionStart hooks)
    context_parser = subparsers.add_parser("context", help="Lightweight context query")
    context_parser.add_argument("directory", help="Project directory")
    context_parser.add_argument("--db-path", type=str, default=None, help="SQLite database path")

    # stats subcommand
    stats_parser = subparsers.add_parser("stats", help="Show detailed memory statistics")
    stats_parser.add_argument("--project", type=str, default=None, help="Filter to a specific project directory")
    stats_parser.add_argument("--db-path", type=str, default=None, help="SQLite database path")
    stats_parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format (default: table)")

    # seed subcommand
    seed_parser = subparsers.add_parser("seed", help="Bootstrap memory for an existing project")
    seed_parser.add_argument("directory", help="Project directory to scan and seed")
    seed_parser.add_argument("--db-path", type=str, default=None, help="SQLite database path")
    seed_parser.add_argument("--dry-run", action="store_true", help="Scan and show what would be stored without storing")

    args = parser.parse_args()

    if args.command == "drain":
        cmd_drain(args)
    elif args.command == "restore":
        cmd_restore(args)
    elif args.command == "capture":
        cmd_capture(args)
    elif args.command == "context":
        cmd_context(args)
    elif args.command == "seed":
        cmd_seed(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        # Default: run MCP server
        if not args.quiet and args.transport != "stdio":
            print(STARTUP_BANNER, file=sys.stderr)
            print(f"Transport: {args.transport}", file=sys.stderr)
            if args.port:
                print(f"Port: {args.port}", file=sys.stderr)
            if args.db_path:
                print(f"Database: {args.db_path}", file=sys.stderr)
            print(file=sys.stderr)

        main(port=args.port, db_path=args.db_path, transport=args.transport)


if __name__ == "__main__":
    cli()
