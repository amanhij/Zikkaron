"""Dual-store Complementary Learning Systems — fast episodic capture + slow semantic abstraction.

Based on:
- McClelland, McNaughton, O'Reilly (1995): Original CLS theory
- Sun et al. (Nature Neuroscience 26:1438, 2023): Go-CLS — only predictable,
  generalizable patterns transfer to semantic storage
- Tadros et al. (Nature Communications, 2022): Random replay strengthens important memories
"""

import logging
import re
from collections import defaultdict

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.storage import StorageEngine

logger = logging.getLogger(__name__)

# Decision/convention keywords → semantic candidate
_DECISION_KEYWORDS = re.compile(
    r"\b(always|never|prefer|standard|convention|rule|guideline|best practice|"
    r"must|should always|should never)\b",
    re.IGNORECASE,
)

# Architecture/pattern keywords → semantic candidate
_ARCHITECTURE_KEYWORDS = re.compile(
    r"\b(pattern|architecture|design|principle|paradigm|framework|methodology|"
    r"approach|strategy|abstraction|interface|protocol)\b",
    re.IGNORECASE,
)

# Tags that indicate semantic content
_SEMANTIC_TAGS = frozenset({
    "rule", "convention", "preference", "standard", "architecture",
    "principle", "guideline", "best-practice", "design-pattern",
})

# Specific-content indicators (file paths, line numbers, error traces)
_SPECIFIC_INDICATORS = re.compile(
    r"(?:"
    r"(?:\.{0,2}/)?(?:[\w@.-]+/)+[\w@.-]+\.\w+"  # file paths
    r"|line \d+"                                     # line numbers
    r"|Traceback \(most recent call last\)"          # tracebacks
    r"|(?:Error|Exception):\s"                       # error messages
    r"|0x[0-9a-fA-F]+"                               # memory addresses
    r")"
)


class DualStoreCLS:
    """Complementary Learning Systems: episodic (hippocampal) + semantic (neocortical).

    Episodic memories are raw, high-fidelity recordings.
    Semantic memories are abstracted schemas derived from recurring patterns.
    Go-CLS consolidation only promotes patterns that appear CONSISTENTLY
    across multiple sessions — one-off workarounds stay episodic.
    """

    def __init__(
        self,
        storage: StorageEngine,
        embeddings: EmbeddingEngine,
        settings: Settings,
    ) -> None:
        self._storage = storage
        self._embeddings = embeddings
        self._settings = settings

    # ── Classification ────────────────────────────────────────────────────

    def classify_memory(self, content: str, tags: list[str], directory: str) -> str:
        """Classify incoming memory as 'episodic' or 'semantic'.

        Rules for semantic classification:
        - Contains decision keywords (always, never, prefer, standard, convention)
        - Contains architecture keywords (pattern, architecture, design, principle)
        - Tags include semantic indicators (rule, convention, preference, standard)
        - Content describes a general pattern (not specific to one file/line)
        """
        # Check tags first (cheapest)
        tag_set = {t.lower() for t in tags}
        if tag_set & _SEMANTIC_TAGS:
            return "semantic"

        has_decision = bool(_DECISION_KEYWORDS.search(content))
        has_architecture = bool(_ARCHITECTURE_KEYWORDS.search(content))
        has_specific = bool(_SPECIFIC_INDICATORS.search(content))

        # If content has specific indicators (file paths, line numbers, error traces),
        # it's likely a specific instance → episodic, even with keyword matches
        if has_specific:
            return "episodic"

        # General content with decision or architecture keywords → semantic
        if has_decision or has_architecture:
            return "semantic"

        return "episodic"

    # ── Pattern Detection (Go-CLS) ───────────────────────────────────────

    def find_recurring_patterns(
        self, directory: str = None, min_occurrences: int = 3
    ) -> list[dict]:
        """Find clusters of similar episodic memories that recur across sessions.

        Algorithm:
        1. Get all episodic memories (optionally filtered by directory)
        2. Group by embedding similarity (threshold: CLUSTER_SIMILARITY_THRESHOLD)
        3. For each cluster with >= min_occurrences members:
           - Check session diversity (>= 2 different sessions)
           - Check generalizability (>= 2 directories OR same directory)
        4. Return qualifying clusters
        """
        # 1. Get episodic memories
        if directory:
            rows = self._storage._conn.execute(
                "SELECT * FROM memories WHERE store_type = 'episodic' "
                "AND heat > 0 AND embedding IS NOT NULL "
                "AND directory_context = ?",
                (directory,),
            ).fetchall()
        else:
            rows = self._storage._conn.execute(
                "SELECT * FROM memories WHERE store_type = 'episodic' "
                "AND heat > 0 AND embedding IS NOT NULL"
            ).fetchall()

        memories = self._storage._rows_to_dicts(rows)
        if len(memories) < min_occurrences:
            return []

        # 2. Greedy clustering by embedding similarity
        threshold = self._settings.CLUSTER_SIMILARITY_THRESHOLD
        clusters: list[list[dict]] = []
        assigned: set[int] = set()

        for i, mem_a in enumerate(memories):
            if mem_a["id"] in assigned:
                continue
            cluster = [mem_a]
            assigned.add(mem_a["id"])

            for mem_b in memories[i + 1:]:
                if mem_b["id"] in assigned:
                    continue
                sim = self._embeddings.similarity(
                    mem_a["embedding"], mem_b["embedding"]
                )
                if sim >= threshold:
                    cluster.append(mem_b)
                    assigned.add(mem_b["id"])

            clusters.append(cluster)

        # 3. Filter by occurrence count and session/directory diversity
        qualifying = []
        for cluster in clusters:
            if len(cluster) < min_occurrences:
                continue

            # Session diversity: check source_episode_id → session_id
            session_ids = set()
            directories = set()
            for mem in cluster:
                directories.add(mem.get("directory_context", ""))
                ep_id = mem.get("source_episode_id")
                if ep_id is not None:
                    ep_row = self._storage._conn.execute(
                        "SELECT session_id FROM episodes WHERE id = ?", (ep_id,)
                    ).fetchone()
                    if ep_row:
                        session_ids.add(ep_row[0])
                else:
                    # No episode linkage — treat created_at date as session proxy
                    created = mem.get("created_at", "")
                    if isinstance(created, str) and len(created) >= 10:
                        session_ids.add(created[:10])  # date part as proxy

            # Go-CLS: require >= 2 different sessions for generalizability
            if len(session_ids) < 2:
                continue

            qualifying.append({
                "memories": cluster,
                "pattern_summary": self._summarize_cluster(cluster),
                "occurrence_count": len(cluster),
                "session_count": len(session_ids),
                "directories": list(directories),
            })

        return qualifying

    # ── Consistency Check ─────────────────────────────────────────────────

    def check_consistency(self, cluster_memories: list[dict]) -> dict:
        """Verify cluster members don't contradict each other.

        Uses negation-pattern detection similar to curation.detect_contradictions.
        """
        _negation_re = re.compile(
            r"\b(not|don't|doesn't|didn't|won't|can't|cannot|isn't|aren't|"
            r"wasn't|weren't|no longer|instead of|rather than|replaced|"
            r"switched from|stopped using|removed|deprecated|dropped|never)\b",
            re.IGNORECASE,
        )

        contradictions = []
        for i, mem_a in enumerate(cluster_memories):
            content_a = mem_a.get("content", "")
            has_neg_a = bool(_negation_re.search(content_a))

            for mem_b in cluster_memories[i + 1:]:
                content_b = mem_b.get("content", "")
                has_neg_b = bool(_negation_re.search(content_b))

                # One has negation, the other doesn't → potential contradiction
                if has_neg_a != has_neg_b:
                    contradictions.append(
                        f"Conflict between memory {mem_a.get('id')} and {mem_b.get('id')}: "
                        f"negation mismatch"
                    )

        return {
            "consistent": len(contradictions) == 0,
            "contradictions": contradictions,
        }

    # ── Schema Abstraction ────────────────────────────────────────────────

    def abstract_to_schema(self, cluster_memories: list[dict]) -> str:
        """Abstract multiple episodic memories into a semantic schema.

        Extracts common words and entities across all memories,
        then builds a generalized statement.
        """
        if not cluster_memories:
            return ""

        # Collect word frequency across all memories
        word_freq: dict[str, int] = defaultdict(int)
        all_contents: list[str] = []

        for mem in cluster_memories:
            content = mem.get("content", "")
            all_contents.append(content)
            words = set(content.lower().split())
            for w in words:
                # Strip punctuation
                clean = w.strip(".,;:!?()[]{}\"'`")
                if len(clean) > 2:
                    word_freq[clean] += 1

        n_memories = len(cluster_memories)

        # Find words that appear in majority of memories (>= 50%)
        common_threshold = max(n_memories / 2, 2)
        common_words = {
            w for w, count in word_freq.items()
            if count >= common_threshold
        }

        # Remove stop words
        stop_words = {
            "the", "and", "for", "with", "that", "this", "from", "was",
            "were", "are", "has", "had", "have", "been", "will", "would",
            "could", "should", "can", "may", "might", "its", "but", "not",
            "all", "any", "each", "also", "into", "than", "then", "when",
            "which", "who", "how", "what", "where", "there", "here", "does",
        }
        meaningful = common_words - stop_words

        if not meaningful:
            # Fallback: use the shortest memory as representative
            shortest = min(all_contents, key=len)
            return f"Recurring pattern: {shortest}"

        # Build schema from common meaningful words, preserving original order
        # from the first memory's structure
        first_content = all_contents[0].lower()
        ordered_common = []
        for word in first_content.split():
            clean = word.strip(".,;:!?()[]{}\"'`")
            if clean in meaningful and clean not in ordered_common:
                ordered_common.append(clean)

        # Add any remaining common words not in first memory
        for word in meaningful:
            if word not in ordered_common:
                ordered_common.append(word)

        # Construct generalized statement
        key_phrase = " ".join(ordered_common[:15])  # Cap at 15 words

        # Collect common tags across memories
        all_tags: dict[str, int] = defaultdict(int)
        for mem in cluster_memories:
            for tag in mem.get("tags", []):
                if isinstance(tag, str):
                    all_tags[tag] += 1
        common_tags = [t for t, c in all_tags.items() if c >= common_threshold]

        schema = f"Recurring pattern across {n_memories} observations: {key_phrase}"
        if common_tags:
            schema += f" [tags: {', '.join(common_tags[:5])}]"

        return schema

    # ── Go-CLS Consolidation Cycle ────────────────────────────────────────

    def consolidation_cycle(self) -> dict:
        """Run Go-CLS consolidation: promote recurring episodic patterns to semantic.

        Steps:
        1. find_recurring_patterns() across all directories
        2. For each qualifying cluster:
           a. check_consistency() — skip if contradictions found
           b. abstract_to_schema() — generate semantic summary
           c. Create new semantic memory
           d. Link episodic memories to semantic memory (derived_from)
           e. Do NOT delete episodic memories
        3. Return stats
        """
        stats = {
            "patterns_found": 0,
            "promoted": 0,
            "skipped_inconsistent": 0,
            "total_episodic": 0,
            "total_semantic": 0,
        }

        patterns = self.find_recurring_patterns()
        stats["patterns_found"] = len(patterns)

        for pattern in patterns:
            cluster_mems = pattern["memories"]

            # a. Check consistency
            consistency = self.check_consistency(cluster_mems)
            if not consistency["consistent"]:
                stats["skipped_inconsistent"] += 1
                continue

            # b. Abstract to schema
            schema = self.abstract_to_schema(cluster_mems)
            if not schema:
                continue

            # c. Check if we already have a similar semantic memory
            schema_embedding = self._embeddings.encode(schema)
            if schema_embedding is not None:
                existing = self._storage.search_vectors(
                    schema_embedding, top_k=3, min_heat=0.0
                )
                skip = False
                for mid, distance in existing:
                    mem = self._storage.get_memory(mid)
                    if mem and mem.get("store_type") == "semantic":
                        sim = self._embeddings.similarity(
                            schema_embedding, mem["embedding"]
                        )
                        if sim > self._settings.CURATION_SIMILARITY_THRESHOLD:
                            skip = True
                            break
                if skip:
                    continue

            # d. Create semantic memory
            directories = pattern["directories"]
            primary_dir = directories[0] if directories else "system"

            semantic_id = self._storage.insert_memory({
                "content": schema,
                "embedding": schema_embedding,
                "tags": ["semantic", "auto-abstracted"],
                "directory_context": primary_dir,
                "heat": 0.8,
                "is_stale": False,
                "embedding_model": self._embeddings.get_model_name(),
            })

            # Set store_type to semantic
            self._storage._conn.execute(
                "UPDATE memories SET store_type = 'semantic' WHERE id = ?",
                (semantic_id,),
            )
            self._storage._conn.commit()

            # e. Link episodic memories to semantic via derived_from
            for mem in cluster_mems:
                self._create_derived_link(mem["id"], semantic_id)

            stats["promoted"] += 1

        # Count totals
        ep_count = self._storage._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE store_type = 'episodic' AND heat > 0"
        ).fetchone()[0]
        sem_count = self._storage._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE store_type = 'semantic' AND heat > 0"
        ).fetchone()[0]
        stats["total_episodic"] = ep_count
        stats["total_semantic"] = sem_count

        logger.info("CLS consolidation cycle: %s", stats)
        return stats

    # ── Dual-Store Query ──────────────────────────────────────────────────

    def query_dual(
        self, query: str, directory: str, prefer: str = "auto"
    ) -> list[dict]:
        """Query both episodic and semantic stores, merge results.

        prefer: "auto" (query analysis), "episodic", or "semantic"
        """
        # Determine weighting
        if prefer == "auto":
            episodic_weight, semantic_weight = self._auto_weight(query)
        elif prefer == "episodic":
            episodic_weight, semantic_weight = 2.0, 1.0
        elif prefer == "semantic":
            episodic_weight, semantic_weight = 1.0, 2.0
        else:
            episodic_weight, semantic_weight = 1.0, 1.0

        query_embedding = self._embeddings.encode(query)
        if query_embedding is None:
            return []

        # Search both stores
        episodic_results = self._search_store(
            query, query_embedding, "episodic", directory
        )
        semantic_results = self._search_store(
            query, query_embedding, "semantic", directory
        )

        # Score and merge
        scored: dict[int, dict] = {}

        for mem, sim in episodic_results:
            scored[mem["id"]] = {
                "memory": mem,
                "score": sim * episodic_weight,
            }

        for mem, sim in semantic_results:
            if mem["id"] in scored:
                scored[mem["id"]]["score"] += sim * semantic_weight
            else:
                scored[mem["id"]] = {
                    "memory": mem,
                    "score": sim * semantic_weight,
                }

        # Sort by combined score
        ranked = sorted(scored.values(), key=lambda x: x["score"], reverse=True)

        results = []
        for item in ranked:
            mem = item["memory"]
            mem["_dual_score"] = round(item["score"], 4)
            mem.pop("embedding", None)
            results.append(mem)

        return results

    # ── Internal Helpers ──────────────────────────────────────────────────

    def _auto_weight(self, query: str) -> tuple[float, float]:
        """Analyze query to determine episodic vs semantic weighting.

        Specific queries (file names, error messages) → episodic
        General queries (patterns, conventions) → semantic
        """
        has_specific = bool(_SPECIFIC_INDICATORS.search(query))
        has_semantic_kw = bool(
            _DECISION_KEYWORDS.search(query)
            or _ARCHITECTURE_KEYWORDS.search(query)
        )

        if has_specific and not has_semantic_kw:
            return 2.0, 1.0  # episodic bias
        elif has_semantic_kw and not has_specific:
            return 1.0, 2.0  # semantic bias
        else:
            return 1.0, 1.0  # balanced

    def _search_store(
        self,
        query: str,
        query_embedding: bytes,
        store_type: str,
        directory: str,
    ) -> list[tuple[dict, float]]:
        """Search a specific store (episodic or semantic) by embedding similarity."""
        if directory:
            rows = self._storage._conn.execute(
                "SELECT * FROM memories WHERE store_type = ? "
                "AND heat > 0 AND embedding IS NOT NULL "
                "AND directory_context = ?",
                (store_type, directory),
            ).fetchall()
        else:
            rows = self._storage._conn.execute(
                "SELECT * FROM memories WHERE store_type = ? "
                "AND heat > 0 AND embedding IS NOT NULL",
                (store_type,),
            ).fetchall()

        memories = self._storage._rows_to_dicts(rows)

        results = []
        for mem in memories:
            if mem.get("embedding"):
                sim = self._embeddings.similarity(
                    query_embedding, mem["embedding"]
                )
                results.append((mem, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:10]

    def _summarize_cluster(self, cluster: list[dict]) -> str:
        """Generate a brief summary of a cluster of memories."""
        contents = [m.get("content", "")[:100] for m in cluster[:3]]
        return " | ".join(contents)

    def _create_derived_link(self, episodic_id: int, semantic_id: int) -> None:
        """Create a derived_from relationship between an episodic and semantic memory."""
        # Create entity nodes for the memories if they don't exist
        src_name = f"memory:{episodic_id}"
        tgt_name = f"memory:{semantic_id}"

        src_entity = self._storage.get_entity_by_name(src_name)
        if src_entity is None:
            src_eid = self._storage.insert_entity(
                {"name": src_name, "type": "file"}
            )
        else:
            src_eid = src_entity["id"]

        tgt_entity = self._storage.get_entity_by_name(tgt_name)
        if tgt_entity is None:
            tgt_eid = self._storage.insert_entity(
                {"name": tgt_name, "type": "file"}
            )
        else:
            tgt_eid = tgt_entity["id"]

        # Check for existing relationship
        existing = self._storage.get_relationship_between(src_eid, tgt_eid)
        if existing:
            self._storage.reinforce_relationship(existing["id"])
        else:
            self._storage.insert_relationship({
                "source_entity_id": src_eid,
                "target_entity_id": tgt_eid,
                "relationship_type": "derived_from",
            })
