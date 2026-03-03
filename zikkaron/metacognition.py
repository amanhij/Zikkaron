"""Metacognition — Zikkaron knowing what it knows and doesn't know.

Implements three capabilities from MetaRAG (Zhou et al., ACM Web 2024)
and Cognitive Workspace (arXiv:2508.13171):

1. Coverage assessment — "Do I have enough knowledge about this topic?"
2. Gap detection — "What don't I know about this project?"
3. Cognitive load management — optimal 4±1 chunk context packing
   with primacy-recency positioning.
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.storage import StorageEngine

logger = logging.getLogger(__name__)


def _extract_entities(query: str) -> list[str]:
    """Re-use retrieval entity extraction without circular import."""
    from zikkaron.retrieval import _extract_query_entities
    return _extract_query_entities(query)


class MetaCognition:
    """Metacognitive layer: coverage assessment, gap detection,
    and cognitive load management (Cowan's 4±1 chunk limit)."""

    def __init__(
        self,
        storage: StorageEngine,
        embeddings: EmbeddingEngine,
        knowledge_graph: KnowledgeGraph,
        settings: Settings,
    ) -> None:
        self._storage = storage
        self._embeddings = embeddings
        self._graph = knowledge_graph
        self._settings = settings
        self._chunk_limit = settings.COGNITIVE_LOAD_LIMIT

    # ── 1. Coverage Assessment ────────────────────────────────────────

    def assess_coverage(self, query: str, directory: str = "") -> dict:
        """Assess how well Zikkaron can answer a query.

        Returns a dict with coverage score, confidence, suggestion,
        identified gaps, and detailed signal breakdowns.
        """
        # a) Memory density via FTS + vector search
        memory_count = 0
        matching_memories = []

        try:
            fts_results = self._storage.search_memories_fts(
                query, min_heat=0.0, limit=50
            )
            if fts_results:
                matching_memories.extend(fts_results)
        except Exception:
            pass

        query_embedding = self._embeddings.encode(query)
        if query_embedding is not None:
            vec_hits = self._storage.search_vectors(
                query_embedding, top_k=50, min_heat=0.0
            )
            seen_ids = {m["id"] for m in matching_memories}
            for mid, _distance in vec_hits:
                if mid not in seen_ids:
                    mem = self._storage.get_memory(mid)
                    if mem:
                        matching_memories.append(mem)
                        seen_ids.add(mid)

        memory_count = len(matching_memories)

        # Density scoring: 0=0.0, 1-2=0.3, 3-5=0.6, 6+=0.9
        if memory_count == 0:
            density = 0.0
        elif memory_count <= 2:
            density = 0.3
        elif memory_count <= 5:
            density = 0.6
        else:
            density = 0.9

        # b) Entity coverage: what fraction of query entities exist in the graph
        query_entities = _extract_entities(query)
        total_query_entities = len(query_entities)
        known_entities = 0
        unknown_entities = []

        for entity_name in query_entities:
            entity = self._storage.get_entity_by_name(entity_name)
            if entity:
                known_entities += 1
            else:
                unknown_entities.append(entity_name)

        if total_query_entities > 0:
            entity_coverage = known_entities / total_query_entities
        else:
            entity_coverage = 0.0

        # c) Recency: age of most recent relevant memory
        recency_score = 0.0
        if matching_memories:
            now = datetime.now(timezone.utc)
            most_recent = None
            for m in matching_memories:
                created = m.get("created_at")
                if created:
                    if isinstance(created, str):
                        try:
                            created = datetime.fromisoformat(created)
                        except (ValueError, TypeError):
                            continue
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    if most_recent is None or created > most_recent:
                        most_recent = created
            if most_recent is not None:
                age_days = (now - most_recent).total_seconds() / 86400
                if age_days < 1:
                    recency_score = 1.0
                elif age_days < 7:
                    recency_score = 0.7
                elif age_days < 30:
                    recency_score = 0.4
                else:
                    recency_score = 0.2

        # d) Confidence: average confidence score of matching memories
        confidence_score = 0.0
        if matching_memories:
            confidences = [
                m.get("confidence", 1.0) for m in matching_memories
            ]
            confidence_score = sum(confidences) / len(confidences)

        # Overall coverage (weighted blend)
        overall = (
            0.3 * density
            + 0.3 * entity_coverage
            + 0.2 * recency_score
            + 0.2 * confidence_score
        )

        # Suggestion
        if overall >= 0.7:
            suggestion = "sufficient"
            detail = "Proceed with confidence — strong knowledge coverage."
        elif overall >= 0.4:
            suggestion = "partial"
            detail = "Answer available but may be incomplete — consider investigating gaps."
        else:
            suggestion = "insufficient"
            detail = "Limited knowledge — investigate further before answering."

        # Identify gaps
        gaps = list(unknown_entities)
        if memory_count == 0:
            gaps.append(f"No memories found matching query: {query[:80]}")

        return {
            "coverage": round(overall, 4),
            "confidence": round(confidence_score, 4),
            "suggestion": suggestion,
            "gaps": gaps,
            "memory_count": memory_count,
            "entity_coverage": round(entity_coverage, 4),
            "recency_score": round(recency_score, 4),
            "detail": detail,
        }

    # ── 2. Gap Detection ──────────────────────────────────────────────

    def detect_gaps(self, directory: str = "") -> list[dict]:
        """Analyze knowledge completeness for a directory/project.

        Returns a list of knowledge gaps with type, description,
        severity, affected entities, and remediation suggestions.
        """
        gaps: list[dict] = []

        # a) Isolated entities: entities with 0 or 1 relationships
        all_entities = self._storage.get_all_entities(
            min_heat=0.0, include_archived=False
        )
        for entity in all_entities:
            neighbors = self._graph._get_adjacent(entity["id"], None)
            if len(neighbors) <= 1:
                gaps.append({
                    "type": "isolated_entity",
                    "description": (
                        f"Entity '{entity['name']}' has only "
                        f"{len(neighbors)} connection(s) — poorly integrated "
                        f"into the knowledge graph."
                    ),
                    "severity": 0.6 if len(neighbors) == 0 else 0.4,
                    "entities": [entity["name"]],
                    "suggestion": (
                        f"Add more context about '{entity['name']}' to "
                        f"strengthen its connections in the knowledge graph."
                    ),
                })

        # b) Stale regions: clusters of memories with heat < 0.3
        if directory:
            dir_memories = self._storage.get_memories_for_directory(
                directory, min_heat=0.0
            )
        else:
            dir_memories = self._storage._rows_to_dicts(
                self._storage._conn.execute(
                    "SELECT * FROM memories WHERE heat > 0"
                ).fetchall()
            )

        stale_memories = [m for m in dir_memories if m.get("heat", 1.0) < 0.3]
        if len(stale_memories) >= 2:
            stale_tags = set()
            for m in stale_memories:
                tags = m.get("tags", [])
                if isinstance(tags, list):
                    stale_tags.update(tags)
            gaps.append({
                "type": "stale_region",
                "description": (
                    f"{len(stale_memories)} memories have decayed below "
                    f"heat 0.3 — knowledge may be outdated."
                ),
                "severity": min(0.9, 0.3 + len(stale_memories) * 0.1),
                "entities": list(stale_tags)[:10],
                "suggestion": (
                    "Review and refresh these memories or validate "
                    "against current project state."
                ),
            })

        # c) Low-confidence zones: memories with confidence < 0.5
        low_conf = [
            m for m in dir_memories
            if m.get("confidence", 1.0) < 0.5
        ]
        if low_conf:
            low_conf_descriptions = []
            for m in low_conf[:5]:
                content_preview = m.get("content", "")[:60]
                low_conf_descriptions.append(content_preview)
            gaps.append({
                "type": "low_confidence",
                "description": (
                    f"{len(low_conf)} memories have confidence below 0.5 "
                    f"— unreliable knowledge detected."
                ),
                "severity": min(0.8, 0.3 + len(low_conf) * 0.1),
                "entities": low_conf_descriptions,
                "suggestion": (
                    "Validate low-confidence memories against "
                    "current source code or documentation."
                ),
            })

        # d) Missing connections: entities that co-occur in content
        #    but have no relationship in the graph
        entity_cooccurrence = defaultdict(set)
        for m in dir_memories:
            content = m.get("content", "")
            entities_in_mem = []
            for entity in all_entities:
                if entity["name"] in content:
                    entities_in_mem.append(entity["id"])
            for i, eid_a in enumerate(entities_in_mem):
                for eid_b in entities_in_mem[i + 1:]:
                    entity_cooccurrence[(eid_a, eid_b)].add(m.get("id"))

        for (eid_a, eid_b), mem_ids in entity_cooccurrence.items():
            if len(mem_ids) < 2:
                continue
            # Check if relationship exists
            existing = self._storage._conn.execute(
                "SELECT id FROM relationships "
                "WHERE (source_entity_id = ? AND target_entity_id = ?) "
                "OR (source_entity_id = ? AND target_entity_id = ?)",
                (eid_a, eid_b, eid_b, eid_a),
            ).fetchone()
            if existing is None:
                name_a = self._storage._conn.execute(
                    "SELECT name FROM entities WHERE id = ?", (eid_a,)
                ).fetchone()
                name_b = self._storage._conn.execute(
                    "SELECT name FROM entities WHERE id = ?", (eid_b,)
                ).fetchone()
                if name_a and name_b:
                    gaps.append({
                        "type": "missing_connection",
                        "description": (
                            f"'{name_a[0]}' and '{name_b[0]}' co-occur in "
                            f"{len(mem_ids)} memories but have no relationship."
                        ),
                        "severity": min(0.7, 0.2 + len(mem_ids) * 0.1),
                        "entities": [name_a[0], name_b[0]],
                        "suggestion": (
                            f"Add a relationship between '{name_a[0]}' and "
                            f"'{name_b[0]}' to capture their connection."
                        ),
                    })

        # e) One-sided knowledge: only errors stored, no solutions (or vice versa)
        error_entities = [
            e for e in all_entities if e.get("type") == "error"
        ]
        solution_entities = [
            e for e in all_entities if e.get("type") == "solution"
        ]
        error_names = {e["name"] for e in error_entities}
        solution_names = {e["name"] for e in solution_entities}

        # Check for "resolved_by" relationships from error entities
        for err_entity in error_entities:
            has_resolution = self._storage._conn.execute(
                "SELECT id FROM relationships "
                "WHERE source_entity_id = ? AND relationship_type = 'resolved_by'",
                (err_entity["id"],),
            ).fetchone()
            if has_resolution is None:
                gaps.append({
                    "type": "one_sided_knowledge",
                    "description": (
                        f"Error '{err_entity['name']}' has no recorded "
                        f"resolution — only the problem is known."
                    ),
                    "severity": 0.5,
                    "entities": [err_entity["name"]],
                    "suggestion": (
                        f"Record how '{err_entity['name']}' was resolved "
                        f"to complete the knowledge."
                    ),
                })

        return gaps

    # ── 3. Cognitive Load Management ──────────────────────────────────

    def manage_context(
        self, memories: list[dict], max_chunks: int | None = None
    ) -> list[dict]:
        """Apply Cowan's 4±1 cognitive load optimization.

        If memories fit within the chunk limit, return as-is.
        Otherwise, group into coherent chunks, rank, select top chunks,
        and apply primacy-recency positioning.
        """
        if max_chunks is None:
            max_chunks = self._chunk_limit

        if not memories:
            return []

        # Step 1: if under limit, return as-is with metadata
        if len(memories) <= max_chunks:
            result = []
            for i, mem in enumerate(memories):
                enriched = dict(mem)
                enriched["_chunk_id"] = i
                enriched["_position_reason"] = "within_limit"
                result.append(enriched)
            return result

        # Step 2: Group related memories into chunks
        chunks = self.chunk_memories(memories)

        # Step 3: Rank chunks by combined importance * heat * confidence
        def _chunk_score(chunk: list[dict]) -> float:
            total = 0.0
            for m in chunk:
                importance = m.get("importance", 0.5)
                heat = m.get("heat", 0.5)
                confidence = m.get("confidence", 1.0)
                total += importance * heat * confidence
            return total / len(chunk) if chunk else 0.0

        scored_chunks = [(i, chunk, _chunk_score(chunk)) for i, chunk in enumerate(chunks)]
        scored_chunks.sort(key=lambda x: x[2], reverse=True)

        # Step 4: Take top max_chunks
        selected = scored_chunks[:max_chunks]
        overflow = scored_chunks[max_chunks:]

        # Step 5: Apply primacy-recency positioning
        positioned = self._apply_primacy_recency(selected)

        # Step 6: Flatten chunks into memory list with metadata
        result = []
        for pos_idx, (chunk_id, chunk, score) in enumerate(positioned):
            reason = self._position_reason(pos_idx, len(positioned))
            for mem in chunk:
                enriched = dict(mem)
                enriched["_chunk_id"] = chunk_id
                enriched["_position_reason"] = reason
                result.append(enriched)

        # Summarize overflow if any
        if overflow:
            overflow_memories = []
            for _, chunk, _ in overflow:
                overflow_memories.extend(chunk)
            summaries = self.summarize_overflow(overflow_memories)
            for summary in summaries:
                summary["_chunk_id"] = -1
                summary["_position_reason"] = "overflow_summary"
                result.append(summary)

        return result

    def chunk_memories(self, memories: list[dict]) -> list[list[dict]]:
        """Group related memories into coherent chunks.

        Uses entity overlap (Jaccard > 0.3) and temporal proximity (< 2h apart).
        Singleton memories that don't cluster stay as individual chunks.
        """
        if not memories:
            return []

        n = len(memories)
        assigned = [False] * n

        # Pre-compute entity sets and timestamps for each memory
        entity_sets: list[set[str]] = []
        timestamps: list[datetime | None] = []

        for mem in memories:
            # Extract entities from content
            entities = set(_extract_entities(mem.get("content", "")))
            # Also add tags
            tags = mem.get("tags", [])
            if isinstance(tags, list):
                entities.update(tags)
            entity_sets.append(entities)

            # Parse timestamp
            created = mem.get("created_at")
            if isinstance(created, str):
                try:
                    created = datetime.fromisoformat(created)
                except (ValueError, TypeError):
                    created = None
            timestamps.append(created)

        chunks: list[list[dict]] = []

        for i in range(n):
            if assigned[i]:
                continue
            chunk = [memories[i]]
            assigned[i] = True

            for j in range(i + 1, n):
                if assigned[j]:
                    continue

                # Check entity overlap (Jaccard > 0.3)
                if entity_sets[i] and entity_sets[j]:
                    intersection = entity_sets[i] & entity_sets[j]
                    union = entity_sets[i] | entity_sets[j]
                    jaccard = len(intersection) / len(union) if union else 0.0
                    if jaccard > 0.3:
                        chunk.append(memories[j])
                        assigned[j] = True
                        continue

                # Check temporal proximity (< 2 hours apart)
                if timestamps[i] is not None and timestamps[j] is not None:
                    t_i = timestamps[i]
                    t_j = timestamps[j]
                    if t_i.tzinfo is None:
                        t_i = t_i.replace(tzinfo=timezone.utc)
                    if t_j.tzinfo is None:
                        t_j = t_j.replace(tzinfo=timezone.utc)
                    diff = abs((t_i - t_j).total_seconds())
                    if diff < 7200:  # 2 hours
                        chunk.append(memories[j])
                        assigned[j] = True
                        continue

            chunks.append(chunk)

        return chunks

    def summarize_overflow(
        self, excess_memories: list[dict], target_count: int = 1
    ) -> list[dict]:
        """Compress multiple low-priority memories into summary chunks.

        Preserves verbatim: high-surprise (>0.7) and high-importance (>0.7).
        Summarizes remaining memories into brief summaries.
        """
        if not excess_memories:
            return []

        # Separate high-value memories that should be preserved
        preserved = []
        to_summarize = []

        for mem in excess_memories:
            surprise = mem.get("surprise_score", 0.0)
            importance = mem.get("importance", 0.5)
            if surprise > 0.7 or importance > 0.7:
                preserved.append(mem)
            else:
                to_summarize.append(mem)

        result = list(preserved)

        # Compress remaining memories into summaries
        if to_summarize:
            # Group into target_count summary chunks
            chunk_size = max(1, len(to_summarize) // max(1, target_count))
            for start in range(0, len(to_summarize), chunk_size):
                batch = to_summarize[start:start + chunk_size]
                if not batch:
                    continue

                # Build summary from content snippets
                snippets = []
                for m in batch:
                    content = m.get("content", "")
                    # Truncate to first 80 chars
                    snippet = content[:80].strip()
                    if len(content) > 80:
                        snippet += "..."
                    snippets.append(snippet)

                summary_content = (
                    f"[Summary of {len(batch)} memories] "
                    + " | ".join(snippets)
                )

                # Aggregate metadata
                avg_heat = sum(m.get("heat", 0.5) for m in batch) / len(batch)
                avg_importance = sum(m.get("importance", 0.5) for m in batch) / len(batch)
                avg_confidence = sum(m.get("confidence", 1.0) for m in batch) / len(batch)

                result.append({
                    "content": summary_content,
                    "heat": avg_heat,
                    "importance": avg_importance,
                    "confidence": avg_confidence,
                    "surprise_score": 0.0,
                    "tags": [],
                    "_is_summary": True,
                    "_summarized_count": len(batch),
                    "_source_ids": [m.get("id") for m in batch if m.get("id")],
                })

        return result

    # ── Internal helpers ──────────────────────────────────────────────

    def _apply_primacy_recency(
        self, scored_chunks: list[tuple[int, list[dict], float]]
    ) -> list[tuple[int, list[dict], float]]:
        """Position chunks for primacy-recency effect.

        - Highest importance → position 0 (primacy, first 20%)
        - Second highest → last position (recency, last 10%)
        - Others → middle positions by descending importance
        """
        if len(scored_chunks) <= 1:
            return list(scored_chunks)

        if len(scored_chunks) == 2:
            # First = highest, second = next highest at end
            return list(scored_chunks)

        # Already sorted by score descending
        first = scored_chunks[0]   # primacy
        last = scored_chunks[1]    # recency (second highest)
        middle = scored_chunks[2:]  # rest in middle

        return [first] + middle + [last]

    @staticmethod
    def _position_reason(position: int, total: int) -> str:
        """Return a human-readable reason for a chunk's position."""
        if position == 0:
            return "primacy"
        if position == total - 1 and total > 1:
            return "recency"
        return "middle"
