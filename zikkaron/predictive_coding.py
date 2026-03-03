"""Predictive coding write gate — stores only prediction errors (surprising information).

The brain maintains a generative model and only encodes prediction errors —
information that violates expectations. For Zikkaron, the "generative model"
is the aggregate of existing memories for a directory context. New observations
that are EXPECTED (low surprisal) are skipped. Only SURPRISING observations
are stored.

References:
  - Friston, "Active Inference" (2020): Free energy minimization drives memory
  - Barron et al. (Progress in Neurobiology, 2020): Hippocampus as prediction error generator
  - Titans (Google, arXiv:2501.00663): Surprise metric drives memory retention in ML
"""

import logging
import re
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.retrieval import HippoRetriever
from zikkaron.storage import StorageEngine

logger = logging.getLogger(__name__)

# Bypass keywords — content matching these is ALWAYS stored regardless of surprisal
_ERROR_BYPASS_RE = re.compile(
    r"\b(error|exception|traceback|failed|bug|crash)\b",
    re.IGNORECASE,
)
_DECISION_BYPASS_RE = re.compile(
    r"\b(decided|chose|switched to|migrated|architecture)\b",
    re.IGNORECASE,
)
_BYPASS_TAGS = frozenset({"important", "critical"})


class PredictiveCodingGate:
    """Write gate that filters incoming memories by surprisal.

    Only stores prediction errors — information that violates the existing
    generative model for a directory context. Boilerplate code changes
    (low surprise) are skipped; novel architectural decisions, unusual bugs,
    and unexpected failures (high surprise) are stored.
    """

    def __init__(
        self,
        storage: StorageEngine,
        embeddings: EmbeddingEngine,
        retriever: HippoRetriever,
        settings: Settings,
    ) -> None:
        self._storage = storage
        self._embeddings = embeddings
        self._retriever = retriever
        self._settings = settings
        self._threshold = settings.WRITE_GATE_THRESHOLD

    # ── Core: Surprisal Computation ──────────────────────────────────────

    def compute_surprisal(
        self, content: str, directory: str, tags: list[str]
    ) -> float:
        """Compute how surprising content is relative to the directory's generative model.

        Combines four signals:
          Signal 1 — Embedding novelty (weight 0.4)
          Signal 2 — Entity novelty (weight 0.25)
          Signal 3 — Temporal novelty (weight 0.2)
          Signal 4 — Structural novelty (weight 0.15)

        Returns a float in [0.0, 1.0] where 1.0 = maximally surprising.
        """
        # Build generative model for this directory
        recent_memories = self._storage.get_memories_for_directory(
            directory, min_heat=0.0
        )
        # Sort by created_at descending, take last 50
        recent_memories.sort(
            key=lambda m: m.get("created_at", ""), reverse=True
        )
        recent_memories = recent_memories[:50]

        if not recent_memories:
            # New directory = somewhat surprising
            return 0.8

        # Signal 1: Embedding novelty (weight 0.4)
        embedding_novelty = self._compute_embedding_novelty(content)

        # Signal 2: Entity novelty (weight 0.25)
        entity_novelty = self._compute_entity_novelty(content, directory)

        # Signal 3: Temporal novelty (weight 0.2)
        temporal_novelty = self._compute_temporal_novelty(content, directory)

        # Signal 4: Structural novelty (weight 0.15)
        structural_novelty = self._compute_structural_novelty(content, directory)

        # Weighted sum
        surprisal = (
            0.40 * embedding_novelty
            + 0.25 * entity_novelty
            + 0.20 * temporal_novelty
            + 0.15 * structural_novelty
        )

        return max(0.0, min(1.0, surprisal))

    def _compute_embedding_novelty(self, content: str) -> float:
        """Signal 1: How novel is this content in embedding space?

        novelty = 1.0 - max_similarity (0.0=identical, 1.0=completely novel)
        """
        query_embedding = self._embeddings.encode(content)
        if query_embedding is None:
            return 0.5

        vec_hits = self._storage.search_vectors(
            query_embedding, top_k=5, min_heat=0.0
        )
        if not vec_hits:
            return 0.8  # No vectors at all = fairly novel

        max_similarity = 0.0
        for mid, _distance in vec_hits:
            mem = self._storage.get_memory(mid)
            if mem and mem.get("embedding"):
                sim = self._embeddings.similarity(
                    query_embedding, mem["embedding"]
                )
                max_similarity = max(max_similarity, sim)

        return max(0.0, min(1.0, 1.0 - max_similarity))

    def _compute_entity_novelty(self, content: str, directory: str) -> float:
        """Signal 2: How many entities in this content are new to the graph?

        entity_novelty = new_entities / total_entities (or 0.5 if no entities)
        """
        # Use knowledge graph entity extraction
        kg = self._retriever._graph
        extracted = kg.extract_entities_typed(content, directory)

        if not extracted:
            return 0.5  # No entities = moderate novelty

        total_entities = len(extracted)
        new_count = 0
        for name, _type, _rel_ctx in extracted:
            existing = self._storage.get_entity_by_name(name)
            if existing is None:
                new_count += 1

        return new_count / total_entities

    def _compute_temporal_novelty(self, content: str, directory: str) -> float:
        """Signal 3: How recently was a related topic discussed?

        Within last hour: 0.1 (recent = expected follow-up)
        1-24h ago: 0.3 (moderate gap)
        >24h or none found: 0.7 (old topic resurfacing = surprising)
        """
        # Collect entity names to check: from extraction AND from existing entities in content
        entity_names_to_check = set()

        # Method 1: Extract entities from content using code patterns
        kg = self._retriever._graph
        extracted = kg.extract_entities_typed(content, directory)
        for name, _type, _rel_ctx in extracted:
            entity_names_to_check.add(name)

        # Method 2: Check which existing entities appear in the content text
        all_entities = self._storage.get_all_entities(
            min_heat=0.0, include_archived=True
        )
        for entity in all_entities:
            if entity["name"] in content and len(entity["name"]) > 1:
                entity_names_to_check.add(entity["name"])

        if not entity_names_to_check:
            return 0.7  # No entities to check = surprising

        # Find most recent memory about any overlapping entity
        now = datetime.now(timezone.utc)
        most_recent_dt = None

        for name in entity_names_to_check:
            # Find memories mentioning this entity
            try:
                rows = self._storage._conn.execute(
                    "SELECT m.created_at FROM memories m "
                    "WHERE m.content LIKE ? AND m.heat > 0 "
                    "ORDER BY m.created_at DESC LIMIT 1",
                    (f"%{name}%",),
                ).fetchall()
                if rows:
                    mem_time_str = rows[0][0]
                    try:
                        mem_dt = datetime.fromisoformat(mem_time_str)
                        if mem_dt.tzinfo is None:
                            mem_dt = mem_dt.replace(tzinfo=timezone.utc)
                        if most_recent_dt is None or mem_dt > most_recent_dt:
                            most_recent_dt = mem_dt
                    except (ValueError, TypeError):
                        pass
            except Exception:
                pass

        if most_recent_dt is None:
            return 0.7  # No recent memory found

        hours_elapsed = (now - most_recent_dt).total_seconds() / 3600.0

        if hours_elapsed < 1.0:
            return 0.1  # Very recent = expected follow-up
        elif hours_elapsed < 24.0:
            return 0.3  # Moderate gap
        else:
            return 0.7  # Old topic resurfacing = surprising

    def _compute_structural_novelty(
        self, content: str, directory: str
    ) -> float:
        """Signal 4: Does this content introduce new relationship types or causal patterns?

        New relationship type in graph: 0.8
        All relationship types already exist: 0.2
        """
        kg = self._retriever._graph
        extracted = kg.extract_entities_typed(content, directory)

        if not extracted:
            return 0.2  # No structure to analyze

        # Collect relationship contexts from extracted entities
        new_rel_contexts = set()
        for _name, _type, rel_context in extracted:
            if rel_context:
                new_rel_contexts.add(rel_context)

        if not new_rel_contexts:
            return 0.2  # No relationship signals

        # Check which relationship types already exist in the directory's subgraph
        existing_rel_types = set()
        rows = self._storage._conn.execute(
            "SELECT DISTINCT r.relationship_type FROM relationships r"
        ).fetchall()
        for row in rows:
            existing_rel_types.add(row[0])

        # Check if any extracted relationship contexts are truly new
        has_new = False
        for rel_ctx in new_rel_contexts:
            if rel_ctx not in existing_rel_types:
                has_new = True
                break

        return 0.8 if has_new else 0.2

    # ── Write Gate Decision ──────────────────────────────────────────────

    def should_store(
        self, content: str, directory: str, tags: list[str]
    ) -> tuple[bool, float, str]:
        """Decide whether to store a memory based on surprisal.

        Returns (should_store, surprisal_score, reason).

        Bypass conditions (always store):
          - Error/exception keywords in content
          - Decision keywords in content
          - Tags contain "important" or "critical"
        """
        content_lower = content.lower()

        # Check bypass conditions FIRST
        if _ERROR_BYPASS_RE.search(content):
            surprisal = self.compute_surprisal(content, directory, tags)
            logger.debug(
                "Write gate BYPASS (error keywords): surprisal=%.3f dir=%s",
                surprisal, directory,
            )
            return (True, surprisal, "bypass_error_keywords")

        if _DECISION_BYPASS_RE.search(content):
            surprisal = self.compute_surprisal(content, directory, tags)
            logger.debug(
                "Write gate BYPASS (decision keywords): surprisal=%.3f dir=%s",
                surprisal, directory,
            )
            return (True, surprisal, "bypass_decision_keywords")

        if _BYPASS_TAGS & set(t.lower() for t in tags):
            surprisal = self.compute_surprisal(content, directory, tags)
            logger.debug(
                "Write gate BYPASS (important/critical tag): surprisal=%.3f dir=%s",
                surprisal, directory,
            )
            return (True, surprisal, "bypass_important_tag")

        # Compute surprisal for gating decision
        surprisal = self.compute_surprisal(content, directory, tags)

        if surprisal >= self._threshold:
            logger.debug(
                "Write gate PASS: surprisal=%.3f >= threshold=%.3f dir=%s",
                surprisal, self._threshold, directory,
            )
            return (True, surprisal, "high_surprisal")
        else:
            logger.debug(
                "Write gate BLOCK: surprisal=%.3f < threshold=%.3f dir=%s",
                surprisal, self._threshold, directory,
            )
            return (False, surprisal, "below_threshold")

    # ── Event Boundary Detection ─────────────────────────────────────────

    def compute_boundary_signal(
        self, content: str, previous_content: str
    ) -> float:
        """Detect event boundaries — transitions between different topics/tasks.

        Encodes both contents, computes similarity.
        boundary_strength = 1.0 - similarity
        If boundary > 0.6, this is a strong topic transition.

        Returns boundary strength in [0.0, 1.0].
        """
        emb_current = self._embeddings.encode(content)
        emb_previous = self._embeddings.encode(previous_content)

        if emb_current is None or emb_previous is None:
            return 0.5  # Can't compute, moderate boundary

        similarity = self._embeddings.similarity(emb_current, emb_previous)
        boundary_strength = max(0.0, min(1.0, 1.0 - similarity))

        if boundary_strength > 0.6:
            logger.debug(
                "Strong topic transition detected: boundary=%.3f",
                boundary_strength,
            )

        return boundary_strength

    # ── Directory Generative Model ───────────────────────────────────────

    def get_directory_model(self, directory: str) -> dict:
        """Build summary of what Zikkaron 'knows' about a directory.

        Returns a dict with:
          - memory_count: number of memories for this directory
          - entity_count: number of unique entities mentioned
          - avg_heat: average heat of directory memories
          - common_tags: most frequent tags
          - recent_topics: recent entity names
          - centroid_embedding: mean of all directory memory embeddings (bytes or None)
        """
        memories = self._storage.get_memories_for_directory(
            directory, min_heat=0.0
        )

        if not memories:
            return {
                "memory_count": 0,
                "entity_count": 0,
                "avg_heat": 0.0,
                "common_tags": [],
                "recent_topics": [],
                "centroid_embedding": None,
            }

        # Memory count
        memory_count = len(memories)

        # Average heat
        avg_heat = sum(m["heat"] for m in memories) / memory_count

        # Common tags
        tag_counter = Counter()
        for m in memories:
            tags = m.get("tags", [])
            if isinstance(tags, str):
                import json
                try:
                    tags = json.loads(tags)
                except (ValueError, TypeError):
                    tags = []
            for tag in tags:
                tag_counter[tag] += 1
        common_tags = [tag for tag, _ in tag_counter.most_common(10)]

        # Entity count and recent topics
        entity_names = set()
        for m in memories:
            content = m.get("content", "")
            entities = self._storage.get_all_entities(
                min_heat=0.0, include_archived=True
            )
            for e in entities:
                if e["name"] in content:
                    entity_names.add(e["name"])

        # Recent topics: entities from most recent memories
        recent_memories = sorted(
            memories, key=lambda m: m.get("created_at", ""), reverse=True
        )[:10]
        recent_entity_names = set()
        all_entities = self._storage.get_all_entities(
            min_heat=0.0, include_archived=True
        )
        for m in recent_memories:
            content = m.get("content", "")
            for e in all_entities:
                if e["name"] in content:
                    recent_entity_names.add(e["name"])

        # Centroid embedding: mean of all memory embeddings
        centroid_embedding = None
        embeddings_list = []
        dim = self._embeddings.get_dimensions()
        for m in memories:
            emb = m.get("embedding")
            if emb is not None:
                arr = np.frombuffer(emb, dtype=np.float32)
                if len(arr) == dim:
                    embeddings_list.append(arr)

        if embeddings_list:
            centroid = np.mean(embeddings_list, axis=0).astype(np.float32)
            centroid_embedding = centroid.tobytes()

        return {
            "memory_count": memory_count,
            "entity_count": len(entity_names),
            "avg_heat": round(avg_heat, 4),
            "common_tags": common_tags,
            "recent_topics": list(recent_entity_names)[:10],
            "centroid_embedding": centroid_embedding,
        }
