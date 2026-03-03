"""Active memory curation engine — deduplication, merging, contradiction detection, and self-improvement."""

import json
import logging
import re

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.storage import StorageEngine
from zikkaron.thermodynamics import MemoryThermodynamics

logger = logging.getLogger(__name__)

# Negation patterns for contradiction detection
_NEGATION_RE = re.compile(
    r"\b(not|don't|doesn't|didn't|won't|can't|cannot|isn't|aren't|wasn't|weren't|"
    r"no longer|instead of|rather than|replaced|switched from|stopped using|"
    r"removed|deprecated|dropped|never)\b",
    re.IGNORECASE,
)

# Verb extraction for entity-action comparison
_ACTION_RE = re.compile(
    r"\b(use|using|uses|prefer|prefers|run|runs|running|install|installed|"
    r"deploy|deployed|enable|enabled|disable|disabled|add|added|remove|removed|"
    r"switch|switched|migrate|migrated|choose|chose|set|configured)\b",
    re.IGNORECASE,
)

# Moderate similarity range for linking
_LINK_LOW = 0.6
_LINK_HIGH = 0.85


class MemoryCurator:
    """Active memory curation on ingestion and self-improvement during consolidation.

    Implements:
    - Merge/link/create decisions on remember
    - Contradiction detection
    - Memify self-improvement cycle (prune, strengthen, reweight, derive)
    """

    def __init__(
        self,
        storage: StorageEngine,
        embeddings: EmbeddingEngine,
        thermodynamics: MemoryThermodynamics,
        settings: Settings,
    ) -> None:
        self._storage = storage
        self._embeddings = embeddings
        self._thermo = thermodynamics
        self._settings = settings

    # ── a. Active Curation on Ingestion ──────────────────────────────────

    def curate_on_remember(
        self,
        content: str,
        context: str,
        tags: list[str],
        embedding: bytes,
        *,
        initial_heat: float = 1.0,
        surprise: float = 0.0,
        importance: float = 0.5,
        valence: float = 0.0,
        file_hash: str | None = None,
        embedding_model: str | None = None,
        contextual_prefix: str | None = None,
    ) -> dict:
        """Decide whether to merge, link, or create a new memory.

        Returns dict with "action" key: "merged", "linked", or "created".
        """
        threshold = self._settings.CURATION_SIMILARITY_THRESHOLD

        # Search existing memories for similar content
        similar = self._find_similar_memories(embedding, min_sim=_LINK_LOW)

        # Check for high similarity -> merge (requires textual overlap too)
        for mem_id, sim in similar:
            if sim >= threshold:
                existing = self._storage.get_memory(mem_id)
                if existing and self._has_textual_overlap(content, existing["content"]):
                    return self._merge_memory(
                        mem_id, content, tags, embedding, contextual_prefix
                    )

        # Check for moderate similarity -> link
        for mem_id, sim in similar:
            if _LINK_LOW <= sim < threshold:
                new_id = self._insert_new_memory(
                    content, context, tags, embedding, initial_heat,
                    file_hash, embedding_model, contextual_prefix,
                    surprise, importance, valence,
                )
                self._create_link(new_id, mem_id)
                return {"action": "linked", "memory_id": new_id, "linked_to": mem_id}

        # No similar memory -> create new
        new_id = self._insert_new_memory(
            content, context, tags, embedding, initial_heat,
            file_hash, embedding_model, contextual_prefix,
            surprise, importance, valence,
        )
        return {"action": "created", "memory_id": new_id}

    def _find_similar_memories(
        self, embedding: bytes, min_sim: float = 0.6
    ) -> list[tuple[int, float]]:
        """Find existing memories above min_sim, sorted by descending similarity."""
        if embedding is None:
            return []

        vec_hits = self._storage.search_vectors(
            embedding, top_k=10, min_heat=0.0
        )
        results = []
        for mid, distance in vec_hits:
            mem = self._storage.get_memory(mid)
            if mem and mem.get("embedding"):
                sim = self._embeddings.similarity(embedding, mem["embedding"])
                if sim >= min_sim:
                    results.append((mid, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def _has_textual_overlap(new_content: str, existing_content: str) -> bool:
        """Check if new content has meaningful textual overlap with existing.

        Prevents merging memories that are semantically similar (high embedding
        similarity) but clearly different pieces of information.
        """
        new_words = set(new_content.lower().split())
        existing_words = set(existing_content.lower().split())
        if not new_words or not existing_words:
            return False
        intersection = new_words & existing_words
        union = new_words | existing_words
        jaccard = len(intersection) / len(union) if union else 0.0
        return jaccard > 0.5

    def _merge_memory(
        self,
        existing_id: int,
        new_content: str,
        new_tags: list[str],
        new_embedding: bytes,
        contextual_prefix: str | None,
    ) -> dict:
        """Merge new content into an existing memory."""
        existing = self._storage.get_memory(existing_id)
        if existing is None:
            # Shouldn't happen, but fall through to create
            return {"action": "created", "memory_id": existing_id}

        # Combine content
        merged_content = existing["content"] + "\n" + new_content

        # Union tags
        existing_tags = existing.get("tags", [])
        if isinstance(existing_tags, str):
            existing_tags = json.loads(existing_tags)
        merged_tags = list(set(existing_tags) | set(new_tags))

        # Re-embed merged content
        embed_text = f"{contextual_prefix}{merged_content}" if contextual_prefix else merged_content
        merged_embedding = self._embeddings.encode(embed_text)

        # Update memory in DB
        self._storage._conn.execute(
            "UPDATE memories SET content = ?, tags = ?, heat = 1.0, "
            "last_accessed = ? WHERE id = ?",
            (merged_content, json.dumps(merged_tags), self._storage._now_iso(), existing_id),
        )
        self._storage._conn.commit()

        # Update embedding in vec0
        if merged_embedding is not None:
            self._storage._conn.execute(
                "UPDATE memories SET embedding = ? WHERE id = ?",
                (merged_embedding, existing_id),
            )
            self._storage._conn.commit()
            try:
                self._storage.update_vector(existing_id, merged_embedding)
            except Exception:
                pass

        # Update FTS content is handled by the trigger on memories table

        logger.info("Merged new content into memory %d", existing_id)
        return {"action": "merged", "memory_id": existing_id}

    def _insert_new_memory(
        self,
        content: str,
        context: str,
        tags: list[str],
        embedding: bytes,
        heat: float,
        file_hash: str | None,
        embedding_model: str | None,
        contextual_prefix: str | None,
        surprise: float,
        importance: float,
        valence: float,
    ) -> int:
        """Insert a brand-new memory and set its scores."""
        memory_id = self._storage.insert_memory({
            "content": content,
            "embedding": embedding,
            "tags": tags,
            "directory_context": context,
            "heat": heat,
            "is_stale": False,
            "file_hash": file_hash,
            "embedding_model": embedding_model,
        })

        if contextual_prefix:
            self._storage._conn.execute(
                "UPDATE memories SET contextual_prefix = ? WHERE id = ?",
                (contextual_prefix, memory_id),
            )
            self._storage._conn.commit()

        self._storage.update_memory_scores(
            memory_id,
            surprise_score=surprise,
            importance=importance,
            emotional_valence=valence,
        )

        return memory_id

    def _create_link(self, new_id: int, existing_id: int) -> None:
        """Create a derived_from relationship between two memories via entities."""
        now = self._storage._now_iso()
        # Use entity system: create ephemeral entity nodes for both memories
        # and link them with a derived_from relationship
        src_entity = self._storage.get_entity_by_name(f"memory:{new_id}")
        if src_entity is None:
            src_eid = self._storage.insert_entity(
                {"name": f"memory:{new_id}", "type": "file"}
            )
        else:
            src_eid = src_entity["id"]

        tgt_entity = self._storage.get_entity_by_name(f"memory:{existing_id}")
        if tgt_entity is None:
            tgt_eid = self._storage.insert_entity(
                {"name": f"memory:{existing_id}", "type": "file"}
            )
        else:
            tgt_eid = tgt_entity["id"]

        self._storage.insert_relationship({
            "source_entity_id": src_eid,
            "target_entity_id": tgt_eid,
            "relationship_type": "derived_from",
        })
        logger.info("Linked memory %d -> derived_from -> memory %d", new_id, existing_id)

    # ── b. Contradiction Detection ───────────────────────────────────────

    def detect_contradictions(
        self, new_content: str, new_embedding: bytes
    ) -> list[dict]:
        """Find existing memories that may contradict new_content.

        Returns list of dicts: {"memory_id", "content", "similarity", "reason"}.
        """
        if new_embedding is None:
            return []

        similar = self._find_similar_memories(new_embedding, min_sim=0.7)
        contradictions = []

        new_has_negation = bool(_NEGATION_RE.search(new_content))
        new_actions = set(a.lower() for a in _ACTION_RE.findall(new_content))

        for mem_id, sim in similar:
            mem = self._storage.get_memory(mem_id)
            if mem is None:
                continue

            old_content = mem["content"]
            old_has_negation = bool(_NEGATION_RE.search(old_content))

            # Check 1: one has negation patterns, the other doesn't
            if new_has_negation != old_has_negation:
                contradictions.append({
                    "memory_id": mem_id,
                    "content": old_content,
                    "similarity": sim,
                    "reason": "negation_mismatch",
                })
                # Reduce confidence of old contradicting memory
                old_confidence = mem.get("confidence", 1.0)
                self._storage._conn.execute(
                    "UPDATE memories SET confidence = ? WHERE id = ?",
                    (max(old_confidence - 0.2, 0.1), mem_id),
                )
                self._storage._conn.commit()
                continue

            # Check 2: same entities but different actions
            old_actions = set(a.lower() for a in _ACTION_RE.findall(old_content))
            if new_actions and old_actions and new_actions != old_actions:
                # Only flag if there's meaningful overlap in subject matter
                # (similarity > 0.7 already ensures topical overlap)
                shared = new_actions & old_actions
                if len(shared) < len(new_actions | old_actions) * 0.5:
                    contradictions.append({
                        "memory_id": mem_id,
                        "content": old_content,
                        "similarity": sim,
                        "reason": "action_divergence",
                    })
                    old_confidence = mem.get("confidence", 1.0)
                    self._storage._conn.execute(
                        "UPDATE memories SET confidence = ? WHERE id = ?",
                        (max(old_confidence - 0.1, 0.1), mem_id),
                    )
                    self._storage._conn.commit()

        return contradictions

    # ── c. Memify Self-Improvement Layer ─────────────────────────────────

    def memify_cycle(self) -> dict:
        """Run the full memify self-improvement cycle.

        Returns stats: {pruned, strengthened, reweighted, derived}.
        """
        stats = {"pruned": 0, "strengthened": 0, "reweighted": 0, "derived": 0}

        self._memify_prune(stats)
        self._memify_strengthen(stats)
        self._memify_reweight(stats)
        self._memify_derive(stats)

        logger.info("Memify cycle complete: %s", stats)
        return stats

    def _memify_prune(self, stats: dict) -> None:
        """Delete memories with heat < 0.01 AND confidence < 0.3 AND access_count == 0."""
        rows = self._storage._conn.execute(
            "SELECT id FROM memories WHERE heat < 0.01 AND confidence < 0.3 "
            "AND access_count = 0"
        ).fetchall()

        for row in rows:
            self._storage.delete_memory(row[0])
            stats["pruned"] += 1

    def _memify_strengthen(self, stats: dict) -> None:
        """Boost importance for memories accessed > 5 times with confidence > 0.8."""
        rows = self._storage._conn.execute(
            "SELECT id, importance FROM memories "
            "WHERE access_count > 5 AND confidence > 0.8 AND importance < 1.0"
        ).fetchall()

        for row in rows:
            mem_id = row[0]
            current_importance = row[1] if row[1] is not None else 0.5
            new_importance = min(current_importance + 0.1, 1.0)
            self._storage._conn.execute(
                "UPDATE memories SET importance = ? WHERE id = ?",
                (new_importance, mem_id),
            )
            stats["strengthened"] += 1

        if rows:
            self._storage._conn.commit()

    def _memify_reweight(self, stats: dict) -> None:
        """Adjust relationship weights based on usage patterns.

        Relationships between frequently co-retrieved memories get weight boost.
        Relationships between rarely-used entities get weight decay.
        """
        rows = self._storage._conn.execute(
            "SELECT r.id, r.weight, r.source_entity_id, r.target_entity_id "
            "FROM relationships r"
        ).fetchall()

        for row in rows:
            rel_id, weight, src_id, tgt_id = row[0], row[1], row[2], row[3]
            if weight is None:
                weight = 1.0

            src = self._storage._conn.execute(
                "SELECT heat FROM entities WHERE id = ?", (src_id,)
            ).fetchone()
            tgt = self._storage._conn.execute(
                "SELECT heat FROM entities WHERE id = ?", (tgt_id,)
            ).fetchone()

            if src is None or tgt is None:
                continue

            src_heat = src[0] if src[0] is not None else 0.0
            tgt_heat = tgt[0] if tgt[0] is not None else 0.0
            avg_heat = (src_heat + tgt_heat) / 2.0

            if avg_heat > 0.7 and weight >= 5.0:
                # Both entities are hot AND relationship is established -> boost
                new_weight = weight + 0.5
            elif avg_heat < 0.1:
                # Both entities are cold -> decay relationship
                new_weight = max(weight * 0.9, 0.1)
            else:
                continue

            if abs(new_weight - weight) > 1e-9:
                self._storage._conn.execute(
                    "UPDATE relationships SET weight = ? WHERE id = ?",
                    (new_weight, rel_id),
                )
                stats["reweighted"] += 1

        if stats["reweighted"] > 0:
            self._storage._conn.commit()

    def _memify_derive(self, stats: dict) -> None:
        """Generate synthetic derived-fact memories for high-weight entity pairs."""
        rows = self._storage._conn.execute(
            "SELECT r.source_entity_id, r.target_entity_id, r.weight "
            "FROM relationships r "
            "WHERE r.weight > 5.0 AND r.relationship_type = 'co_occurrence'"
        ).fetchall()

        for row in rows:
            src_id, tgt_id, weight = row[0], row[1], row[2]

            # Check if co-occurrence count is high enough (weight as proxy)
            if weight < 10.0:
                continue

            src_entity = self._storage._conn.execute(
                "SELECT name FROM entities WHERE id = ?", (src_id,)
            ).fetchone()
            tgt_entity = self._storage._conn.execute(
                "SELECT name FROM entities WHERE id = ?", (tgt_id,)
            ).fetchone()

            if src_entity is None or tgt_entity is None:
                continue

            src_name = src_entity[0]
            tgt_name = tgt_entity[0]

            # Check if we already derived a fact for this pair
            derived_content = f"{src_name} and {tgt_name} are frequently modified together"
            existing = self._storage._conn.execute(
                "SELECT id FROM memories WHERE content = ?", (derived_content,)
            ).fetchone()
            if existing:
                continue

            embedding = self._embeddings.encode(derived_content)
            memory_id = self._storage.insert_memory({
                "content": derived_content,
                "embedding": embedding,
                "tags": ["derived", "auto-generated"],
                "directory_context": "system",
                "heat": 0.5,
                "is_stale": False,
                "embedding_model": self._embeddings.get_model_name(),
            })
            self._storage.update_memory_scores(
                memory_id,
                importance=0.6,
                surprise_score=0.0,
                emotional_valence=0.0,
            )
            stats["derived"] += 1
