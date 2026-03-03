"""Memory reconsolidation — memories become labile on retrieval and are rewritten based on context.

Based on:
- Nader et al. (Nature, 2000): Consolidated memories return to labile state on retrieval
- Osan-Tort-Amaral (PLoS ONE, 2011): Three outcomes based on mismatch magnitude
- Morris et al. (2006): Prediction error is NECESSARY for reconsolidation

Three outcomes based on mismatch between stored memory and current context:
- mismatch < low_threshold: Passive retrieval, no change
- low_threshold <= mismatch < high_threshold: RECONSOLIDATE — update memory with current context
- mismatch >= high_threshold: EXTINCTION — archive old, create new
"""

import logging
import os
from datetime import datetime, timezone

import numpy as np

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.storage import StorageEngine

logger = logging.getLogger(__name__)


class ReconsolidationEngine:
    """Implements memory reconsolidation with plasticity/stability dynamics."""

    def __init__(
        self,
        storage: StorageEngine,
        embeddings: EmbeddingEngine,
        settings: Settings,
    ) -> None:
        self._storage = storage
        self._embeddings = embeddings
        self._settings = settings

    def compute_mismatch(
        self, memory: dict, current_context: str, current_directory: str
    ) -> float:
        """Compute multi-signal mismatch between stored memory and current retrieval context.

        Signals:
          1. Embedding distance (weight=0.5): 1.0 - cosine_similarity
          2. Directory distance (weight=0.2): 0.0/0.5/1.0
          3. Temporal distance (weight=0.15): hours since last access, normalized
          4. Tag divergence (weight=0.15): 1.0 - jaccard_similarity
        """
        # Signal 1: Embedding distance
        context_embedding = self._embeddings.encode(current_context)
        memory_embedding = memory.get("embedding")

        if context_embedding is not None and memory_embedding is not None:
            cosine_sim = self._embeddings.similarity(memory_embedding, context_embedding)
            embedding_distance = 1.0 - cosine_sim
        else:
            embedding_distance = 0.5  # neutral fallback

        # Signal 2: Directory distance
        mem_dir = memory.get("directory_context", "")
        if mem_dir == current_directory:
            dir_distance = 0.0
        elif os.path.dirname(mem_dir) == os.path.dirname(current_directory):
            dir_distance = 0.5
        else:
            dir_distance = 1.0

        # Signal 3: Temporal distance
        last_accessed_str = memory.get("last_accessed")
        now = datetime.now(timezone.utc)
        if last_accessed_str:
            if isinstance(last_accessed_str, str):
                try:
                    last_accessed = datetime.fromisoformat(last_accessed_str)
                    if last_accessed.tzinfo is None:
                        last_accessed = last_accessed.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    last_accessed = now
            else:
                last_accessed = last_accessed_str
                if last_accessed.tzinfo is None:
                    last_accessed = last_accessed.replace(tzinfo=timezone.utc)
        else:
            last_accessed = now

        hours_elapsed = (now - last_accessed).total_seconds() / 3600.0
        temporal_distance = min(hours_elapsed / 168.0, 1.0)  # normalize to 1 week

        # Signal 4: Tag divergence (Jaccard distance)
        memory_tags = set(memory.get("tags", []))
        # Extract simple tokens from context as pseudo-entities
        context_tokens = set(current_context.lower().split())
        if memory_tags and context_tokens:
            intersection = len(memory_tags & context_tokens)
            union = len(memory_tags | context_tokens)
            jaccard_sim = intersection / union if union > 0 else 0.0
            tag_divergence = 1.0 - jaccard_sim
        elif not memory_tags and not context_tokens:
            tag_divergence = 0.0
        else:
            tag_divergence = 1.0

        # Weighted sum
        mismatch = (
            0.5 * embedding_distance
            + 0.2 * dir_distance
            + 0.15 * temporal_distance
            + 0.15 * tag_divergence
        )
        return max(0.0, min(1.0, mismatch))

    def should_reconsolidate(self, memory: dict, mismatch: float) -> str:
        """Determine reconsolidation action based on mismatch and memory state.

        Returns:
          "none" — no modification
          "update" — merge new context into existing memory
          "archive" — archive old memory, create new one
        """
        if memory.get("is_protected", False):
            return "none"

        stability = memory.get("stability", 0.0)
        plasticity = memory.get("plasticity", 1.0)

        # Stable memories need more mismatch to trigger reconsolidation
        effective_low = self._settings.RECONSOLIDATION_LOW_THRESHOLD + (stability * 0.2)
        effective_high = self._settings.RECONSOLIDATION_HIGH_THRESHOLD + (stability * 0.1)

        # Recently accessed (high plasticity) memories are MORE susceptible
        if plasticity > 0.5:
            effective_low -= 0.1
            effective_high -= 0.1

        if mismatch < effective_low:
            return "none"
        if mismatch < effective_high:
            return "update"
        return "archive"

    def reconsolidate(
        self, memory_id: int, current_context: str, current_directory: str
    ) -> dict:
        """Run reconsolidation on a single memory given current retrieval context.

        Returns dict with action taken and details.
        """
        memory = self._storage.get_memory(memory_id)
        if memory is None:
            return {"action": "none", "reason": "not_found"}

        mismatch = self.compute_mismatch(memory, current_context, current_directory)
        action = self.should_reconsolidate(memory, mismatch)

        # Always update plasticity on retrieval (memory becomes labile)
        self.update_plasticity(memory_id)

        if action == "none":
            return {
                "action": "none",
                "memory_id": memory_id,
                "mismatch": mismatch,
                "new_content": None,
            }

        now_iso = self._storage._now_iso()
        recon_count = memory.get("reconsolidation_count", 0) + 1

        if action == "update":
            new_content = self._update_memory_content(memory_id, current_context)
            self._storage._conn.execute(
                "UPDATE memories SET reconsolidation_count = ?, last_reconsolidated = ? "
                "WHERE id = ?",
                (recon_count, now_iso, memory_id),
            )
            self._storage._conn.commit()
            return {
                "action": "update",
                "memory_id": memory_id,
                "mismatch": mismatch,
                "new_content": new_content,
            }

        # action == "archive"
        archive_id = self._archive_memory(memory_id, mismatch, "extinction")

        # Create new memory with current context in the same directory
        new_mem_id = self._storage.insert_memory({
            "content": current_context,
            "embedding": self._embeddings.encode(current_context),
            "tags": memory.get("tags", []),
            "directory_context": current_directory,
            "heat": 1.0,
            "is_stale": False,
            "embedding_model": self._embeddings.get_model_name(),
        })

        # Update reconsolidation tracking on the NEW memory
        self._storage._conn.execute(
            "UPDATE memories SET reconsolidation_count = ?, last_reconsolidated = ? "
            "WHERE id = ?",
            (recon_count, now_iso, new_mem_id),
        )
        self._storage._conn.commit()

        return {
            "action": "archive",
            "memory_id": new_mem_id,
            "original_memory_id": memory_id,
            "archive_id": archive_id,
            "mismatch": mismatch,
            "new_content": current_context,
        }

    def update_plasticity(self, memory_id: int) -> float:
        """Spike plasticity on access, applying exponential decay since last update.

        Plasticity decays with half-life of PLASTICITY_HALF_LIFE_HOURS (~6h),
        then spikes by PLASTICITY_SPIKE on each access.
        """
        memory = self._storage.get_memory(memory_id)
        if memory is None:
            return 0.0

        now = datetime.now(timezone.utc)
        last_update_str = memory.get("last_excitability_update")
        created_str = memory.get("created_at")

        # Determine reference time for decay
        ref_time = now
        for ts_str in (last_update_str, created_str):
            if ts_str:
                if isinstance(ts_str, str):
                    try:
                        ref_time = datetime.fromisoformat(ts_str)
                        if ref_time.tzinfo is None:
                            ref_time = ref_time.replace(tzinfo=timezone.utc)
                        break
                    except (ValueError, TypeError):
                        continue
                else:
                    ref_time = ts_str
                    if ref_time.tzinfo is None:
                        ref_time = ref_time.replace(tzinfo=timezone.utc)
                    break

        elapsed_hours = max(0.0, (now - ref_time).total_seconds() / 3600.0)
        half_life = self._settings.PLASTICITY_HALF_LIFE_HOURS

        # Exponential decay: plasticity * 2^(-elapsed / half_life)
        current_plasticity = memory.get("plasticity", 1.0)
        if elapsed_hours > 0 and half_life > 0:
            current_plasticity *= 2 ** (-elapsed_hours / half_life)

        # Spike
        new_plasticity = min(current_plasticity + self._settings.PLASTICITY_SPIKE, 1.0)

        # Persist
        now_iso = now.isoformat()
        self._storage._conn.execute(
            "UPDATE memories SET plasticity = ?, last_excitability_update = ? WHERE id = ?",
            (new_plasticity, now_iso, memory_id),
        )
        self._storage._conn.commit()

        return new_plasticity

    def update_stability(self, memory_id: int, was_useful: bool) -> float:
        """Update stability based on usefulness feedback.

        Useful retrievals increase stability; frequent non-useful retrievals decrease it.
        """
        memory = self._storage.get_memory(memory_id)
        if memory is None:
            return 0.0

        stability = memory.get("stability", 0.0)
        increment = self._settings.STABILITY_INCREMENT

        if was_useful:
            stability = min(stability + increment, 1.0)
        elif memory.get("access_count", 0) > 5:
            stability = max(stability - increment * 0.5, 0.0)

        self._storage._conn.execute(
            "UPDATE memories SET stability = ? WHERE id = ?",
            (stability, memory_id),
        )
        self._storage._conn.commit()

        return stability

    def _archive_memory(self, memory_id: int, mismatch: float, reason: str) -> int:
        """Archive a memory's content before modification or extinction."""
        memory = self._storage.get_memory(memory_id)
        if memory is None:
            return -1

        archive_id = self._storage.insert_archive({
            "original_memory_id": memory_id,
            "content": memory["content"],
            "embedding": memory.get("embedding"),
            "mismatch_score": mismatch,
            "archive_reason": reason,
        })
        return archive_id

    def _update_memory_content(self, memory_id: int, new_context: str) -> str:
        """Merge new context into existing memory content.

        If the merged content exceeds 2000 chars, keep first 500 + last 500
        of old content plus full new context.
        """
        memory = self._storage.get_memory(memory_id)
        if memory is None:
            return new_context

        old_content = memory["content"]

        # Smart merge
        merged = f"{old_content}\n--- Updated context ---\n{new_context}"

        if len(merged) > 2000:
            # Summarize: keep first 500 + last 500 of old + full new
            old_prefix = old_content[:500]
            old_suffix = old_content[-500:] if len(old_content) > 500 else ""
            if old_suffix:
                merged = f"{old_prefix}\n...\n{old_suffix}\n--- Updated context ---\n{new_context}"
            else:
                merged = f"{old_prefix}\n--- Updated context ---\n{new_context}"

        # Re-encode embedding
        new_embedding = self._embeddings.encode(merged)

        # Update in storage
        self._storage._conn.execute(
            "UPDATE memories SET content = ?, embedding = ? WHERE id = ?",
            (merged, new_embedding, memory_id),
        )
        self._storage._conn.commit()

        # Also update sqlite-vec vector
        if new_embedding is not None:
            try:
                self._storage.delete_vector(memory_id)
            except Exception:
                pass
            self._storage.insert_vector(memory_id, new_embedding)

        return merged
