"""Engram allocation — competitive memory slot storage based on excitability.

Implements the Josselyn & Frankland (2007) / Rashid et al. (2016) model:
neurons (slots) compete via CREB-like excitability. High-excitability slots
win the competition and memories stored nearby in time share the same slot,
creating automatic temporal linking with zero explicit logic.
"""

import logging
from datetime import datetime, timezone

from zikkaron.config import Settings
from zikkaron.storage import StorageEngine

logger = logging.getLogger(__name__)


class EngramAllocator:
    """Competitive memory slot allocator with excitability-based temporal linking."""

    def __init__(self, storage: StorageEngine, settings: Settings) -> None:
        self._storage = storage
        self._settings = settings
        self._num_slots = settings.HOPFIELD_MAX_PATTERNS
        self._half_life = settings.EXCITABILITY_HALF_LIFE_HOURS
        self._boost = settings.EXCITABILITY_BOOST
        # Ensure slot table is populated
        self._storage.init_engram_slots(self._num_slots)

    def allocate(self, memory_id: int) -> dict:
        """Allocate a memory to the most excitable slot.

        Returns dict with slot_index, excitability, temporally_linked IDs, and link_count.
        """
        # Find the slot with highest current (decayed) excitability
        best_slot = 0
        best_excitability = -1.0
        all_slots = self._storage.get_all_engram_slots()

        for slot in all_slots:
            exc = self._compute_decayed_excitability(
                slot["excitability"], slot.get("last_activated")
            )
            if exc > best_excitability:
                best_excitability = exc
                best_slot = slot["slot_index"]

        # Get memories already in this slot (these are temporally linked)
        existing_memories = self._storage.get_memories_in_slot(best_slot)
        linked_ids = [m["id"] for m in existing_memories if m["id"] != memory_id]

        # Assign memory to the winning slot
        self._storage.assign_memory_slot(memory_id, best_slot)

        # Boost the winning slot's excitability
        new_excitability = self.boost_excitability(best_slot)

        # Apply lateral inhibition to adjacent slots
        self.apply_lateral_inhibition(best_slot)

        # Update the memory's excitability field to match slot
        now = self._storage._now_iso()
        self._storage._conn.execute(
            "UPDATE memories SET excitability = ?, last_excitability_update = ? WHERE id = ?",
            (new_excitability, now, memory_id),
        )
        self._storage._conn.commit()

        return {
            "slot_index": best_slot,
            "excitability": round(new_excitability, 4),
            "temporally_linked": linked_ids,
            "link_count": len(linked_ids),
        }

    def get_excitability(self, slot_index: int) -> float:
        """Get the current decayed excitability for a slot."""
        slot = self._storage.get_engram_slot(slot_index)
        if slot is None:
            return 0.0
        return self._compute_decayed_excitability(
            slot["excitability"], slot.get("last_activated")
        )

    def boost_excitability(self, slot_index: int) -> float:
        """Boost a slot's excitability by EXCITABILITY_BOOST, capped at 1.0."""
        current = self.get_excitability(slot_index)
        new_exc = min(current + self._boost, 1.0)
        now = self._storage._now_iso()
        self._storage.update_engram_slot(slot_index, new_exc, now)
        return new_exc

    def get_temporally_linked(self, memory_id: int) -> list[int]:
        """Return all other memory IDs in the same slot as this memory."""
        mem = self._storage.get_memory(memory_id)
        if mem is None or mem.get("slot_index") is None:
            return []
        memories = self._storage.get_memories_in_slot(mem["slot_index"])
        return [m["id"] for m in memories if m["id"] != memory_id]

    def apply_lateral_inhibition(self, activated_slot: int) -> None:
        """Reduce excitability of slots within ±2 of the activated slot."""
        inhibition = self._boost * 0.5
        for offset in range(-2, 3):
            if offset == 0:
                continue
            neighbor = activated_slot + offset
            if neighbor < 0 or neighbor >= self._num_slots:
                continue
            current = self.get_excitability(neighbor)
            new_exc = max(current - inhibition, 0.0)
            now = self._storage._now_iso()
            self._storage.update_engram_slot(neighbor, new_exc, now)

    def get_slot_statistics(self) -> dict:
        """Return slot occupancy and excitability statistics."""
        occupancy = self._storage.get_slot_occupancy()
        all_slots = self._storage.get_all_engram_slots()

        excitabilities = []
        for slot in all_slots:
            exc = self._compute_decayed_excitability(
                slot["excitability"], slot.get("last_activated")
            )
            excitabilities.append(exc)

        occupied = len(occupancy)
        avg_exc = sum(excitabilities) / len(excitabilities) if excitabilities else 0.0
        max_exc = max(excitabilities) if excitabilities else 0.0

        return {
            "total_slots": self._num_slots,
            "occupied_slots": occupied,
            "avg_excitability": round(avg_exc, 4),
            "max_excitability": round(max_exc, 4),
            "slot_distribution": occupancy,
        }

    def _compute_decayed_excitability(
        self, stored_excitability: float, last_activated: str | None
    ) -> float:
        """Apply exponential decay to stored excitability based on elapsed time."""
        if last_activated is None or stored_excitability <= 0.0:
            return 0.0
        try:
            last_dt = datetime.fromisoformat(last_activated)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return 0.0
        now = datetime.now(timezone.utc)
        elapsed_hours = (now - last_dt).total_seconds() / 3600.0
        if elapsed_hours < 0:
            elapsed_hours = 0.0
        # Exponential decay with half-life: E(t) = E0 * 2^(-t/half_life)
        decayed = stored_excitability * (2.0 ** (-elapsed_hours / self._half_life))
        return decayed
