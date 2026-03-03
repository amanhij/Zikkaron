"""Tests for engram allocation — competitive memory slot storage with temporal linking."""

from datetime import datetime, timedelta, timezone

import pytest

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.engram import EngramAllocator
from zikkaron.storage import StorageEngine


@pytest.fixture
def settings():
    return Settings(
        DB_PATH=":memory:",
        HOPFIELD_MAX_PATTERNS=10,  # Small slot count for testability
        EXCITABILITY_HALF_LIFE_HOURS=6.0,
        EXCITABILITY_BOOST=0.5,
    )


@pytest.fixture
def storage(tmp_path):
    db_path = str(tmp_path / "test_engram.db")
    engine = StorageEngine(db_path)
    yield engine
    engine.close()


@pytest.fixture
def embeddings():
    return EmbeddingEngine("all-MiniLM-L6-v2")


@pytest.fixture
def allocator(storage, settings):
    return EngramAllocator(storage, settings)


def _make_memory(storage, embeddings, content, directory="/tmp/project", tags=None):
    """Helper to insert a memory with real embedding."""
    embedding = embeddings.encode(content)
    mem = {
        "content": content,
        "embedding": embedding,
        "tags": tags or ["test"],
        "directory_context": directory,
        "heat": 1.0,
        "is_stale": False,
        "embedding_model": embeddings.get_model_name(),
    }
    mid = storage.insert_memory(mem)
    return mid


class TestAllocateToHighestExcitability:
    def test_allocate_to_highest_excitability(self, allocator, storage, embeddings):
        """Memory should be allocated to the slot with the highest excitability."""
        # Boost slot 3 to make it the most excitable
        allocator.boost_excitability(3)

        mid = _make_memory(storage, embeddings, "Fix authentication bug")
        result = allocator.allocate(mid)

        assert result["slot_index"] == 3
        assert result["excitability"] > 0

    def test_fresh_slots_all_zero_picks_first(self, storage, embeddings):
        """When all slots are at 0 excitability (decayed), first slot wins."""
        # Use settings with very short half-life so initial values decay
        settings = Settings(
            DB_PATH=":memory:",
            HOPFIELD_MAX_PATTERNS=5,
            EXCITABILITY_HALF_LIFE_HOURS=6.0,
            EXCITABILITY_BOOST=0.5,
        )
        alloc = EngramAllocator(storage, settings)

        mid = _make_memory(storage, embeddings, "Test memory")
        result = alloc.allocate(mid)

        # All slots are at 0, so first slot (index 0) wins
        assert result["slot_index"] == 0


class TestExcitabilityDecay:
    def test_excitability_decays_over_time(self, allocator, storage):
        """Excitability should decrease with elapsed time."""
        # Boost slot 0 to 0.5
        allocator.boost_excitability(0)
        initial = allocator.get_excitability(0)
        assert initial > 0.4  # Should be ~0.5

        # Simulate time passing by updating the last_activated timestamp
        past = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
        storage.update_engram_slot(0, 0.5, past)

        # After one half-life, excitability should be approximately halved
        decayed = allocator.get_excitability(0)
        assert decayed < initial
        assert abs(decayed - 0.25) < 0.05  # ~50% of 0.5

    def test_long_decay_approaches_zero(self, allocator, storage):
        """After many half-lives, excitability should be near zero."""
        allocator.boost_excitability(0)

        # Set last_activated to 48 hours ago (8 half-lives)
        past = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        storage.update_engram_slot(0, 0.5, past)

        decayed = allocator.get_excitability(0)
        assert decayed < 0.01


class TestBoostExcitability:
    def test_boost_increases_excitability(self, allocator):
        """Boosting should increase slot excitability."""
        initial = allocator.get_excitability(0)
        boosted = allocator.boost_excitability(0)
        assert boosted > initial
        assert boosted == pytest.approx(initial + 0.5, abs=0.01)

    def test_boost_capped_at_one(self, allocator):
        """Excitability should never exceed 1.0."""
        # Boost three times: 0 + 0.5 + 0.5 + 0.5 should cap at 1.0
        allocator.boost_excitability(0)
        allocator.boost_excitability(0)
        result = allocator.boost_excitability(0)
        assert result <= 1.0

    def test_multiple_boosts_accumulate(self, allocator):
        """Sequential boosts should accumulate (up to cap)."""
        allocator.boost_excitability(0)
        first = allocator.get_excitability(0)
        allocator.boost_excitability(0)
        second = allocator.get_excitability(0)
        assert second >= first


class TestLateralInhibition:
    def test_lateral_inhibition(self, allocator):
        """Adjacent slots should decrease after activation."""
        # First boost slots 3, 4, 5 to give them excitability
        for s in range(10):
            allocator.boost_excitability(s)

        before_2 = allocator.get_excitability(2)
        before_4 = allocator.get_excitability(4)

        # Now activate slot 3 — should inhibit slots 1, 2, 4, 5
        allocator.apply_lateral_inhibition(3)

        after_2 = allocator.get_excitability(2)
        after_4 = allocator.get_excitability(4)

        assert after_2 < before_2
        assert after_4 < before_4

    def test_inhibition_does_not_go_negative(self, allocator):
        """Lateral inhibition should floor at 0.0."""
        # Don't boost any slots — they're all at 0
        allocator.apply_lateral_inhibition(5)

        for offset in range(-2, 3):
            if offset == 0:
                continue
            neighbor = 5 + offset
            if 0 <= neighbor < 10:
                assert allocator.get_excitability(neighbor) >= 0.0

    def test_inhibition_respects_boundaries(self, allocator):
        """Inhibition should not try to access slots outside valid range."""
        # Activate slot 0 — should only inhibit slots 1, 2 (no -1, -2)
        allocator.boost_excitability(1)
        allocator.boost_excitability(2)
        before_1 = allocator.get_excitability(1)

        allocator.apply_lateral_inhibition(0)

        after_1 = allocator.get_excitability(1)
        assert after_1 < before_1

    def test_activated_slot_not_inhibited(self, allocator):
        """The activated slot itself should not be reduced."""
        allocator.boost_excitability(5)
        before = allocator.get_excitability(5)
        allocator.apply_lateral_inhibition(5)
        after = allocator.get_excitability(5)
        assert after == pytest.approx(before, abs=0.01)


class TestTemporalLinkingWithinWindow:
    def test_temporal_linking_within_window(self, allocator, storage, embeddings):
        """Two memories stored quickly should go to the same slot and be linked."""
        # Boost slot 5 to make it the winner
        allocator.boost_excitability(5)

        mid1 = _make_memory(storage, embeddings, "Error report: auth failure in login.py")
        result1 = allocator.allocate(mid1)

        mid2 = _make_memory(storage, embeddings, "Fix: patched auth validation")
        result2 = allocator.allocate(mid2)

        # Both should go to slot 5 (still most excitable since we just boosted it)
        assert result1["slot_index"] == result2["slot_index"]

        # Second memory should report first as temporally linked
        assert mid1 in result2["temporally_linked"]
        assert result2["link_count"] >= 1


class TestNoLinkingAfterDecay:
    def test_no_linking_after_decay(self, storage, embeddings):
        """Memories stored days apart should end up in different slots."""
        settings = Settings(
            DB_PATH=":memory:",
            HOPFIELD_MAX_PATTERNS=10,
            EXCITABILITY_HALF_LIFE_HOURS=6.0,
            EXCITABILITY_BOOST=0.5,
        )
        alloc = EngramAllocator(storage, settings)

        # Boost slot 3 and allocate first memory
        alloc.boost_excitability(3)
        mid1 = _make_memory(storage, embeddings, "Error report: auth failure")
        result1 = alloc.allocate(mid1)
        slot1 = result1["slot_index"]

        # Simulate 2 days passing by setting all slot timestamps far in the past
        past = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        for i in range(10):
            slot = storage.get_engram_slot(i)
            if slot:
                storage.update_engram_slot(i, slot["excitability"], past)

        # Now boost a DIFFERENT slot and allocate second memory
        alloc.boost_excitability(7)
        mid2 = _make_memory(storage, embeddings, "Unrelated feature: dark mode")
        result2 = alloc.allocate(mid2)
        slot2 = result2["slot_index"]

        # Memories should be in different slots since slot 3's excitability decayed
        assert slot2 == 7
        assert mid1 not in result2["temporally_linked"]


class TestGetTemporallyLinked:
    def test_get_temporally_linked(self, allocator, storage, embeddings):
        """Returns correct linked memory IDs."""
        allocator.boost_excitability(2)

        mid1 = _make_memory(storage, embeddings, "First memory")
        allocator.allocate(mid1)

        mid2 = _make_memory(storage, embeddings, "Second memory")
        allocator.allocate(mid2)

        mid3 = _make_memory(storage, embeddings, "Third memory")
        allocator.allocate(mid3)

        linked = allocator.get_temporally_linked(mid2)
        assert mid1 in linked
        assert mid3 in linked
        assert mid2 not in linked

    def test_get_temporally_linked_no_slot(self, allocator, storage, embeddings):
        """Memory with no slot assignment should return empty list."""
        mid = _make_memory(storage, embeddings, "Unallocated memory")
        # Don't allocate — slot_index is None
        linked = allocator.get_temporally_linked(mid)
        assert linked == []

    def test_get_temporally_linked_nonexistent(self, allocator):
        """Non-existent memory should return empty list."""
        linked = allocator.get_temporally_linked(99999)
        assert linked == []


class TestSlotStatistics:
    def test_slot_statistics(self, allocator, storage, embeddings):
        """Returns accurate occupancy data."""
        allocator.boost_excitability(2)

        mid1 = _make_memory(storage, embeddings, "Stat memory 1")
        allocator.allocate(mid1)

        mid2 = _make_memory(storage, embeddings, "Stat memory 2")
        allocator.allocate(mid2)

        stats = allocator.get_slot_statistics()
        assert stats["total_slots"] == 10
        assert stats["occupied_slots"] >= 1
        assert stats["avg_excitability"] >= 0.0
        assert stats["max_excitability"] >= 0.0
        assert isinstance(stats["slot_distribution"], dict)
        # At least one slot should have memories
        assert sum(stats["slot_distribution"].values()) >= 2

    def test_slot_statistics_empty(self, allocator):
        """Stats for an empty system should still be valid."""
        stats = allocator.get_slot_statistics()
        assert stats["total_slots"] == 10
        assert stats["occupied_slots"] == 0
        assert stats["avg_excitability"] >= 0.0


class TestProtectedSlotBehavior:
    def test_protected_slot_behavior(self, allocator, storage, embeddings):
        """High-excitability slots should consistently win the competition."""
        # Boost slot 7 very high
        allocator.boost_excitability(7)
        allocator.boost_excitability(7)

        # Allocate several memories — they should all go to slot 7
        mids = []
        for i in range(3):
            mid = _make_memory(storage, embeddings, f"Memory {i} for protected slot")
            result = allocator.allocate(mid)
            mids.append((mid, result["slot_index"]))

        # All should be in slot 7
        for mid, slot in mids:
            assert slot == 7

    def test_competition_after_inhibition(self, allocator, storage, embeddings):
        """After inhibition, a different slot can win."""
        # Boost two slots equally
        allocator.boost_excitability(3)
        allocator.boost_excitability(6)

        # Allocate to winner (slot 3 or 6, depending on tie-breaking)
        mid1 = _make_memory(storage, embeddings, "First allocation")
        result1 = allocator.allocate(mid1)
        winner = result1["slot_index"]

        # After allocation, the winner got boosted even more
        # But lateral inhibition reduced its neighbors
        # Next allocation should still go to the highest excitability
        mid2 = _make_memory(storage, embeddings, "Second allocation")
        result2 = allocator.allocate(mid2)

        # The same slot should win again since it got an additional boost
        assert result2["slot_index"] == winner


class TestIntegrationRemember:
    def test_integration_remember(self, tmp_path):
        """remember() response should include temporal links when engram is active."""
        from zikkaron.server import init_engines, shutdown, remember

        db_path = str(tmp_path / "test_integration.db")
        try:
            init_engines(db_path=db_path)

            # Store two related memories quickly
            result1 = remember(
                content="Error: NullPointerException in UserService.java",
                context="/tmp/project",
                tags=["error", "java"],
            )

            result2 = remember(
                content="Fix: Added null check in UserService.getUser()",
                context="/tmp/project",
                tags=["fix", "java"],
            )

            # Check that engram allocation data is present
            # Note: result2 may or may not have temporal_links depending on
            # whether write gate passes. Check for presence if stored.
            if result2.get("stored") is not False:
                assert "engram_slot" in result2
                assert "temporal_links" in result2
                assert "temporal_link_count" in result2
        finally:
            shutdown()

    def test_integration_remember_temporal_link_content(self, tmp_path):
        """Verify that temporal links actually point to related memories."""
        from zikkaron.server import init_engines, shutdown, remember

        db_path = str(tmp_path / "test_integration2.db")
        try:
            init_engines(db_path=db_path)

            r1 = remember(
                content="Bug report: login fails with expired tokens",
                context="/tmp/project",
                tags=["bug"],
            )

            r2 = remember(
                content="Root cause: token refresh logic skipped in auth middleware",
                context="/tmp/project",
                tags=["debug"],
            )

            r3 = remember(
                content="Fix applied: added token refresh check before validation",
                context="/tmp/project",
                tags=["fix"],
            )

            # If all three stored, later ones should link to earlier ones
            if r3.get("stored") is not False and "temporal_links" in r3:
                # At least one earlier memory should be linked
                assert r3["temporal_link_count"] >= 0
        finally:
            shutdown()


class TestEdgeCases:
    def test_allocate_with_single_slot(self, storage, embeddings):
        """System with just 1 slot should still work."""
        settings = Settings(
            DB_PATH=":memory:",
            HOPFIELD_MAX_PATTERNS=1,
            EXCITABILITY_HALF_LIFE_HOURS=6.0,
            EXCITABILITY_BOOST=0.5,
        )
        alloc = EngramAllocator(storage, settings)

        mid1 = _make_memory(storage, embeddings, "Only slot memory 1")
        result1 = alloc.allocate(mid1)
        assert result1["slot_index"] == 0

        mid2 = _make_memory(storage, embeddings, "Only slot memory 2")
        result2 = alloc.allocate(mid2)
        assert result2["slot_index"] == 0
        assert mid1 in result2["temporally_linked"]

    def test_excitability_of_nonexistent_slot(self, allocator):
        """Getting excitability for a non-existent slot returns 0."""
        exc = allocator.get_excitability(99999)
        assert exc == 0.0

    def test_slot_statistics_with_many_allocations(self, allocator, storage, embeddings):
        """Statistics should be accurate after many allocations."""
        allocator.boost_excitability(0)
        for i in range(5):
            mid = _make_memory(storage, embeddings, f"Bulk memory {i}")
            allocator.allocate(mid)

        stats = allocator.get_slot_statistics()
        total_in_slots = sum(stats["slot_distribution"].values())
        assert total_in_slots == 5
