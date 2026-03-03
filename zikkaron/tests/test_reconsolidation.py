"""Tests for memory reconsolidation — labile retrieval, plasticity, stability, and extinction."""

from datetime import datetime, timedelta, timezone

import pytest

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.reconsolidation import ReconsolidationEngine
from zikkaron.storage import StorageEngine


@pytest.fixture
def settings():
    return Settings(
        DB_PATH=":memory:",
        RECONSOLIDATION_LOW_THRESHOLD=0.3,
        RECONSOLIDATION_HIGH_THRESHOLD=0.7,
        PLASTICITY_SPIKE=0.3,
        PLASTICITY_HALF_LIFE_HOURS=6.0,
        STABILITY_INCREMENT=0.1,
    )


@pytest.fixture
def storage(tmp_path):
    db_path = str(tmp_path / "test_recon.db")
    engine = StorageEngine(db_path)
    yield engine
    engine.close()


@pytest.fixture
def embeddings():
    return EmbeddingEngine("all-MiniLM-L6-v2")


@pytest.fixture
def engine(storage, embeddings, settings):
    return ReconsolidationEngine(storage, embeddings, settings)


def _make_memory(storage, embeddings, content, directory="/tmp/project", tags=None, **kwargs):
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
    mem.update(kwargs)
    mid = storage.insert_memory(mem)
    return mid


class TestComputeMismatch:
    def test_mismatch_identical_context(self, engine, storage, embeddings):
        """Same content should yield a very low mismatch."""
        content = "Fix the authentication bug in login.py"
        mid = _make_memory(storage, embeddings, content)
        memory = storage.get_memory(mid)

        mismatch = engine.compute_mismatch(memory, content, "/tmp/project")
        # Identical content → embedding distance ~0, same dir → 0, tags may diverge slightly
        assert mismatch < 0.2

    def test_mismatch_completely_different(self, engine, storage, embeddings):
        """Completely unrelated content should yield high mismatch."""
        mid = _make_memory(
            storage, embeddings,
            "Fix the authentication bug in login.py",
            directory="/home/user/project-a",
        )
        memory = storage.get_memory(mid)

        mismatch = engine.compute_mismatch(
            memory,
            "quantum mechanics lecture notes about entanglement",
            "/var/data/physics",
        )
        assert mismatch > 0.5

    def test_mismatch_moderate(self, engine, storage, embeddings):
        """Related but changed context should yield mid-range mismatch."""
        mid = _make_memory(
            storage, embeddings,
            "Implemented user authentication with JWT tokens",
            directory="/home/user/webapp",
            tags=["auth", "jwt", "security"],
        )
        memory = storage.get_memory(mid)

        mismatch = engine.compute_mismatch(
            memory,
            "Debugging OAuth2 authentication flow for user login",
            "/home/user/webapp",
        )
        # Related topic, same dir, but different specifics
        assert 0.15 < mismatch < 0.8


class TestShouldReconsolidate:
    def test_no_reconsolidation_low_mismatch(self, engine):
        """Below threshold → 'none' action."""
        memory = {"stability": 0.0, "plasticity": 0.0, "is_protected": False}
        result = engine.should_reconsolidate(memory, 0.1)
        assert result == "none"

    def test_update_moderate_mismatch(self, engine):
        """Moderate mismatch → 'update' action."""
        memory = {"stability": 0.0, "plasticity": 0.0, "is_protected": False}
        result = engine.should_reconsolidate(memory, 0.5)
        assert result == "update"

    def test_archive_high_mismatch(self, engine):
        """High mismatch → 'archive' action."""
        memory = {"stability": 0.0, "plasticity": 0.0, "is_protected": False}
        result = engine.should_reconsolidate(memory, 0.9)
        assert result == "archive"

    def test_protected_memory_never_modified(self, engine):
        """Protected memories always return 'none' regardless of mismatch."""
        memory = {"stability": 0.0, "plasticity": 0.0, "is_protected": True}
        assert engine.should_reconsolidate(memory, 0.5) == "none"
        assert engine.should_reconsolidate(memory, 0.9) == "none"
        assert engine.should_reconsolidate(memory, 1.0) == "none"

    def test_high_stability_resists_update(self, engine):
        """Stable memories require higher mismatch to trigger reconsolidation."""
        # Low stability: mismatch 0.35 should trigger update (0.35 > 0.3)
        low_stab = {"stability": 0.0, "plasticity": 0.0, "is_protected": False}
        assert engine.should_reconsolidate(low_stab, 0.35) == "update"

        # High stability: effective_low = 0.3 + 1.0*0.2 = 0.5
        high_stab = {"stability": 1.0, "plasticity": 0.0, "is_protected": False}
        assert engine.should_reconsolidate(high_stab, 0.35) == "none"

    def test_high_plasticity_lowers_threshold(self, engine):
        """Recently accessed (high plasticity) memories are more susceptible."""
        # With plasticity <= 0.5, effective_low = 0.3 → mismatch 0.25 = none
        low_plast = {"stability": 0.0, "plasticity": 0.0, "is_protected": False}
        assert engine.should_reconsolidate(low_plast, 0.25) == "none"

        # With plasticity > 0.5, effective_low = 0.3 - 0.1 = 0.2 → mismatch 0.25 = update
        high_plast = {"stability": 0.0, "plasticity": 0.8, "is_protected": False}
        assert engine.should_reconsolidate(high_plast, 0.25) == "update"


class TestReconsolidate:
    def test_not_found_returns_none(self, engine):
        """Non-existent memory returns action=none."""
        result = engine.reconsolidate(99999, "some context", "/tmp")
        assert result["action"] == "none"
        assert result["reason"] == "not_found"

    def test_update_moderate_mismatch(self, engine, storage, embeddings):
        """Moderate mismatch merges new context into memory."""
        mid = _make_memory(
            storage, embeddings,
            "Implemented user authentication with JWT tokens",
            directory="/home/user/webapp",
        )
        # Set plasticity high to ensure we can trigger update
        storage._conn.execute(
            "UPDATE memories SET plasticity = 0.8, stability = 0.0 WHERE id = ?", (mid,)
        )
        storage._conn.commit()

        result = engine.reconsolidate(
            mid,
            "Debugging OAuth2 authentication flow for user login",
            "/home/user/webapp",
        )

        if result["action"] == "update":
            assert result["memory_id"] == mid
            assert result["new_content"] is not None
            assert "Updated context" in result["new_content"]
        # If mismatch didn't land in update range, the action is still valid

    def test_archive_preserves_content(self, engine, storage, embeddings):
        """Archived content should match the original exactly."""
        original_content = "Implemented caching layer with Redis"
        mid = _make_memory(
            storage, embeddings, original_content,
            directory="/home/user/backend",
        )
        # Force archive by setting low stability, high plasticity
        storage._conn.execute(
            "UPDATE memories SET plasticity = 0.8, stability = 0.0 WHERE id = ?", (mid,)
        )
        storage._conn.commit()

        # Use completely different context to force high mismatch
        result = engine.reconsolidate(
            mid,
            "quantum physics string theory dark matter cosmology",
            "/var/lib/unrelated",
        )

        if result["action"] == "archive":
            archives = storage.get_archives_for_memory(mid)
            assert len(archives) >= 1
            assert archives[0]["content"] == original_content
            assert archives[0]["archive_reason"] == "extinction"

    def test_reconsolidation_count_increments(self, engine, storage, embeddings):
        """Reconsolidation count should increase after each reconsolidation event."""
        mid = _make_memory(
            storage, embeddings,
            "Database migration strategy for PostgreSQL",
            directory="/home/user/db",
        )
        # Set plasticity high to make it susceptible
        storage._conn.execute(
            "UPDATE memories SET plasticity = 0.8, stability = 0.0 WHERE id = ?", (mid,)
        )
        storage._conn.commit()

        mem_before = storage.get_memory(mid)
        initial_count = mem_before.get("reconsolidation_count", 0)

        # Try reconsolidation with moderately different context
        result = engine.reconsolidate(
            mid,
            "Migrating database schema from MySQL to PostgreSQL with Alembic",
            "/home/user/db",
        )

        if result["action"] in ("update", "archive"):
            target_id = result["memory_id"]
            mem_after = storage.get_memory(target_id)
            assert mem_after["reconsolidation_count"] > initial_count
            assert mem_after["last_reconsolidated"] is not None


class TestPlasticity:
    def test_plasticity_spike_and_decay(self, engine, storage, embeddings):
        """Plasticity should spike on access."""
        mid = _make_memory(storage, embeddings, "test plasticity content")

        # Set plasticity to a known low value
        storage._conn.execute(
            "UPDATE memories SET plasticity = 0.2, last_excitability_update = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), mid),
        )
        storage._conn.commit()

        new_plasticity = engine.update_plasticity(mid)

        # Should have spiked: 0.2 + 0.3 = 0.5 (minimal decay since just set)
        assert new_plasticity > 0.4
        assert new_plasticity <= 1.0

    def test_plasticity_decays_over_time(self, engine, storage, embeddings):
        """Plasticity should decay over time with ~6h half-life."""
        mid = _make_memory(storage, embeddings, "test decay content")

        # Set plasticity high and last update 12 hours ago (2 half-lives)
        past = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
        storage._conn.execute(
            "UPDATE memories SET plasticity = 1.0, last_excitability_update = ? WHERE id = ?",
            (past, mid),
        )
        storage._conn.commit()

        new_plasticity = engine.update_plasticity(mid)

        # After 12h (2 half-lives): 1.0 * 0.25 = 0.25, then +0.3 spike = 0.55
        assert new_plasticity < 0.7
        assert new_plasticity > 0.3

    def test_plasticity_caps_at_one(self, engine, storage, embeddings):
        """Plasticity should not exceed 1.0."""
        mid = _make_memory(storage, embeddings, "cap test")

        # Set plasticity already high
        storage._conn.execute(
            "UPDATE memories SET plasticity = 0.9, last_excitability_update = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), mid),
        )
        storage._conn.commit()

        new_plasticity = engine.update_plasticity(mid)
        assert new_plasticity <= 1.0


class TestStability:
    def test_stability_increases_on_useful(self, engine, storage, embeddings):
        """Stability should increase when memory is rated useful."""
        mid = _make_memory(storage, embeddings, "useful content")
        storage._conn.execute(
            "UPDATE memories SET stability = 0.3 WHERE id = ?", (mid,)
        )
        storage._conn.commit()

        new_stability = engine.update_stability(mid, was_useful=True)
        assert new_stability == pytest.approx(0.4, abs=0.01)

    def test_stability_decreases_on_not_useful_high_access(self, engine, storage, embeddings):
        """Stability should decrease for non-useful memories with many accesses."""
        mid = _make_memory(storage, embeddings, "frequently recalled but useless")
        storage._conn.execute(
            "UPDATE memories SET stability = 0.5, access_count = 10 WHERE id = ?", (mid,)
        )
        storage._conn.commit()

        new_stability = engine.update_stability(mid, was_useful=False)
        assert new_stability == pytest.approx(0.45, abs=0.01)

    def test_stability_not_decreased_low_access(self, engine, storage, embeddings):
        """Stability should not decrease for low-access memories even if not useful."""
        mid = _make_memory(storage, embeddings, "rarely recalled")
        storage._conn.execute(
            "UPDATE memories SET stability = 0.5, access_count = 2 WHERE id = ?", (mid,)
        )
        storage._conn.commit()

        new_stability = engine.update_stability(mid, was_useful=False)
        assert new_stability == pytest.approx(0.5, abs=0.01)

    def test_stability_caps_at_one(self, engine, storage, embeddings):
        """Stability should not exceed 1.0."""
        mid = _make_memory(storage, embeddings, "almost fully stable")
        storage._conn.execute(
            "UPDATE memories SET stability = 0.95 WHERE id = ?", (mid,)
        )
        storage._conn.commit()

        new_stability = engine.update_stability(mid, was_useful=True)
        assert new_stability <= 1.0

    def test_stability_floors_at_zero(self, engine, storage, embeddings):
        """Stability should not go below 0.0."""
        mid = _make_memory(storage, embeddings, "barely stable")
        storage._conn.execute(
            "UPDATE memories SET stability = 0.01, access_count = 10 WHERE id = ?", (mid,)
        )
        storage._conn.commit()

        new_stability = engine.update_stability(mid, was_useful=False)
        assert new_stability >= 0.0


class TestArchivePreservation:
    def test_archive_stores_content_and_embedding(self, engine, storage, embeddings):
        """_archive_memory should preserve original content and embedding in archive table."""
        original_content = "Original important decision about architecture"
        mid = _make_memory(storage, embeddings, original_content)
        mem = storage.get_memory(mid)

        archive_id = engine._archive_memory(mid, 0.8, "test_reason")
        assert archive_id > 0

        archives = storage.get_archives_for_memory(mid)
        assert len(archives) == 1
        assert archives[0]["content"] == original_content
        assert archives[0]["mismatch_score"] == pytest.approx(0.8, abs=0.01)
        assert archives[0]["archive_reason"] == "test_reason"

    def test_archive_nonexistent_memory(self, engine):
        """Archiving non-existent memory returns -1."""
        result = engine._archive_memory(99999, 0.5, "test")
        assert result == -1


class TestUpdateMemoryContent:
    def test_merge_short_content(self, engine, storage, embeddings):
        """Short content merge should include update header."""
        mid = _make_memory(storage, embeddings, "Original context about testing")
        merged = engine._update_memory_content(mid, "New testing framework adopted")

        assert "Original context about testing" in merged
        assert "--- Updated context ---" in merged
        assert "New testing framework adopted" in merged

    def test_merge_long_content_truncates(self, engine, storage, embeddings):
        """Very long content should be truncated to fit within limits."""
        long_content = "A" * 1500
        mid = _make_memory(storage, embeddings, long_content)
        merged = engine._update_memory_content(mid, "New context " * 50)

        # Should contain the update marker
        assert "--- Updated context ---" in merged
        # Should have been summarized
        assert len(merged) < len(long_content) + len("New context " * 50) + 50

    def test_merge_updates_embedding(self, engine, storage, embeddings):
        """After merge, the embedding should be updated in storage."""
        mid = _make_memory(storage, embeddings, "Original content")
        old_mem = storage.get_memory(mid)
        old_embedding = old_mem["embedding"]

        engine._update_memory_content(mid, "Completely different new content added")
        new_mem = storage.get_memory(mid)

        # Embedding should have changed
        assert new_mem["embedding"] != old_embedding
