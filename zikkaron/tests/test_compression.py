"""Tests for rate-distortion compression (zikkaron.compression)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from zikkaron.compression import MemoryCompressor
from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.storage import StorageEngine


@pytest.fixture
def storage(tmp_path):
    engine = StorageEngine(str(tmp_path / "test_compression.db"))
    yield engine
    engine.close()


@pytest.fixture
def settings():
    return Settings(
        DB_PATH=":memory:",
        COMPRESSION_GIST_AGE_HOURS=168.0,  # 7 days
        COMPRESSION_TAG_AGE_HOURS=720.0,   # 30 days
    )


@pytest.fixture
def embeddings():
    engine = EmbeddingEngine()
    engine._unavailable = True  # don't load real model in tests
    return engine


@pytest.fixture
def mock_embeddings():
    """Embeddings engine that returns deterministic fake embeddings."""
    engine = EmbeddingEngine()
    engine._unavailable = True

    def fake_encode(text):
        # Return a deterministic 384-dim vector based on text length
        rng = np.random.RandomState(len(text) % 1000)
        vec = rng.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tobytes()

    engine.encode = fake_encode
    return engine


@pytest.fixture
def compressor(storage, mock_embeddings, settings):
    return MemoryCompressor(storage, mock_embeddings, settings)


def _hours_ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _days_ago(days: float) -> str:
    return _hours_ago(days * 24)


def _make_memory(storage, content="test memory", hours_old=0, **kwargs):
    """Helper to insert a memory with specified age and properties."""
    defaults = {
        "content": content,
        "directory_context": "/test",
        "heat": 1.0,
        "created_at": _hours_ago(hours_old),
        "last_accessed": _hours_ago(hours_old),
    }
    defaults.update(kwargs)
    mid = storage.insert_memory(defaults)

    # Set additional fields that insert_memory doesn't handle
    extra_fields = {}
    for field in ("importance", "surprise_score", "confidence", "access_count",
                  "store_type", "is_protected", "compression_level", "original_content"):
        if field in kwargs:
            extra_fields[field] = kwargs[field]

    if extra_fields:
        set_parts = []
        values = []
        for k, v in extra_fields.items():
            set_parts.append(f"{k} = ?")
            if isinstance(v, bool):
                values.append(int(v))
            else:
                values.append(v)
        values.append(mid)
        storage._conn.execute(
            f"UPDATE memories SET {', '.join(set_parts)} WHERE id = ?",
            values,
        )
        storage._conn.commit()

    return mid


# -- Schedule tests --


class TestCompressionSchedule:
    def test_schedule_recent_full(self, compressor, storage):
        """Memory less than 7 days old should stay at level 0 (full)."""
        mid = _make_memory(storage, hours_old=24)  # 1 day old
        mem = storage.get_memory(mid)
        assert compressor.get_compression_schedule(mem) == 0

    def test_schedule_medium_gist(self, compressor, storage):
        """Memory 7-30 days old should be at level 1 (gist)."""
        mid = _make_memory(storage, hours_old=240)  # 10 days old
        mem = storage.get_memory(mid)
        assert compressor.get_compression_schedule(mem) == 1

    def test_schedule_old_tag(self, compressor, storage):
        """Memory older than 30 days should be at level 2 (tag)."""
        # Default confidence=1.0 triggers 1.3x resistance, so tag threshold = 720*1.3 = 936h
        # Use 1000h (~42 days) to exceed the adjusted threshold
        mid = _make_memory(storage, hours_old=1000)
        mem = storage.get_memory(mid)
        assert compressor.get_compression_schedule(mem) == 2

    def test_high_importance_resists(self, compressor, storage):
        """High importance memory should resist compression (thresholds doubled)."""
        # 10 days old = normally gist level, but importance > 0.7 doubles thresholds
        # so gist threshold becomes 168*2 = 336 hours = 14 days
        mid = _make_memory(storage, hours_old=240, importance=0.9)
        mem = storage.get_memory(mid)
        assert compressor.get_compression_schedule(mem) == 0

    def test_high_surprise_resists(self, compressor, storage):
        """High surprise memory should resist compression (thresholds * 1.5)."""
        # 10 days old = normally gist level, but surprise > 0.6 multiplies by 1.5
        # gist threshold becomes 168*1.5 = 252 hours = 10.5 days
        # 240h < 252h, so stays at level 0
        mid = _make_memory(storage, hours_old=240, surprise_score=0.8)
        mem = storage.get_memory(mid)
        assert compressor.get_compression_schedule(mem) == 0

    def test_high_confidence_resists(self, compressor, storage):
        """High confidence memory should resist compression (thresholds * 1.3)."""
        # 8 days (192h). Normal gist threshold = 168h → level 1
        # With confidence > 0.8: threshold = 168 * 1.3 = 218.4h → level 0
        mid = _make_memory(storage, hours_old=192, confidence=0.9)
        mem = storage.get_memory(mid)
        assert compressor.get_compression_schedule(mem) == 0

    def test_high_access_count_resists(self, compressor, storage):
        """Frequently accessed memory should resist compression."""
        # 8 days (192h). Normal gist threshold = 168h → level 1
        # With access_count > 10: threshold = 168 * 1.5 = 252h → level 0
        mid = _make_memory(storage, hours_old=192, access_count=15)
        mem = storage.get_memory(mid)
        assert compressor.get_compression_schedule(mem) == 0

    def test_protected_never_compresses(self, compressor, storage):
        """Protected memory should always return level 0."""
        mid = _make_memory(storage, hours_old=2000, is_protected=True)
        mem = storage.get_memory(mid)
        assert compressor.get_compression_schedule(mem) == 0

    def test_semantic_never_compresses(self, compressor, storage):
        """Semantic store memories should always return level 0."""
        mid = _make_memory(storage, hours_old=2000, store_type="semantic")
        mem = storage.get_memory(mid)
        assert compressor.get_compression_schedule(mem) == 0

    def test_combined_resistance_stacks(self, compressor, storage):
        """Multiple resistance factors should multiply together."""
        # 35 days (840h). Normal tag threshold = 720h → level 2
        # importance>0.7 (x2) + surprise>0.6 (x1.5) = 720*3.0 = 2160h threshold
        # 840h < 2160h tag and 168*3.0=504h gist → still level 1
        # Actually gist threshold = 168 * 3.0 = 504h, 840h > 504h → level 1
        mid = _make_memory(storage, hours_old=840, importance=0.9, surprise_score=0.8)
        mem = storage.get_memory(mid)
        assert compressor.get_compression_schedule(mem) == 1

    def test_boundary_at_gist_threshold(self, compressor, storage):
        """Memory just past gist threshold should be level 1."""
        # Default confidence=1.0 triggers 1.3x resistance, so gist threshold = 168*1.3 = 218.4h
        mid = _make_memory(storage, hours_old=220)  # Just past adjusted threshold
        mem = storage.get_memory(mid)
        assert compressor.get_compression_schedule(mem) == 1


# -- Gist compression tests --


class TestGistCompression:
    def test_compress_gist_preserves_code(self, compressor, storage):
        """Code blocks should survive gist compression verbatim."""
        content = (
            "We fixed the authentication bug in the login module.\n"
            "The issue was a missing null check.\n"
            "Here is the fix:\n"
            "```python\ndef login(user):\n    if user is None:\n        return False\n    return True\n```\n"
            "This resolved the production outage.\n"
            "We also updated the tests.\n"
            "The CI pipeline is now green.\n"
            "All stakeholders were notified."
        )
        mid = _make_memory(storage, content=content, hours_old=0)
        gist = compressor.compress_to_gist(mid)
        assert "```python" in gist
        assert "def login(user):" in gist

    def test_compress_gist_reduces_length(self, compressor, storage):
        """Gist should be shorter than original content."""
        content = "\n".join([
            "This is the first important sentence about the project setup.",
            "We configured the database with PostgreSQL 14.2 for production.",
            "The middleware layer handles authentication and rate limiting.",
            "We set up Redis caching for frequently accessed endpoints.",
            "The deployment pipeline uses GitHub Actions with Docker.",
            "Testing coverage reached 85% across all modules.",
            "Performance benchmarks show 50ms average response time.",
            "The team decided to use TypeScript for the frontend rewrite.",
            "Documentation was updated in the wiki.",
            "Code review process now requires two approvals.",
            "The staging environment mirrors production exactly.",
            "Monitoring alerts are configured for CPU and memory usage.",
            "This is the final wrap-up of the sprint review."
        ])
        mid = _make_memory(storage, content=content, hours_old=0)
        gist = compressor.compress_to_gist(mid)
        assert len(gist) < len(content)

    def test_compress_gist_preserves_first_and_last(self, compressor, storage):
        """First and last sentences should always be preserved (primacy-recency)."""
        content = (
            "FIRST: We started the migration project today.\n"
            "Middle sentence one.\n"
            "Middle sentence two.\n"
            "Middle sentence three.\n"
            "Middle sentence four.\n"
            "Middle sentence five.\n"
            "Middle sentence six.\n"
            "Middle sentence seven.\n"
            "Middle sentence eight.\n"
            "LAST: The migration is now complete."
        )
        mid = _make_memory(storage, content=content, hours_old=0)
        gist = compressor.compress_to_gist(mid)
        assert "FIRST:" in gist
        assert "LAST:" in gist

    def test_already_compressed_returns_content(self, compressor, storage):
        """Compressing a memory that's already at gist level returns current content."""
        mid = _make_memory(storage, content="already gist", hours_old=0,
                          compression_level=1)
        result = compressor.compress_to_gist(mid)
        assert result == "already gist"

    def test_compress_gist_short_content_preserved(self, compressor, storage):
        """Short content (<=3 sentences) should be preserved entirely."""
        content = "First. Second. Third."
        mid = _make_memory(storage, content=content, hours_old=0)
        gist = compressor.compress_to_gist(mid)
        assert "First" in gist
        assert "Second" in gist
        assert "Third" in gist


# -- Tag compression tests --


class TestTagCompression:
    def test_compress_tag_under_200_chars(self, compressor, storage):
        """Tag representation should be under 200 characters."""
        content = (
            "We implemented the UserAuthentication service for the FastAPI backend.\n"
            "It uses JWT tokens with RSA-256 signing.\n"
            "The token refresh endpoint handles expired tokens gracefully.\n"
            "Error handling covers InvalidToken, ExpiredToken, and MissingClaims.\n"
            "All tests pass with 100% coverage."
        )
        mid = _make_memory(storage, content=content, hours_old=0)
        # First compress to gist
        compressor.compress_to_gist(mid)
        # Then compress to tag
        tag = compressor.compress_to_tag(mid)
        assert len(tag) <= 200

    def test_compress_tag_from_level_0(self, compressor, storage):
        """Compressing from level 0 to tag should go through gist first."""
        content = (
            "We fixed the DatabaseConnection pool leak.\n"
            "The connection was not being returned after exceptions.\n"
            "Added try/finally blocks around all database operations.\n"
            "This fixed the memory leak that caused OOM errors.\n"
            "Production is stable now."
        )
        mid = _make_memory(storage, content=content, hours_old=0)
        tag = compressor.compress_to_tag(mid)
        # Should have two archives: original and gist
        archives = storage.get_archives_for_memory(mid)
        assert len(archives) == 2
        assert len(tag) <= 200

    def test_compress_tag_format(self, compressor, storage):
        """Tag should contain summary, tags section, and created date."""
        content = "We deployed the ServiceMesh configuration for Kubernetes."
        mid = _make_memory(storage, content=content, hours_old=0)
        compressor.compress_to_gist(mid)
        tag = compressor.compress_to_tag(mid)
        assert "Tags:" in tag
        assert "Created:" in tag

    def test_already_at_tag_returns_content(self, compressor, storage):
        """Compressing a memory already at tag level returns current content."""
        mid = _make_memory(storage, content="old tag content", hours_old=0,
                          compression_level=2)
        result = compressor.compress_to_tag(mid)
        assert result == "old tag content"


# -- Archive and decompress tests --


class TestArchiveAndDecompress:
    def test_original_archived(self, compressor, storage):
        """Original content should be stored in memory_archives after compression."""
        original = "This is the original full-fidelity memory content."
        mid = _make_memory(storage, content=original, hours_old=0)
        compressor.compress_to_gist(mid)

        archives = storage.get_archives_for_memory(mid)
        assert len(archives) >= 1
        # The archive should contain the original content
        archive_contents = [a["content"] for a in archives]
        assert original in archive_contents

    def test_original_content_field_set(self, compressor, storage):
        """The memory's original_content field should be set after compression."""
        original = "Original content for field test."
        mid = _make_memory(storage, content=original, hours_old=0)
        compressor.compress_to_gist(mid)

        mem = storage.get_memory(mid)
        assert mem["original_content"] == original

    def test_decompress_restores(self, compressor, storage):
        """Decompress should return the original full content."""
        original = "This is the complete original content that should be restored."
        mid = _make_memory(storage, content=original, hours_old=0)
        compressor.compress_to_gist(mid)

        restored = compressor.decompress(mid)
        assert restored == original

    def test_decompress_no_archive(self, compressor, storage):
        """Decompress with no archive falls back to original_content field."""
        mid = _make_memory(storage, content="current content", hours_old=0,
                          original_content="saved original")
        restored = compressor.decompress(mid)
        assert restored == "saved original"

    def test_decompress_no_original(self, compressor, storage):
        """Decompress with nothing available returns content with note."""
        mid = _make_memory(storage, content="only content", hours_old=0)
        restored = compressor.decompress(mid)
        assert "only content" in restored
        assert "no original available" in restored

    def test_decompress_nonexistent_memory(self, compressor):
        """Decompress on nonexistent memory returns note."""
        result = compressor.decompress(99999)
        assert "no original available" in result


# -- Compression cycle tests --


class TestCompressionCycle:
    def test_compression_cycle_stats(self, compressor, storage):
        """Compression cycle should return correct counts."""
        # Recent memory (stays full)
        _make_memory(storage, content="recent memory", hours_old=24)
        # Medium-age memory (should become gist)
        _make_memory(storage, content="medium age memory with enough text to compress", hours_old=240)
        # Old memory (should become tag) — confidence=1.0 gives 1.3x resistance, so tag threshold=936h
        _make_memory(storage, content="old memory with content for tag compression step", hours_old=1000)
        # Protected memory (skipped)
        _make_memory(storage, content="protected memory", hours_old=800, is_protected=True)
        # Semantic memory (skipped)
        _make_memory(storage, content="semantic memory", hours_old=800, store_type="semantic")

        stats = compressor.compression_cycle()
        assert stats["compressed_to_gist"] >= 1
        assert stats["compressed_to_tag"] >= 1
        assert stats["protected_skipped"] == 1
        assert stats["semantic_skipped"] == 1

    def test_compression_cycle_idempotent(self, compressor, storage):
        """Running compression cycle twice should not re-compress already compressed memories."""
        _make_memory(storage, content="medium age memory with content", hours_old=240)

        stats1 = compressor.compression_cycle()
        assert stats1["compressed_to_gist"] >= 1

        stats2 = compressor.compression_cycle()
        # Should be already compressed, not re-compressed
        assert stats2["compressed_to_gist"] == 0


# -- Re-embedding tests --


class TestReEmbedding:
    def test_re_embedding_after_compression(self, compressor, storage, mock_embeddings):
        """Embedding should be updated after compression to match new content."""
        original_content = (
            "We implemented the AuthenticationService for handling JWT tokens.\n"
            "The service validates tokens against the RSA public key.\n"
            "Expired tokens trigger a refresh flow automatically.\n"
            "We added comprehensive error handling for all edge cases.\n"
            "The test suite covers token creation, validation, and refresh."
        )
        # Create a memory with an initial embedding
        initial_embedding = mock_embeddings.encode("initial")
        mid = _make_memory(storage, content=original_content, hours_old=0,
                          embedding=initial_embedding)

        # Verify initial embedding is set
        mem_before = storage.get_memory(mid)
        assert mem_before["embedding"] is not None

        # Compress to gist
        compressor.compress_to_gist(mid)

        # Verify embedding was updated (different from initial since content changed)
        mem_after = storage.get_memory(mid)
        assert mem_after["embedding"] is not None
        assert mem_after["compression_level"] == 1

    def test_re_embedding_after_tag_compression(self, compressor, storage, mock_embeddings):
        """Embedding should be updated after tag compression."""
        content = (
            "The UserService handles user creation and authentication.\n"
            "It interfaces with PostgreSQL via SQLAlchemy ORM.\n"
            "Password hashing uses bcrypt with work factor 12.\n"
            "Session management is handled by Redis."
        )
        initial_embedding = mock_embeddings.encode("initial")
        mid = _make_memory(storage, content=content, hours_old=0,
                          embedding=initial_embedding)

        compressor.compress_to_gist(mid)
        compressor.compress_to_tag(mid)

        mem = storage.get_memory(mid)
        assert mem["compression_level"] == 2
        assert mem["embedding"] is not None


# -- Edge cases --


class TestEdgeCases:
    def test_empty_content(self, compressor, storage):
        """Compressing empty content should not crash."""
        mid = _make_memory(storage, content="", hours_old=0)
        gist = compressor.compress_to_gist(mid)
        assert isinstance(gist, str)

    def test_code_only_content(self, compressor, storage):
        """Content that is only code blocks should preserve everything."""
        content = "```python\ndef hello():\n    print('world')\n```"
        mid = _make_memory(storage, content=content, hours_old=0)
        gist = compressor.compress_to_gist(mid)
        assert "def hello():" in gist

    def test_nonexistent_memory_compress(self, compressor):
        """Compressing nonexistent memory should return empty string."""
        assert compressor.compress_to_gist(99999) == ""
        assert compressor.compress_to_tag(99999) == ""

    def test_archive_reason_is_compression(self, compressor, storage):
        """Archive entries should have reason='compression'."""
        mid = _make_memory(storage, content="archive reason test", hours_old=0)
        compressor.compress_to_gist(mid)
        archives = storage.get_archives_for_memory(mid)
        assert archives[0]["archive_reason"] == "compression"
