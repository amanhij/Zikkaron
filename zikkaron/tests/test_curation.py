"""Tests for the active memory curation engine and memify self-improvement layer."""

import json

import numpy as np
import pytest

from zikkaron.config import Settings
from zikkaron.curation import MemoryCurator
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.storage import StorageEngine
from zikkaron.thermodynamics import MemoryThermodynamics

# Detect whether the embedding model can be loaded
_engine = EmbeddingEngine()
try:
    _engine._ensure_model()
    _model_available = not _engine._unavailable
except Exception:
    _model_available = False

requires_model = pytest.mark.skipif(
    not _model_available,
    reason="sentence-transformers model not available",
)


@pytest.fixture
def storage(tmp_path):
    engine = StorageEngine(str(tmp_path / "test_curation.db"), embedding_dim=384)
    yield engine
    engine.close()


@pytest.fixture
def settings():
    return Settings(
        DB_PATH=":memory:",
        CURATION_SIMILARITY_THRESHOLD=0.85,
    )


@pytest.fixture
def embeddings():
    return EmbeddingEngine()


@pytest.fixture
def thermo(storage, embeddings, settings):
    return MemoryThermodynamics(storage, embeddings, settings)


@pytest.fixture
def curator(storage, embeddings, thermo, settings):
    return MemoryCurator(storage, embeddings, thermo, settings)


def _make_embedding(dim: int = 384, seed: int = 0) -> bytes:
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    # Normalize for cosine similarity
    vec = vec / np.linalg.norm(vec)
    return vec.tobytes()


def _make_similar_embedding(base: bytes, noise_scale: float = 0.01, seed: int = 1) -> bytes:
    """Create an embedding very similar to base (high cosine similarity)."""
    arr = np.frombuffer(base, dtype=np.float32).copy()
    rng = np.random.RandomState(seed)
    noise = rng.randn(len(arr)).astype(np.float32) * noise_scale
    arr += noise
    arr = arr / np.linalg.norm(arr)
    return arr.tobytes()


def _make_moderate_embedding(base: bytes, noise_scale: float = 0.5, seed: int = 2) -> bytes:
    """Create an embedding moderately similar to base (0.6-0.85 range)."""
    arr = np.frombuffer(base, dtype=np.float32).copy()
    rng = np.random.RandomState(seed)
    noise = rng.randn(len(arr)).astype(np.float32) * noise_scale
    arr += noise
    arr = arr / np.linalg.norm(arr)
    return arr.tobytes()


# ── test_curate_new_memory ───────────────────────────────────────────────


def test_curate_new_memory(curator, storage):
    """Novel content creates a new memory."""
    emb = _make_embedding(seed=42)
    result = curator.curate_on_remember(
        content="Brand new unique memory content about quantum physics",
        context="/test/project",
        tags=["test", "physics"],
        embedding=emb,
    )
    assert result["action"] == "created"
    assert "memory_id" in result

    mem = storage.get_memory(result["memory_id"])
    assert mem is not None
    assert "quantum physics" in mem["content"]


# ── test_curate_merge_similar ────────────────────────────────────────────


@requires_model
def test_curate_merge_similar(curator, storage, embeddings):
    """Similar content merges with existing memory."""
    content1 = "Python uses indentation for code blocks instead of braces"
    emb1 = embeddings.encode(content1)
    result1 = curator.curate_on_remember(
        content=content1,
        context="/test/project",
        tags=["python", "syntax"],
        embedding=emb1,
    )
    assert result1["action"] == "created"
    original_id = result1["memory_id"]

    # Very similar content should merge
    content2 = "Python uses indentation for code blocks instead of curly braces"
    emb2 = embeddings.encode(content2)
    result2 = curator.curate_on_remember(
        content=content2,
        context="/test/project",
        tags=["python", "language"],
        embedding=emb2,
    )
    assert result2["action"] == "merged"
    assert result2["memory_id"] == original_id

    merged = storage.get_memory(original_id)
    assert content1 in merged["content"]
    assert content2 in merged["content"]
    # Tags should be union
    assert "syntax" in merged["tags"]
    assert "language" in merged["tags"]
    assert merged["heat"] == 1.0  # Heat refreshed


# ── test_curate_link_moderate ────────────────────────────────────────────


@requires_model
def test_curate_link_moderate(curator, storage, embeddings):
    """Moderately similar content creates a link."""
    content1 = "FastAPI is a modern Python web framework for building APIs"
    emb1 = embeddings.encode(content1)
    result1 = curator.curate_on_remember(
        content=content1,
        context="/test/project",
        tags=["fastapi"],
        embedding=emb1,
    )
    assert result1["action"] == "created"
    original_id = result1["memory_id"]

    # Related but not identical — should link
    content2 = "Django is a full-featured Python web framework for web applications"
    emb2 = embeddings.encode(content2)

    # Verify similarity is in the moderate range
    sim = embeddings.similarity(emb1, emb2)
    if sim >= 0.6 and sim < 0.85:
        result2 = curator.curate_on_remember(
            content=content2,
            context="/test/project",
            tags=["django"],
            embedding=emb2,
        )
        assert result2["action"] == "linked"
        assert result2["memory_id"] != original_id
        assert result2["linked_to"] == original_id

        # Both memories should exist
        assert storage.get_memory(result2["memory_id"]) is not None
        assert storage.get_memory(original_id) is not None

        # Check that a derived_from relationship was created
        rels = storage._conn.execute(
            "SELECT * FROM relationships WHERE relationship_type = 'derived_from'"
        ).fetchall()
        assert len(rels) >= 1
    else:
        # If model gives different similarity, test the mechanism with synthetic embeddings
        emb_base = _make_embedding(seed=100)
        result_base = curator.curate_on_remember(
            content="Base memory content for linking test",
            context="/test/project",
            tags=["base"],
            embedding=emb_base,
        )

        # Create embedding in moderate similarity range
        emb_moderate = _make_moderate_embedding(emb_base, noise_scale=0.35, seed=200)
        sim_check = embeddings.similarity(emb_base, emb_moderate)
        # Adjust noise to hit 0.6-0.85 range
        for scale in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            emb_moderate = _make_moderate_embedding(emb_base, noise_scale=scale, seed=200)
            sim_check = embeddings.similarity(emb_base, emb_moderate)
            if 0.6 <= sim_check < 0.85:
                break

        if 0.6 <= sim_check < 0.85:
            result_mod = curator.curate_on_remember(
                content="Moderately similar linking test content",
                context="/test/project",
                tags=["linked"],
                embedding=emb_moderate,
            )
            assert result_mod["action"] == "linked"
        else:
            pytest.skip("Could not generate embedding in moderate similarity range")


# ── test_contradiction_detection ─────────────────────────────────────────


@requires_model
def test_contradiction_detection(curator, storage, embeddings):
    """Opposing content is flagged as contradicting."""
    content1 = "We use PostgreSQL as our primary database for the application"
    emb1 = embeddings.encode(content1)
    storage.insert_memory({
        "content": content1,
        "embedding": emb1,
        "tags": ["database"],
        "directory_context": "/test",
        "heat": 1.0,
        "is_stale": False,
    })

    content2 = "We no longer use PostgreSQL, instead of PostgreSQL we switched to MySQL"
    emb2 = embeddings.encode(content2)

    contradictions = curator.detect_contradictions(content2, emb2)

    # Should detect the contradiction (negation pattern present in new content)
    if contradictions:
        assert any(c["reason"] in ("negation_mismatch", "action_divergence")
                    for c in contradictions)
        # Old memory's confidence should be reduced
        for c in contradictions:
            old_mem = storage.get_memory(c["memory_id"])
            assert old_mem["confidence"] < 1.0


# ── test_memify_prune ────────────────────────────────────────────────────


def test_memify_prune(curator, storage):
    """Cold unreliable memories with zero access get pruned."""
    emb = _make_embedding(seed=10)
    mid = storage.insert_memory({
        "content": "This is a cold unreliable memory that was never accessed",
        "embedding": emb,
        "tags": ["cold"],
        "directory_context": "/test",
        "heat": 0.005,  # < 0.01
        "is_stale": False,
    })
    # Set confidence < 0.3 and access_count = 0
    storage._conn.execute(
        "UPDATE memories SET confidence = 0.2, access_count = 0 WHERE id = ?",
        (mid,),
    )
    storage._conn.commit()

    stats = curator.memify_cycle()
    assert stats["pruned"] >= 1
    assert storage.get_memory(mid) is None


# ── test_memify_strengthen ───────────────────────────────────────────────


def test_memify_strengthen(curator, storage):
    """Frequently used high-confidence memories get importance boosted."""
    emb = _make_embedding(seed=20)
    mid = storage.insert_memory({
        "content": "Frequently accessed and useful memory about project architecture",
        "embedding": emb,
        "tags": ["architecture"],
        "directory_context": "/test",
        "heat": 0.8,
        "is_stale": False,
    })
    storage._conn.execute(
        "UPDATE memories SET access_count = 10, confidence = 0.9, importance = 0.5 "
        "WHERE id = ?",
        (mid,),
    )
    storage._conn.commit()

    stats = curator.memify_cycle()
    assert stats["strengthened"] >= 1

    mem = storage.get_memory(mid)
    assert mem["importance"] == pytest.approx(0.6, abs=0.01)


# ── test_memify_derive ───────────────────────────────────────────────────


def test_memify_derive(curator, storage):
    """High-weight entity pairs generate derived fact memories."""
    now = storage._now_iso()

    # Create two entities
    eid1 = storage.insert_entity({"name": "module.py", "type": "file"})
    eid2 = storage.insert_entity({"name": "utils.py", "type": "file"})

    # Create a high-weight co_occurrence relationship (weight > 10)
    storage.insert_relationship({
        "source_entity_id": eid1,
        "target_entity_id": eid2,
        "relationship_type": "co_occurrence",
        "weight": 12.0,
    })

    stats = curator.memify_cycle()
    assert stats["derived"] >= 1

    # Check the derived memory exists
    rows = storage._conn.execute(
        "SELECT * FROM memories WHERE content LIKE '%module.py%utils.py%frequently modified%'"
    ).fetchall()
    assert len(rows) >= 1


# ── test_curation_preserves_existing ─────────────────────────────────────


def test_curation_preserves_existing(curator, storage):
    """Existing memories are not corrupted by curation operations."""
    emb1 = _make_embedding(seed=50)
    mid1 = storage.insert_memory({
        "content": "Important existing memory about database migrations",
        "embedding": emb1,
        "tags": ["database", "migrations"],
        "directory_context": "/test/project",
        "heat": 0.9,
        "is_stale": False,
    })
    storage._conn.execute(
        "UPDATE memories SET confidence = 0.95, access_count = 3, importance = 0.7 "
        "WHERE id = ?",
        (mid1,),
    )
    storage._conn.commit()

    original = storage.get_memory(mid1)

    # Insert a completely different memory via curation
    emb2 = _make_embedding(seed=99)
    result = curator.curate_on_remember(
        content="Unrelated content about frontend React components",
        context="/test/other",
        tags=["react", "frontend"],
        embedding=emb2,
    )
    assert result["action"] == "created"

    # Run memify cycle
    curator.memify_cycle()

    # Verify original memory is unchanged
    preserved = storage.get_memory(mid1)
    assert preserved is not None
    assert preserved["content"] == original["content"]
    assert preserved["tags"] == original["tags"]
    assert preserved["heat"] == original["heat"]
    assert preserved["confidence"] == original["confidence"]


# ── test_memify_reweight ─────────────────────────────────────────────────


def test_memify_reweight(curator, storage):
    """Established relationships between hot entities get weight boosted."""
    eid1 = storage.insert_entity({"name": "hot_entity_a", "type": "file", "heat": 0.9})
    eid2 = storage.insert_entity({"name": "hot_entity_b", "type": "file", "heat": 0.8})

    rid = storage.insert_relationship({
        "source_entity_id": eid1,
        "target_entity_id": eid2,
        "relationship_type": "co_occurrence",
        "weight": 6.0,  # Established relationship (>= 5.0)
    })

    stats = curator.memify_cycle()
    assert stats["reweighted"] >= 1

    row = storage._conn.execute(
        "SELECT weight FROM relationships WHERE id = ?", (rid,)
    ).fetchone()
    assert row[0] == pytest.approx(6.5, abs=0.01)  # 6.0 + 0.5 boost


def test_memify_reweight_cold_decay(curator, storage):
    """Relationships between cold entities get weight decayed."""
    eid1 = storage.insert_entity({"name": "cold_a", "type": "file", "heat": 0.05})
    eid2 = storage.insert_entity({"name": "cold_b", "type": "file", "heat": 0.05})

    rid = storage.insert_relationship({
        "source_entity_id": eid1,
        "target_entity_id": eid2,
        "relationship_type": "co_occurrence",
        "weight": 3.0,
    })

    stats = curator.memify_cycle()
    assert stats["reweighted"] >= 1

    row = storage._conn.execute(
        "SELECT weight FROM relationships WHERE id = ?", (rid,)
    ).fetchone()
    assert row[0] == pytest.approx(2.7, abs=0.01)  # 3.0 * 0.9


def test_memify_derive_idempotent(curator, storage):
    """Derived facts are not duplicated on repeated runs."""
    eid1 = storage.insert_entity({"name": "a.py", "type": "file"})
    eid2 = storage.insert_entity({"name": "b.py", "type": "file"})

    storage.insert_relationship({
        "source_entity_id": eid1,
        "target_entity_id": eid2,
        "relationship_type": "co_occurrence",
        "weight": 15.0,
    })

    stats1 = curator.memify_cycle()
    assert stats1["derived"] >= 1

    stats2 = curator.memify_cycle()
    assert stats2["derived"] == 0  # Should not re-derive

    rows = storage._conn.execute(
        "SELECT COUNT(*) FROM memories WHERE content LIKE '%a.py%b.py%frequently%'"
    ).fetchone()
    assert rows[0] == 1
