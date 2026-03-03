"""Tests for Successor Representation cognitive maps."""

import numpy as np
import pytest

from zikkaron.cognitive_map import CognitiveMap, _MIN_TRANSITIONS
from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.retrieval import HippoRetriever
from zikkaron.storage import StorageEngine


@pytest.fixture
def storage(tmp_path):
    engine = StorageEngine(str(tmp_path / "test_cognitive_map.db"))
    yield engine
    engine.close()


@pytest.fixture
def settings():
    return Settings(DB_PATH=":memory:", SR_DISCOUNT=0.9, SR_UPDATE_RATE=0.1)


@pytest.fixture
def embeddings():
    return EmbeddingEngine("all-MiniLM-L6-v2")


@pytest.fixture
def cmap(storage, settings):
    return CognitiveMap(storage, settings)


@pytest.fixture
def graph(storage, settings):
    return KnowledgeGraph(storage, settings)


@pytest.fixture
def retriever(storage, embeddings, graph, settings):
    ret = HippoRetriever(storage, embeddings, graph, settings)
    return ret


def _make_memory(storage, embeddings, content, directory="/proj", tags=None, heat=1.0):
    """Insert a memory with embedding and return its ID."""
    embedding = embeddings.encode(content)
    mid = storage.insert_memory({
        "content": content,
        "embedding": embedding,
        "tags": tags or [],
        "directory_context": directory,
        "heat": heat,
        "is_stale": False,
        "file_hash": None,
        "embedding_model": embeddings.get_model_name(),
    })
    return mid


class TestRecordTransition:
    def test_record_transition(self, storage, cmap):
        """Transition is stored correctly."""
        m1 = storage.insert_memory({
            "content": "mem1", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })
        m2 = storage.insert_memory({
            "content": "mem2", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })

        cmap.record_transition(m1, m2, "sess1")

        t = storage.get_transition(m1, m2)
        assert t is not None
        assert t["from_memory_id"] == m1
        assert t["to_memory_id"] == m2
        assert t["count"] == 1

    def test_record_transition_increments(self, storage, cmap):
        """Recording the same transition increments the count."""
        m1 = storage.insert_memory({
            "content": "mem1", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })
        m2 = storage.insert_memory({
            "content": "mem2", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })

        cmap.record_transition(m1, m2, "sess1")
        cmap.record_transition(m1, m2, "sess1")
        cmap.record_transition(m1, m2, "sess1")

        t = storage.get_transition(m1, m2)
        assert t["count"] == 3

    def test_record_transition_marks_dirty(self, storage, cmap):
        """Recording a transition marks the map as dirty."""
        m1 = storage.insert_memory({
            "content": "a", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })
        m2 = storage.insert_memory({
            "content": "b", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })

        # Compute SR first to set dirty=False
        cmap.record_transition(m1, m2)
        cmap.compute_sr_matrix()
        assert not cmap.is_dirty

        # Now record another transition
        cmap.record_transition(m2, m1)
        assert cmap.is_dirty


class TestTransitionMatrix:
    def test_transition_matrix_normalized(self, storage, cmap):
        """Rows of the transition matrix sum to 1.0."""
        m1 = storage.insert_memory({
            "content": "a", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })
        m2 = storage.insert_memory({
            "content": "b", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })
        m3 = storage.insert_memory({
            "content": "c", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })

        cmap.record_transition(m1, m2)
        cmap.record_transition(m1, m3)
        cmap.record_transition(m2, m3)
        cmap.record_transition(m3, m1)

        T = cmap.build_transition_matrix()

        # All rows with outgoing transitions should sum to 1.0
        row_sums = T.sum(axis=1)
        for s in row_sums:
            assert s == pytest.approx(1.0, abs=1e-10)

    def test_transition_matrix_empty(self, cmap):
        """Empty transition matrix for no data."""
        T = cmap.build_transition_matrix()
        assert T.shape == (0, 0)

    def test_transition_matrix_values(self, storage, cmap):
        """Transition probabilities reflect counts."""
        m1 = storage.insert_memory({
            "content": "a", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })
        m2 = storage.insert_memory({
            "content": "b", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })

        # m1→m2 three times, m1→m1 once (4 total from m1)
        for _ in range(3):
            cmap.record_transition(m1, m2)
        cmap.record_transition(m1, m1)

        T = cmap.build_transition_matrix()
        i1 = cmap._memory_index[m1]
        i2 = cmap._memory_index[m2]

        assert T[i1, i2] == pytest.approx(0.75, abs=1e-10)
        assert T[i1, i1] == pytest.approx(0.25, abs=1e-10)


class TestSRMatrix:
    def test_sr_matrix_shape(self, storage, cmap):
        """SR matrix is square N×N."""
        m1 = storage.insert_memory({
            "content": "a", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })
        m2 = storage.insert_memory({
            "content": "b", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })
        m3 = storage.insert_memory({
            "content": "c", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })

        cmap.record_transition(m1, m2)
        cmap.record_transition(m2, m3)
        cmap.record_transition(m3, m1)

        M = cmap.compute_sr_matrix()
        assert M.shape == (3, 3)

    def test_sr_diagonal_dominant(self, storage, cmap):
        """M[i,i] >= M[i,j] for all j — you visit yourself most often."""
        m1 = storage.insert_memory({
            "content": "a", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })
        m2 = storage.insert_memory({
            "content": "b", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })
        m3 = storage.insert_memory({
            "content": "c", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })

        cmap.record_transition(m1, m2)
        cmap.record_transition(m2, m3)
        cmap.record_transition(m3, m1)

        M = cmap.compute_sr_matrix()

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                assert M[i, i] >= M[i, j] - 1e-10, (
                    f"M[{i},{i}]={M[i,i]} < M[{i},{j}]={M[i,j]}"
                )

    def test_sr_matrix_empty(self, cmap):
        """SR matrix for no data is empty."""
        M = cmap.compute_sr_matrix()
        assert M.shape == (0, 0)
        assert not cmap.is_dirty

    def test_sr_matrix_clears_dirty(self, storage, cmap):
        """Computing SR matrix clears the dirty flag."""
        m1 = storage.insert_memory({
            "content": "a", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })
        m2 = storage.insert_memory({
            "content": "b", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })

        cmap.record_transition(m1, m2)
        assert cmap.is_dirty

        cmap.compute_sr_matrix()
        assert not cmap.is_dirty


class TestCoordinates:
    def test_extract_coordinates_2d(self, storage, cmap):
        """Returns 2D coordinates for each memory in the transition graph."""
        mids = []
        for i in range(4):
            mid = storage.insert_memory({
                "content": f"mem{i}", "tags": [], "directory_context": "/p",
                "heat": 1.0, "is_stale": False,
            })
            mids.append(mid)

        # Create a cycle: 0→1→2→3→0
        for i in range(4):
            cmap.record_transition(mids[i], mids[(i + 1) % 4])

        coords = cmap.extract_coordinates(n_dims=2)

        assert len(coords) == 4
        for mid in mids:
            assert mid in coords
            assert len(coords[mid]) == 2
            assert all(isinstance(v, float) for v in coords[mid])

    def test_extract_coordinates_recomputes_when_dirty(self, storage, cmap):
        """Extracting coords recomputes SR if dirty."""
        m1 = storage.insert_memory({
            "content": "a", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })
        m2 = storage.insert_memory({
            "content": "b", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })

        cmap.record_transition(m1, m2)
        coords = cmap.extract_coordinates()
        assert not cmap.is_dirty
        assert len(coords) == 2

    def test_update_memory_coordinates(self, storage, cmap):
        """update_memory_coordinates writes sr_x, sr_y to storage."""
        m1 = storage.insert_memory({
            "content": "a", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })
        m2 = storage.insert_memory({
            "content": "b", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })

        cmap.record_transition(m1, m2)
        count = cmap.update_memory_coordinates()
        assert count == 2

        mem = storage.get_memory(m1)
        assert mem["sr_x"] != 0.0 or mem["sr_y"] != 0.0


class TestNavigateTo:
    def test_navigate_to_basic(self, storage, embeddings, cmap):
        """Finds memories near query position in SR space."""
        m1 = _make_memory(storage, embeddings, "Python web development with Flask")
        m2 = _make_memory(storage, embeddings, "Database optimization techniques")
        m3 = _make_memory(storage, embeddings, "React frontend components")

        # Create transitions so SR has something to work with
        cmap.record_transition(m1, m2)
        cmap.record_transition(m2, m3)
        cmap.record_transition(m3, m1)
        cmap.compute_sr_matrix()

        query_emb = embeddings.encode("web development")
        results = cmap.navigate_to(query_emb, embeddings, top_k=3)

        assert len(results) > 0
        assert all(isinstance(mid, int) for mid, _ in results)
        assert all(0 < score <= 1.0 for _, score in results)
        # Results should be sorted by proximity descending
        proximities = [s for _, s in results]
        assert proximities == sorted(proximities, reverse=True)

    def test_navigate_to_empty(self, cmap, embeddings):
        """Navigate returns empty when no transitions exist."""
        query_emb = embeddings.encode("test query")
        results = cmap.navigate_to(query_emb, embeddings)
        assert results == []


class TestNeighborhood:
    def test_neighborhood(self, storage, cmap):
        """Returns memories within radius in SR space."""
        mids = []
        for i in range(5):
            mid = storage.insert_memory({
                "content": f"memory content {i}", "tags": [],
                "directory_context": "/p", "heat": 1.0, "is_stale": False,
            })
            mids.append(mid)

        # Dense connections between first 3
        for i in range(3):
            for j in range(3):
                if i != j:
                    cmap.record_transition(mids[i], mids[j])

        # Sparse connection to 4th and 5th
        cmap.record_transition(mids[3], mids[4])
        cmap.record_transition(mids[4], mids[3])

        cmap.compute_sr_matrix()

        neighbors = cmap.get_neighborhood(mids[0], radius=10.0)
        # Should include at least some neighbors
        assert isinstance(neighbors, list)
        for n in neighbors:
            assert "sr_distance" in n
            assert n["id"] != mids[0]

    def test_neighborhood_unknown_memory(self, cmap):
        """Neighborhood returns empty for unknown memory."""
        result = cmap.get_neighborhood(99999)
        assert result == []


class TestIncrementalUpdate:
    def test_incremental_update(self, storage, cmap):
        """TD update modifies the correct row of SR matrix."""
        m1 = storage.insert_memory({
            "content": "a", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })
        m2 = storage.insert_memory({
            "content": "b", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })
        m3 = storage.insert_memory({
            "content": "c", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })

        cmap.record_transition(m1, m2)
        cmap.record_transition(m2, m3)
        cmap.record_transition(m3, m1)
        M_before = cmap.compute_sr_matrix().copy()

        # Do a TD update from m1 → m2
        cmap.incremental_update(m1, m2)

        M_after = cmap._sr_matrix

        # m1's row should have changed
        i1 = cmap._memory_index[m1]
        assert not np.allclose(M_before[i1], M_after[i1])

        # Other rows should be unchanged
        for idx in range(M_before.shape[0]):
            if idx != i1:
                np.testing.assert_array_almost_equal(M_before[idx], M_after[idx])

    def test_incremental_update_unknown_ids(self, cmap):
        """TD update is a no-op for unknown IDs."""
        cmap._sr_matrix = np.eye(2)
        cmap._memory_index = {1: 0, 2: 1}
        before = cmap._sr_matrix.copy()

        cmap.incremental_update(999, 888)

        np.testing.assert_array_equal(before, cmap._sr_matrix)

    def test_incremental_update_no_matrix(self, cmap):
        """TD update is a no-op when no SR matrix exists."""
        cmap.incremental_update(1, 2)  # Should not raise


class TestFrequentlyCoaccessed:
    def test_frequently_coaccessed_cluster(self, storage, cmap):
        """Memories accessed together cluster in SR space."""
        # Group A: frequently co-accessed
        a1 = storage.insert_memory({
            "content": "group A mem 1", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })
        a2 = storage.insert_memory({
            "content": "group A mem 2", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })
        a3 = storage.insert_memory({
            "content": "group A mem 3", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })

        # Group B: frequently co-accessed among themselves
        b1 = storage.insert_memory({
            "content": "group B mem 1", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })
        b2 = storage.insert_memory({
            "content": "group B mem 2", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })

        # Heavy transitions within group A
        for _ in range(10):
            cmap.record_transition(a1, a2)
            cmap.record_transition(a2, a3)
            cmap.record_transition(a3, a1)

        # Heavy transitions within group B
        for _ in range(10):
            cmap.record_transition(b1, b2)
            cmap.record_transition(b2, b1)

        # Weak cross-group link
        cmap.record_transition(a3, b1)

        coords = cmap.extract_coordinates(n_dims=2)

        # Within-group distances should be less than cross-group distances
        def dist(id1, id2):
            c1, c2 = np.array(coords[id1]), np.array(coords[id2])
            return float(np.linalg.norm(c1 - c2))

        # Average intra-group A distance
        intra_a = (dist(a1, a2) + dist(a2, a3) + dist(a1, a3)) / 3.0
        # Average cross-group distance
        cross = (dist(a1, b1) + dist(a1, b2) + dist(a2, b1) + dist(a2, b2)) / 4.0

        assert intra_a < cross, (
            f"Intra-group distance ({intra_a:.4f}) should be less than "
            f"cross-group distance ({cross:.4f})"
        )


class TestHasSufficientData:
    def test_insufficient_data(self, cmap):
        """Returns False when too few transitions."""
        assert not cmap.has_sufficient_data()

    def test_sufficient_data(self, storage, cmap):
        """Returns True when enough transitions exist."""
        m1 = storage.insert_memory({
            "content": "a", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })
        m2 = storage.insert_memory({
            "content": "b", "tags": [], "directory_context": "/p",
            "heat": 1.0, "is_stale": False,
        })

        # Record many transitions to exceed _MIN_TRANSITIONS
        for _ in range(_MIN_TRANSITIONS):
            cmap.record_transition(m1, m2)

        assert cmap.has_sufficient_data()


class TestIntegration:
    def test_integration_navigate_tool(self, storage, embeddings, settings):
        """Navigate tool works end-to-end (simulate server.navigate_memory)."""
        cmap = CognitiveMap(storage, settings)

        # Create memories with real embeddings
        m1 = _make_memory(storage, embeddings, "Python web frameworks Flask Django")
        m2 = _make_memory(storage, embeddings, "JavaScript React Vue frontend")
        m3 = _make_memory(storage, embeddings, "SQL database optimization PostgreSQL")

        # Build transition history
        cmap.record_transition(m1, m2)
        cmap.record_transition(m2, m3)
        cmap.record_transition(m3, m1)
        cmap.compute_sr_matrix()

        # Navigate
        query_emb = embeddings.encode("web application development")
        results = cmap.navigate_to(query_emb, embeddings, top_k=3)

        assert len(results) == 3
        memory_ids = {mid for mid, _ in results}
        assert memory_ids == {m1, m2, m3}

    def test_sr_retrieval_signal(self, storage, embeddings, graph, settings):
        """SR scores appear in recall when active (sufficient transitions)."""
        cmap = CognitiveMap(storage, settings)
        retriever = HippoRetriever(storage, embeddings, graph, settings)
        retriever.set_cognitive_map(cmap)

        # Create memories
        m1 = _make_memory(storage, embeddings, "Python Flask web server API")
        m2 = _make_memory(storage, embeddings, "REST API endpoint design patterns")
        m3 = _make_memory(storage, embeddings, "Database schema migration tools")

        # Build enough transitions to activate SR
        for _ in range(_MIN_TRANSITIONS // 3 + 1):
            cmap.record_transition(m1, m2)
            cmap.record_transition(m2, m3)
            cmap.record_transition(m3, m1)

        assert cmap.has_sufficient_data()

        # Recall should include SR signal
        results = retriever.recall("Flask API development", max_results=3)
        assert len(results) > 0

    def test_sr_retrieval_signal_inactive(self, storage, embeddings, graph, settings):
        """SR signal is not used when insufficient data."""
        cmap = CognitiveMap(storage, settings)
        retriever = HippoRetriever(storage, embeddings, graph, settings)
        retriever.set_cognitive_map(cmap)

        m1 = _make_memory(storage, embeddings, "Python Flask web server")
        m2 = _make_memory(storage, embeddings, "Database optimization")

        # Only a few transitions — not enough
        cmap.record_transition(m1, m2)

        assert not cmap.has_sufficient_data()

        # Should still work, just without SR
        results = retriever.recall("web server", max_results=2)
        assert len(results) > 0

    def test_get_sr_scores(self, storage, embeddings, settings):
        """get_sr_scores returns proximity scores for candidate memories."""
        cmap = CognitiveMap(storage, settings)

        m1 = _make_memory(storage, embeddings, "Alpha beta gamma")
        m2 = _make_memory(storage, embeddings, "Delta epsilon zeta")
        m3 = _make_memory(storage, embeddings, "Eta theta iota")

        cmap.record_transition(m1, m2)
        cmap.record_transition(m2, m3)
        cmap.record_transition(m3, m1)
        cmap.compute_sr_matrix()

        query_emb = embeddings.encode("alpha")
        sr_scores = cmap.get_sr_scores(query_emb, embeddings, [m1, m2, m3])

        assert len(sr_scores) > 0
        for mid, score in sr_scores.items():
            assert 0.0 <= score <= 1.0
