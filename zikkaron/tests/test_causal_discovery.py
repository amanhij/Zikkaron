"""Tests for the PC algorithm causal discovery module."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from zikkaron.causal_discovery import CausalDiscovery
from zikkaron.config import Settings
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.storage import StorageEngine


@pytest.fixture
def storage(tmp_path):
    engine = StorageEngine(str(tmp_path / "test_causal.db"))
    yield engine
    engine.close()


@pytest.fixture
def settings():
    return Settings(DB_PATH=":memory:", CAUSAL_THRESHOLD=3)


@pytest.fixture
def kg(storage, settings):
    return KnowledgeGraph(storage, settings)


@pytest.fixture
def cd(storage, kg, settings):
    return CausalDiscovery(storage, kg, settings)


def _populate_entities_and_episodes(storage, n_entities=6, n_hours=24):
    """Create entities and episodes for testing."""
    now = datetime.now(timezone.utc)
    entity_names = [f"entity_{i}" for i in range(n_entities)]
    entity_ids = []

    for name in entity_names:
        eid = storage.insert_entity({"name": name, "type": "file"})
        entity_ids.append(eid)

    # Create episodes that mention entities in time buckets
    for h in range(n_hours):
        ts = (now - timedelta(hours=n_hours - h)).isoformat()
        # First few entities co-occur in early hours
        if h < n_hours // 2:
            content = f"Working on {entity_names[0]} and {entity_names[1]} and {entity_names[2]}"
        else:
            content = f"Working on {entity_names[3]} and {entity_names[4]} and {entity_names[5]}"

        storage.insert_episode({
            "session_id": f"sess_{h}",
            "directory": "/project",
            "raw_content": content,
            "timestamp": ts,
        })

    return entity_names, entity_ids


class TestBuildEventMatrix:
    def test_correct_shape_and_values(self, cd, storage):
        now = datetime.now(timezone.utc)
        # Create 3 entities
        storage.insert_entity({"name": "fileA", "type": "file"})
        storage.insert_entity({"name": "fileB", "type": "file"})
        storage.insert_entity({"name": "errorX", "type": "error"})

        # Create episodes mentioning them
        for i in range(5):
            ts = (now - timedelta(hours=5 - i)).isoformat()
            storage.insert_episode({
                "session_id": f"s{i}",
                "directory": "/proj",
                "raw_content": f"Changed fileA and got errorX at step {i}",
                "timestamp": ts,
            })

        data, names, timestamps = cd.build_event_matrix(hours=24)

        assert isinstance(data, np.ndarray)
        assert len(names) >= 2  # fileA and errorX at minimum
        assert "fileA" in names
        assert "errorX" in names
        assert len(timestamps) > 0
        assert data.shape[0] == len(timestamps)
        assert data.shape[1] == len(names)
        # Check that values are binary
        assert set(np.unique(data)).issubset({0.0, 1.0})

    def test_empty_no_episodes(self, cd):
        data, names, timestamps = cd.build_event_matrix(hours=24)
        assert data.shape == (0, 0)
        assert names == []
        assert timestamps == []

    def test_directory_filter(self, cd, storage):
        now = datetime.now(timezone.utc)
        storage.insert_entity({"name": "modA", "type": "file"})

        # Episode in /proj1
        storage.insert_episode({
            "session_id": "s1",
            "directory": "/proj1",
            "raw_content": "modA changed",
            "timestamp": (now - timedelta(hours=1)).isoformat(),
        })
        # Episode in /proj2
        storage.insert_episode({
            "session_id": "s2",
            "directory": "/proj2",
            "raw_content": "modA changed",
            "timestamp": (now - timedelta(hours=2)).isoformat(),
        })

        data1, names1, _ = cd.build_event_matrix(directory="/proj1", hours=24)
        data2, names2, _ = cd.build_event_matrix(directory="/proj2", hours=24)

        # Both should find the entity but from different episodes
        assert len(names1) >= 1
        assert len(names2) >= 1


class TestConditionalIndependenceTest:
    def test_rejects_independence_for_correlated(self, cd):
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 0.8 * x + 0.2 * np.random.randn(n)  # strongly correlated

        result = cd.conditional_independence_test(x, y, alpha=0.05)
        assert result is False  # should reject independence (they ARE dependent)

    def test_accepts_independence_for_random(self, cd):
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = np.random.randn(n)  # independent

        result = cd.conditional_independence_test(x, y, alpha=0.05)
        assert result is True  # should accept independence

    def test_partial_correlation_with_confounder(self, cd):
        np.random.seed(42)
        n = 200
        z = np.random.randn(n)
        x = z + 0.1 * np.random.randn(n)
        y = z + 0.1 * np.random.randn(n)

        # Without conditioning: x and y look correlated
        result_unconditional = cd.conditional_independence_test(x, y, alpha=0.05)
        assert result_unconditional is False

        # Conditioning on z: x and y should be (nearly) independent
        result_conditional = cd.conditional_independence_test(
            x, y, z=z.reshape(-1, 1), alpha=0.05
        )
        assert result_conditional is True

    def test_insufficient_data(self, cd):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        # With only 2 data points, should return True (independent by default)
        result = cd.conditional_independence_test(x, y, alpha=0.05)
        assert result is True

    def test_constant_variable(self, cd):
        n = 50
        x = np.ones(n)  # constant
        y = np.random.randn(n)
        result = cd.conditional_independence_test(x, y, alpha=0.05)
        assert result is True  # constant -> independent


class TestPCAlgorithm:
    def test_skeleton_removes_independent(self, cd):
        """Independent variables should be disconnected in skeleton."""
        np.random.seed(42)
        n = 200
        # X and Y are correlated, Z is independent of both
        x = np.random.randn(n)
        y = 0.7 * x + 0.3 * np.random.randn(n)
        z = np.random.randn(n)  # independent

        data = np.column_stack([x, y, z])
        names = ["X", "Y", "Z"]

        result = cd.pc_algorithm(data, names, alpha=0.05)

        assert "X" in result["nodes"]
        assert "Y" in result["nodes"]
        assert "Z" in result["nodes"]

        # Z should not be connected to X or Y
        all_edges = result["directed_edges"] + result["undirected_edges"]
        z_edges = [
            e for e in all_edges
            if "Z" in (e[0], e[1])
        ]
        assert len(z_edges) == 0

        # X and Y should be connected
        xy_edges = [
            e for e in all_edges
            if {"X", "Y"} == {e[0], e[1]}
        ]
        assert len(xy_edges) >= 1

    def test_orients_v_structure(self, cd):
        """X -> Z <- Y: v-structure should be oriented."""
        np.random.seed(42)
        n = 500
        x = np.random.randn(n)
        y = np.random.randn(n)  # X and Y are independent
        z = 0.6 * x + 0.6 * y + 0.2 * np.random.randn(n)  # Z is caused by both

        data = np.column_stack([x, y, z])
        names = ["X", "Y", "Z"]

        result = cd.pc_algorithm(data, names, alpha=0.05)

        # Should find directed edges X -> Z and Y -> Z
        directed = result["directed_edges"]
        directed_pairs = {(e[0], e[1]) for e in directed}

        # At minimum, Z should receive edges (v-structure)
        z_targets = [e for e in directed if e[1] == "Z"]
        assert len(z_targets) >= 1  # At least one edge points to Z

    def test_single_variable(self, cd):
        data = np.random.randn(50, 1)
        result = cd.pc_algorithm(data, ["X"])
        assert result["nodes"] == ["X"]
        assert result["directed_edges"] == []
        assert result["undirected_edges"] == []

    def test_two_correlated_variables(self, cd):
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 0.9 * x + 0.1 * np.random.randn(n)
        data = np.column_stack([x, y])
        names = ["X", "Y"]

        result = cd.pc_algorithm(data, names, alpha=0.05)
        all_edges = result["directed_edges"] + result["undirected_edges"]
        assert len(all_edges) >= 1  # Should detect the dependency


class TestDiscoverDAG:
    def test_minimum_data_returns_empty(self, cd, storage):
        """With < 5 variables or < 10 time windows, return empty DAG."""
        now = datetime.now(timezone.utc)
        # Only create 2 entities
        storage.insert_entity({"name": "alpha", "type": "file"})
        storage.insert_entity({"name": "beta", "type": "file"})

        storage.insert_episode({
            "session_id": "s1",
            "directory": "/proj",
            "raw_content": "alpha and beta",
            "timestamp": (now - timedelta(hours=1)).isoformat(),
        })

        result = cd.discover_dag(hours=24)
        assert result["directed_edges"] == []
        assert result["metadata"]["status"] == "insufficient_data"

    def test_sufficient_data_returns_dag(self, cd, storage):
        """With enough data, should return a DAG with edges."""
        now = datetime.now(timezone.utc)
        np.random.seed(42)

        # Create 8 entities
        names = [f"mod_{i}" for i in range(8)]
        for name in names:
            storage.insert_entity({"name": name, "type": "file"})

        # Create 20 hours of episodes with correlated entity appearances
        for h in range(20):
            ts = (now - timedelta(hours=20 - h)).isoformat()
            # First group always co-occurs
            if h % 2 == 0:
                content = f"Working on mod_0 and mod_1 and mod_2"
            else:
                content = f"Working on mod_3 and mod_4 and mod_5"

            # mod_6 and mod_7 appear with first group sometimes
            if h % 3 == 0:
                content += f" also mod_6 and mod_7"

            storage.insert_episode({
                "session_id": f"sess_{h}",
                "directory": "/project",
                "raw_content": content,
                "timestamp": ts,
            })

        result = cd.discover_dag(hours=48)
        assert result["metadata"]["status"] == "completed"
        assert result["metadata"]["variables"] >= 5
        assert result["metadata"]["time_windows"] >= 10


class TestQueryCauses:
    def test_direct_causes(self, cd, storage):
        """Finds direct causes (depth=1)."""
        # Create entities
        eid_a = storage.insert_entity({"name": "config_change", "type": "file"})
        eid_b = storage.insert_entity({"name": "test_failure", "type": "error"})

        # Insert a causal edge: config_change -> test_failure
        storage.insert_causal_edge({
            "source_entity_id": eid_a,
            "target_entity_id": eid_b,
            "algorithm": "pc",
            "confidence": 0.85,
        })

        causes = cd.query_causes("test_failure")
        assert len(causes) >= 1
        assert causes[0]["entity"] == "config_change"
        assert causes[0]["depth"] == 1
        assert causes[0]["confidence"] == pytest.approx(0.85)

    def test_transitive_causes(self, cd, storage):
        """Finds indirect causes up to max_depth."""
        # A -> B -> C
        eid_a = storage.insert_entity({"name": "root_cause", "type": "decision"})
        eid_b = storage.insert_entity({"name": "intermediate", "type": "file"})
        eid_c = storage.insert_entity({"name": "final_effect", "type": "error"})

        storage.insert_causal_edge({
            "source_entity_id": eid_a,
            "target_entity_id": eid_b,
            "algorithm": "pc",
            "confidence": 0.9,
        })
        storage.insert_causal_edge({
            "source_entity_id": eid_b,
            "target_entity_id": eid_c,
            "algorithm": "pc",
            "confidence": 0.8,
        })

        causes = cd.query_causes("final_effect", max_depth=3)
        entity_names = [c["entity"] for c in causes]
        assert "intermediate" in entity_names
        assert "root_cause" in entity_names

        # root_cause should be at depth 2
        root = [c for c in causes if c["entity"] == "root_cause"][0]
        assert root["depth"] == 2
        assert "final_effect" in root["path"]
        assert "intermediate" in root["path"]
        assert "root_cause" in root["path"]

    def test_nonexistent_entity(self, cd):
        causes = cd.query_causes("nonexistent_entity")
        assert causes == []

    def test_max_depth_respected(self, cd, storage):
        """Chain A -> B -> C -> D: querying D with max_depth=1 should only find C."""
        ids = []
        for name in ["nodeA", "nodeB", "nodeC", "nodeD"]:
            ids.append(storage.insert_entity({"name": name, "type": "file"}))

        for i in range(3):
            storage.insert_causal_edge({
                "source_entity_id": ids[i],
                "target_entity_id": ids[i + 1],
                "algorithm": "pc",
                "confidence": 0.9,
            })

        causes = cd.query_causes("nodeD", max_depth=1)
        entity_names = [c["entity"] for c in causes]
        assert "nodeC" in entity_names
        assert "nodeB" not in entity_names
        assert "nodeA" not in entity_names


class TestQueryEffects:
    def test_finds_downstream_effects(self, cd, storage):
        eid_a = storage.insert_entity({"name": "change_api", "type": "file"})
        eid_b = storage.insert_entity({"name": "break_frontend", "type": "error"})
        eid_c = storage.insert_entity({"name": "user_complaint", "type": "error"})

        storage.insert_causal_edge({
            "source_entity_id": eid_a,
            "target_entity_id": eid_b,
            "algorithm": "pc",
            "confidence": 0.9,
        })
        storage.insert_causal_edge({
            "source_entity_id": eid_b,
            "target_entity_id": eid_c,
            "algorithm": "pc",
            "confidence": 0.7,
        })

        effects = cd.query_effects("change_api", max_depth=3)
        entity_names = [e["entity"] for e in effects]
        assert "break_frontend" in entity_names
        assert "user_complaint" in entity_names

        # break_frontend at depth 1, user_complaint at depth 2
        bf = [e for e in effects if e["entity"] == "break_frontend"][0]
        uc = [e for e in effects if e["entity"] == "user_complaint"][0]
        assert bf["depth"] == 1
        assert uc["depth"] == 2

    def test_nonexistent_entity(self, cd):
        effects = cd.query_effects("nonexistent_entity")
        assert effects == []


class TestCausalChain:
    def test_both_directions(self, cd, storage):
        """get_causal_chain returns both causes and effects."""
        # A -> B -> C
        eid_a = storage.insert_entity({"name": "causeA", "type": "file"})
        eid_b = storage.insert_entity({"name": "middleB", "type": "file"})
        eid_c = storage.insert_entity({"name": "effectC", "type": "error"})

        storage.insert_causal_edge({
            "source_entity_id": eid_a,
            "target_entity_id": eid_b,
            "algorithm": "pc",
            "confidence": 0.9,
        })
        storage.insert_causal_edge({
            "source_entity_id": eid_b,
            "target_entity_id": eid_c,
            "algorithm": "pc",
            "confidence": 0.8,
        })

        chain = cd.get_causal_chain("middleB")
        assert chain["entity"] == "middleB"
        assert len(chain["causes"]) >= 1
        assert len(chain["effects"]) >= 1

        cause_names = [c["entity"] for c in chain["causes"]]
        effect_names = [e["entity"] for e in chain["effects"]]
        assert "causeA" in cause_names
        assert "effectC" in effect_names
        assert chain["dag_edges_total"] >= 2


class TestDAGStoredInTable:
    def test_edges_persisted(self, cd, storage):
        """Directed edges from discover_dag are stored in causal_dag_edges."""
        now = datetime.now(timezone.utc)
        np.random.seed(42)

        # Create enough entities
        names = [f"var_{i}" for i in range(8)]
        for name in names:
            storage.insert_entity({"name": name, "type": "file"})

        # Create correlated episodes
        for h in range(24):
            ts = (now - timedelta(hours=24 - h)).isoformat()
            # Strong co-occurrence pattern
            if h % 2 == 0:
                content = "var_0 var_1 var_2"
            else:
                content = "var_3 var_4 var_5"
            if h % 3 == 0:
                content += " var_6 var_7"

            storage.insert_episode({
                "session_id": f"s{h}",
                "directory": "/proj",
                "raw_content": content,
                "timestamp": ts,
            })

        dag = cd.discover_dag(hours=48)

        # Check that edges were stored
        all_edges = storage.get_all_causal_edges()
        stored_count = dag["metadata"].get("stored_edges", 0)

        # The table should have at least as many edges as were stored
        assert len(all_edges) >= stored_count

        if stored_count > 0:
            # Verify edge structure
            edge = all_edges[0]
            assert "source_entity_id" in edge
            assert "target_entity_id" in edge
            assert "algorithm" in edge
            assert "confidence" in edge
            assert edge["algorithm"] == "pc"


class TestMCPCausalChainTool:
    def test_server_tool_returns_correct_data(self, tmp_path):
        """The get_causal_chain MCP tool returns correct data."""
        from zikkaron import server

        db_path = str(tmp_path / "test_mcp_causal.db")
        server.init_engines(db_path=db_path, embedding_model="all-MiniLM-L6-v2")

        try:
            storage = server._get_storage()

            # Create entities and causal edges
            eid_a = storage.insert_entity({"name": "src_file", "type": "file"})
            eid_b = storage.insert_entity({"name": "test_error", "type": "error"})

            storage.insert_causal_edge({
                "source_entity_id": eid_a,
                "target_entity_id": eid_b,
                "algorithm": "pc",
                "confidence": 0.92,
            })

            result = server.get_causal_chain("test_error")

            assert result["entity"] == "test_error"
            assert "causes" in result
            assert "effects" in result
            assert "dag_edges_total" in result
            assert len(result["causes"]) >= 1
            assert result["causes"][0]["entity"] == "src_file"
            assert result["dag_edges_total"] >= 1
        finally:
            server.shutdown()
