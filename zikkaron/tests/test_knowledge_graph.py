"""Tests for the typed temporal knowledge graph."""

from datetime import datetime, timedelta, timezone

import pytest

from zikkaron.config import Settings
from zikkaron.knowledge_graph import KnowledgeGraph, VALID_REL_TYPES
from zikkaron.storage import StorageEngine


@pytest.fixture
def storage(tmp_path):
    engine = StorageEngine(str(tmp_path / "test_kg.db"))
    yield engine
    engine.close()


@pytest.fixture
def settings():
    return Settings(DB_PATH=":memory:", CAUSAL_THRESHOLD=3)


@pytest.fixture
def graph(storage, settings):
    return KnowledgeGraph(storage, settings)


class TestAddTypedRelationship:
    def test_creates_relationship_with_type(self, graph, storage):
        rid = graph.add_relationship("module_a", "module_b", "imports")
        row = storage._conn.execute(
            "SELECT * FROM relationships WHERE id = ?", (rid,)
        ).fetchone()
        row = storage._row_to_dict(row)
        assert row["relationship_type"] == "imports"
        assert row["weight"] == 1.0

    def test_all_valid_types_accepted(self, graph):
        for i, rt in enumerate(sorted(VALID_REL_TYPES)):
            rid = graph.add_relationship(f"src_{i}", f"tgt_{i}", rt)
            assert rid > 0

    def test_invalid_type_raises(self, graph):
        with pytest.raises(ValueError, match="Invalid rel_type"):
            graph.add_relationship("a", "b", "nonexistent_type")

    def test_stores_event_time_and_record_time(self, graph, storage):
        event = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        rid = graph.add_relationship("x", "y", "calls", event_time=event)
        row = storage._conn.execute(
            "SELECT event_time, record_time FROM relationships WHERE id = ?",
            (rid,),
        ).fetchone()
        assert row[0] is not None  # event_time
        assert row[1] is not None  # record_time
        assert "2025-06-15" in row[0]

    def test_confidence_stored(self, graph, storage):
        rid = graph.add_relationship("a", "b", "derived_from", confidence=0.75)
        row = storage._conn.execute(
            "SELECT confidence FROM relationships WHERE id = ?", (rid,)
        ).fetchone()
        assert row[0] == pytest.approx(0.75)


class TestRelationshipReinforcement:
    def test_weight_increases_on_repeat(self, graph, storage):
        rid1 = graph.add_relationship("foo", "bar", "co_occurrence")
        rid2 = graph.add_relationship("foo", "bar", "co_occurrence")
        assert rid1 == rid2  # same relationship returned
        row = storage._conn.execute(
            "SELECT weight FROM relationships WHERE id = ?", (rid1,)
        ).fetchone()
        assert row[0] == pytest.approx(2.0)

    def test_triple_reinforcement(self, graph, storage):
        rid = graph.add_relationship("a", "b", "imports")
        graph.add_relationship("a", "b", "imports")
        graph.add_relationship("a", "b", "imports")
        row = storage._conn.execute(
            "SELECT weight FROM relationships WHERE id = ?", (rid,)
        ).fetchone()
        assert row[0] == pytest.approx(3.0)

    def test_different_types_are_separate(self, graph, storage):
        rid1 = graph.add_relationship("x", "y", "imports")
        rid2 = graph.add_relationship("x", "y", "calls")
        assert rid1 != rid2


class TestBiTemporal:
    def test_event_time_vs_record_time(self, graph, storage):
        past = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rid = graph.add_relationship("a", "b", "imports", event_time=past)
        row = storage._conn.execute(
            "SELECT event_time, record_time FROM relationships WHERE id = ?",
            (rid,),
        ).fetchone()
        event_time = datetime.fromisoformat(row[0])
        record_time = datetime.fromisoformat(row[1])
        assert event_time.year == 2024
        assert record_time.year >= 2026  # recorded now

    def test_relationships_at_time_filters(self, graph):
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        t3 = datetime(2025, 12, 1, tzinfo=timezone.utc)

        graph.add_relationship("srv", "db", "calls", event_time=t1)
        graph.add_relationship("srv", "cache", "calls", event_time=t3)

        # Query at t2 — should see db but not cache
        results = graph.get_relationships_at_time("srv", t2)
        names = [r.get("target_name") or r.get("source_name") for r in results]
        assert any("db" in str(n) for n in names)
        # cache relationship has event_time t3 > t2, should be excluded
        target_names = [r["target_name"] for r in results]
        assert "cache" not in target_names

    def test_relationship_history_returns_all(self, graph):
        for i in range(3):
            graph.add_relationship("mod", "lib", "imports")

        history = graph.get_relationship_history("mod", "lib")
        assert len(history) >= 1
        assert history[0]["weight"] == pytest.approx(3.0)

    def test_history_nonexistent_returns_empty(self, graph):
        assert graph.get_relationship_history("none1", "none2") == []


class TestCausalDetection:
    def test_repeated_a_before_b_creates_causal(self, graph, storage, settings):
        # Create entities with co_occurrence weight >= CAUSAL_THRESHOLD
        eid_a = storage.insert_entity({"name": "config_change", "type": "file"})
        eid_b = storage.insert_entity({"name": "ImportError", "type": "error"})
        storage.insert_relationship({
            "source_entity_id": eid_a,
            "target_entity_id": eid_b,
            "relationship_type": "co_occurrence",
            "weight": 3.0,
        })

        # Create episodes where config_change appears BEFORE ImportError
        for i in range(4):
            ts = (datetime.now(timezone.utc) - timedelta(hours=10 - i)).isoformat()
            storage.insert_episode({
                "session_id": f"sess_{i}",
                "directory": "/proj",
                "raw_content": f"Modified config_change in step {i}, then got ImportError",
                "timestamp": ts,
            })

        created = graph.detect_causality()
        assert created >= 1

        # Verify causal edge exists
        rows = storage._conn.execute(
            "SELECT * FROM relationships WHERE relationship_type = 'caused_by' "
            "AND is_causal = 1"
        ).fetchall()
        assert len(rows) >= 1

    def test_no_causality_below_threshold(self, graph, storage):
        eid_a = storage.insert_entity({"name": "alpha", "type": "variable"})
        eid_b = storage.insert_entity({"name": "beta", "type": "variable"})
        storage.insert_relationship({
            "source_entity_id": eid_a,
            "target_entity_id": eid_b,
            "relationship_type": "co_occurrence",
            "weight": 1.0,  # below threshold of 3
        })

        storage.insert_episode({
            "session_id": "s1",
            "directory": "/proj",
            "raw_content": "alpha then beta",
        })

        created = graph.detect_causality()
        assert created == 0


class TestTypedEntityExtraction:
    def test_import_pattern(self, graph):
        content = "from pathlib import Path\nimport os"
        results = graph.extract_entities_typed(content, "/proj")
        names = [r[0] for r in results]
        assert "pathlib" in names
        assert "Path" in names
        assert "os" in names

    def test_import_relationship_context(self, graph):
        content = "from flask import Blueprint"
        results = graph.extract_entities_typed(content, "/proj")
        # Path should have "imports" context
        imports = [r for r in results if r[2] == "imports"]
        assert len(imports) >= 1
        assert imports[0][0] == "Blueprint"

    def test_def_and_call_pattern(self, graph):
        content = "def process():\n    return 1\n\nresult = process()\n"
        results = graph.extract_entities_typed(content, "/proj")
        names = [r[0] for r in results]
        assert "process" in names
        # Should have a "calls" context entry
        call_entries = [r for r in results if r[2] == "calls"]
        assert len(call_entries) >= 1

    def test_error_fix_pattern(self, graph):
        content = "Fixed the ValueError by updating the parser"
        results = graph.extract_entities_typed(content, "/proj")
        resolved = [r for r in results if r[2] == "resolved_by"]
        assert len(resolved) >= 1

    def test_decision_pattern(self, graph):
        content = "decided to use Redis instead of Memcached"
        results = graph.extract_entities_typed(content, "/proj")
        decisions = [r for r in results if r[2] == "decided_to_use"]
        names = [d[0] for d in decisions]
        assert "Redis" in names
        assert "Memcached" in names


class TestGraphTraversal:
    def test_get_neighbors_depth_1(self, graph):
        graph.add_relationship("A", "B", "imports")
        graph.add_relationship("A", "C", "calls")
        graph.add_relationship("B", "D", "imports")

        neighbors = graph.get_neighbors("A", depth=1)
        names = [n["entity_name"] for n in neighbors]
        assert "B" in names
        assert "C" in names
        assert "D" not in names  # depth 1 only

    def test_get_neighbors_depth_2(self, graph):
        graph.add_relationship("A", "B", "imports")
        graph.add_relationship("B", "C", "calls")
        graph.add_relationship("C", "D", "imports")

        neighbors = graph.get_neighbors("A", depth=2)
        names = [n["entity_name"] for n in neighbors]
        assert "B" in names
        assert "C" in names
        assert "D" not in names  # depth 3

    def test_get_neighbors_filtered_by_type(self, graph):
        graph.add_relationship("A", "B", "imports")
        graph.add_relationship("A", "C", "calls")

        neighbors = graph.get_neighbors("A", depth=1, rel_types=["imports"])
        names = [n["entity_name"] for n in neighbors]
        assert "B" in names
        assert "C" not in names

    def test_nonexistent_entity_returns_empty(self, graph):
        assert graph.get_neighbors("nonexistent") == []


class TestSubgraphExtraction:
    def test_subgraph_contains_seed_nodes(self, graph):
        graph.add_relationship("X", "Y", "imports")
        graph.add_relationship("Y", "Z", "calls")

        sg = graph.get_subgraph(["X"], depth=2)
        node_names = [n["name"] for n in sg["nodes"]]
        assert "X" in node_names
        assert "Y" in node_names
        assert "Z" in node_names

    def test_subgraph_edges(self, graph):
        graph.add_relationship("A", "B", "imports")
        graph.add_relationship("B", "C", "calls")

        sg = graph.get_subgraph(["A"], depth=2)
        assert len(sg["edges"]) >= 2
        edge_types = [e["relationship_type"] for e in sg["edges"]]
        assert "imports" in edge_types
        assert "calls" in edge_types

    def test_subgraph_multiple_seeds(self, graph):
        graph.add_relationship("P", "Q", "imports")
        graph.add_relationship("R", "S", "calls")

        sg = graph.get_subgraph(["P", "R"], depth=1)
        node_names = [n["name"] for n in sg["nodes"]]
        assert "P" in node_names
        assert "Q" in node_names
        assert "R" in node_names
        assert "S" in node_names

    def test_subgraph_nonexistent_seed(self, graph):
        sg = graph.get_subgraph(["nonexistent"], depth=1)
        assert sg["nodes"] == []
        assert sg["edges"] == []
