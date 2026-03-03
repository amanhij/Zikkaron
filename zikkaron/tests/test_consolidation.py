import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from zikkaron.config import Settings
from zikkaron.consolidation import AstrocyteEngine
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.storage import StorageEngine


@pytest.fixture
def storage(tmp_path):
    engine = StorageEngine(str(tmp_path / "test_consolidation.db"))
    yield engine
    engine.close()


@pytest.fixture
def settings():
    return Settings(
        DB_PATH=":memory:",
        IDLE_THRESHOLD_SECONDS=1,
        DECAY_FACTOR=0.95,
        COLD_THRESHOLD=0.05,
        DAEMON_CHECK_INTERVAL=1,
    )


@pytest.fixture
def embeddings():
    engine = EmbeddingEngine()
    engine._unavailable = True  # don't load real model in tests
    return engine


@pytest.fixture
def engine(storage, embeddings, settings):
    return AstrocyteEngine(storage, embeddings, settings)


def _hours_ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


class TestDecayApplication:
    def test_memory_heat_decreases(self, engine, storage):
        mid = storage.insert_memory({
            "content": "decay test",
            "directory_context": "/proj",
            "heat": 1.0,
            "last_accessed": _hours_ago(24),
        })
        engine.force_consolidate()
        mem = storage.get_memory(mid)
        # Enhanced decay: default confidence=1.0 slows decay slightly
        # effective_factor = 1.0 - (1.0 - 0.95) / (1.0 + 1.0 * 0.1)
        effective_factor = 1.0 - (1.0 - 0.95) / 1.1
        expected = 1.0 * (effective_factor ** 24)
        assert mem["heat"] == pytest.approx(expected, abs=1e-4)

    def test_entity_heat_decreases(self, engine, storage):
        eid = storage.insert_entity({
            "name": "test_func",
            "type": "function",
            "heat": 1.0,
            "last_accessed": _hours_ago(10),
        })
        engine.force_consolidate()
        ent = storage.get_entity_by_name("test_func")
        expected = 1.0 * (0.95 ** 10)
        assert ent["heat"] == pytest.approx(expected, abs=1e-4)

    def test_recent_memory_barely_decays(self, engine, storage):
        mid = storage.insert_memory({
            "content": "fresh memory",
            "directory_context": "/proj",
            "heat": 1.0,
            "last_accessed": _hours_ago(0.01),
        })
        engine.force_consolidate()
        mem = storage.get_memory(mid)
        assert mem["heat"] > 0.99


class TestColdArchival:
    def test_memory_archived_below_threshold(self, engine, storage):
        # 0.1 * 0.95^60 ≈ 0.0046, well below 0.05
        mid = storage.insert_memory({
            "content": "old memory",
            "directory_context": "/proj",
            "heat": 0.1,
            "last_accessed": _hours_ago(60),
        })
        engine.force_consolidate()
        mem = storage.get_memory(mid)
        assert mem["heat"] == 0.0

    def test_entity_archived_below_threshold(self, engine, storage):
        eid = storage.insert_entity({
            "name": "old_func",
            "type": "function",
            "heat": 0.1,
            "last_accessed": _hours_ago(60),
        })
        engine.force_consolidate()
        ent = storage.get_entity_by_name("old_func")
        assert ent["heat"] == 0.0
        assert ent["archived"] is True

    def test_hot_memory_not_archived(self, engine, storage):
        mid = storage.insert_memory({
            "content": "hot memory",
            "directory_context": "/proj",
            "heat": 0.9,
            "last_accessed": _hours_ago(1),
        })
        engine.force_consolidate()
        mem = storage.get_memory(mid)
        assert mem["heat"] > 0.05


class TestEntityExtraction:
    def test_extracts_file_paths(self, engine, storage):
        storage.insert_episode({
            "session_id": "sess1",
            "directory": "/proj",
            "raw_content": "Edited zikkaron/server.py and tests/test_server.py",
        })
        engine._last_consolidated_episode_id = 0
        engine.force_consolidate()
        assert storage.get_entity_by_name("zikkaron/server.py") is not None
        assert storage.get_entity_by_name("tests/test_server.py") is not None

    def test_extracts_function_defs(self, engine, storage):
        storage.insert_episode({
            "session_id": "sess1",
            "directory": "/proj",
            "raw_content": "def process_items():\n    pass\nclass DataProcessor:\n    pass",
        })
        engine._last_consolidated_episode_id = 0
        engine.force_consolidate()
        assert storage.get_entity_by_name("process_items") is not None
        assert storage.get_entity_by_name("DataProcessor") is not None

    def test_extracts_imports(self, engine, storage):
        storage.insert_episode({
            "session_id": "sess1",
            "directory": "/proj",
            "raw_content": "import os\nfrom pathlib import Path\n",
        })
        engine._last_consolidated_episode_id = 0
        engine.force_consolidate()
        assert storage.get_entity_by_name("os") is not None
        assert storage.get_entity_by_name("pathlib") is not None

    def test_extracts_errors(self, engine, storage):
        storage.insert_episode({
            "session_id": "sess1",
            "directory": "/proj",
            "raw_content": "Traceback (most recent call last)\nValueError: invalid input",
        })
        engine._last_consolidated_episode_id = 0
        engine.force_consolidate()
        assert storage.get_entity_by_name("Traceback") is not None
        assert storage.get_entity_by_name("ValueError") is not None

    def test_extracts_js_require(self, engine, storage):
        storage.insert_episode({
            "session_id": "sess1",
            "directory": "/proj",
            "raw_content": "const express = require('express')",
        })
        engine._last_consolidated_episode_id = 0
        engine.force_consolidate()
        assert storage.get_entity_by_name("express") is not None

    def test_reinforces_existing_entity(self, engine, storage):
        eid = storage.insert_entity({
            "name": "my_func",
            "type": "function",
            "heat": 0.5,
            "last_accessed": _hours_ago(10),
        })
        storage.insert_episode({
            "session_id": "sess1",
            "directory": "/proj",
            "raw_content": "def my_func():\n    pass",
        })
        engine._last_consolidated_episode_id = 0
        engine.force_consolidate()
        ent = storage.get_entity_by_name("my_func")
        # Heat was decayed from 0.5 then reinforced by +0.1
        # Decay alone: 0.5 * 0.95^10 ≈ 0.299, after reinforce ≈ 0.399
        # The exact value depends on ordering, but it should be > decayed value
        decayed_only = 0.5 * (0.95 ** 10)
        assert ent["heat"] > decayed_only


class TestRelationshipBuilding:
    def test_cooccurring_entities_get_relationship(self, engine, storage):
        storage.insert_episode({
            "session_id": "sess1",
            "directory": "/proj",
            "raw_content": "def handler():\n    pass\nimport flask",
        })
        engine._last_consolidated_episode_id = 0
        engine.force_consolidate()

        e1 = storage.get_entity_by_name("handler")
        e2 = storage.get_entity_by_name("flask")
        assert e1 is not None
        assert e2 is not None
        rel = storage.get_relationship_between(e1["id"], e2["id"])
        assert rel is not None
        assert rel["relationship_type"] == "co_occurrence"

    def test_repeated_cooccurrence_increases_weight(self, engine, storage):
        for i in range(3):
            storage.insert_episode({
                "session_id": f"sess{i}",
                "directory": "/proj",
                "raw_content": "def parse():\nimport json",
            })

        engine._last_consolidated_episode_id = 0
        engine.force_consolidate()

        e1 = storage.get_entity_by_name("parse")
        e2 = storage.get_entity_by_name("json")
        rel = storage.get_relationship_between(e1["id"], e2["id"])
        assert rel is not None
        # First episode creates at weight 1.0, next two reinforce by +1.0 each
        assert rel["weight"] == pytest.approx(3.0)


class TestDuplicateMerge:
    def test_near_identical_memories_merged(self, storage, settings):
        # Use a mock embeddings engine that reports high similarity
        mock_emb = MagicMock(spec=EmbeddingEngine)
        mock_emb.similarity.return_value = 0.98

        vec_a = np.ones(384, dtype=np.float32).tobytes()
        vec_b = np.ones(384, dtype=np.float32).tobytes()

        id_a = storage.insert_memory({
            "content": "how to configure the database",
            "embedding": vec_a,
            "directory_context": "/proj",
            "heat": 0.8,
        })
        id_b = storage.insert_memory({
            "content": "how to configure the database connection",
            "embedding": vec_b,
            "directory_context": "/proj",
            "heat": 0.5,
        })

        engine = AstrocyteEngine(storage, mock_emb, settings)
        engine.force_consolidate()

        # Higher-heat memory survives
        assert storage.get_memory(id_a) is not None
        assert storage.get_memory(id_b) is None

    def test_dissimilar_memories_kept(self, storage, settings):
        mock_emb = MagicMock(spec=EmbeddingEngine)
        mock_emb.similarity.return_value = 0.4

        vec_a = np.ones(384, dtype=np.float32).tobytes()
        vec_b = np.zeros(384, dtype=np.float32).tobytes()
        vec_b = np.ones(384, dtype=np.float32).tobytes()  # different content, low sim

        id_a = storage.insert_memory({
            "content": "database configuration",
            "embedding": vec_a,
            "directory_context": "/proj",
            "heat": 0.8,
        })
        id_b = storage.insert_memory({
            "content": "hiking trail map",
            "embedding": vec_b,
            "directory_context": "/proj",
            "heat": 0.5,
        })

        engine = AstrocyteEngine(storage, mock_emb, settings)
        engine.force_consolidate()

        assert storage.get_memory(id_a) is not None
        assert storage.get_memory(id_b) is not None


class TestDaemonLifecycle:
    def test_starts_and_stops(self, engine):
        engine.start()
        assert engine.is_running is True
        assert engine._thread is not None
        assert engine._thread.is_alive()

        engine.stop()
        assert engine.is_running is False
        assert engine._thread is None

    def test_double_start_is_noop(self, engine):
        engine.start()
        thread1 = engine._thread
        engine.start()
        assert engine._thread is thread1
        engine.stop()

    def test_stop_without_start(self, engine):
        engine.stop()  # should not raise
        assert engine.is_running is False


class TestActivityTracking:
    def test_record_activity_updates_timestamp(self, engine):
        old = engine.last_activity
        time.sleep(0.01)
        engine.record_activity()
        assert engine.last_activity > old

    def test_idle_detection(self, engine, storage):
        # Set last_activity far in the past so the daemon considers us idle
        engine.last_activity = datetime.now(timezone.utc) - timedelta(seconds=600)
        # Insert an episode so consolidation has something to do
        storage.insert_episode({
            "session_id": "sess1",
            "directory": "/proj",
            "raw_content": "def idle_test(): pass",
        })
        engine.start()
        engine._last_consolidated_episode_id = 0
        # Give the daemon loop time to fire
        time.sleep(2)
        engine.stop()

        # Entity should have been extracted during idle consolidation
        assert storage.get_entity_by_name("idle_test") is not None


class TestConsolidationLog:
    def test_log_entry_created(self, engine, storage):
        engine.force_consolidate()
        stats = storage.get_memory_stats()
        assert stats["last_consolidation"] is not None

    def test_force_consolidate_returns_stats(self, engine):
        result = engine.force_consolidate()
        assert "memories_added" in result
        assert "memories_updated" in result
        assert "memories_archived" in result
        assert "memories_deleted" in result
