"""Tests for the AstrocytePool domain-aware consolidation system."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from zikkaron.astrocyte_pool import AstrocytePool, DOMAIN_DEFINITIONS
from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.storage import StorageEngine
from zikkaron.thermodynamics import MemoryThermodynamics


def _hours_ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


@pytest.fixture
def storage(tmp_path):
    engine = StorageEngine(str(tmp_path / "test_pool.db"))
    yield engine
    engine.close()


@pytest.fixture
def settings():
    return Settings(
        DB_PATH=":memory:",
        DECAY_FACTOR=0.95,
        COLD_THRESHOLD=0.05,
    )


@pytest.fixture
def embeddings():
    engine = EmbeddingEngine()
    engine._unavailable = True
    return engine


@pytest.fixture
def knowledge_graph(storage, settings):
    return KnowledgeGraph(storage, settings)


@pytest.fixture
def thermo(storage, embeddings, settings):
    return MemoryThermodynamics(storage, embeddings, settings)


@pytest.fixture
def pool(storage, embeddings, knowledge_graph, thermo, settings):
    p = AstrocytePool(storage, embeddings, knowledge_graph, thermo, settings)
    p.init_processes()
    return p


class TestProcessInitialization:
    def test_creates_four_processes(self, pool):
        stats = pool.get_process_stats()
        assert len(stats) == 4

    def test_correct_domain_names(self, pool):
        names = {s["name"] for s in pool.get_process_stats()}
        assert names == {"code-patterns", "decisions", "errors", "dependencies"}

    def test_processes_persisted_in_db(self, storage, pool):
        procs = storage.get_astrocyte_processes()
        assert len(procs) == 4

    def test_idempotent_initialization(self, storage, embeddings, knowledge_graph, thermo, settings):
        """Calling init_processes twice doesn't duplicate."""
        p1 = AstrocytePool(storage, embeddings, knowledge_graph, thermo, settings)
        p1.init_processes()
        p2 = AstrocytePool(storage, embeddings, knowledge_graph, thermo, settings)
        p2.init_processes()
        procs = storage.get_astrocyte_processes()
        assert len(procs) == 4

    def test_decay_multipliers(self, pool):
        stats = pool.get_process_stats()
        by_name = {s["name"]: s for s in stats}
        assert by_name["code-patterns"]["decay_multiplier"] == 1.0
        assert by_name["decisions"]["decay_multiplier"] == 1.5
        assert by_name["errors"]["decay_multiplier"] == 0.7
        assert by_name["dependencies"]["decay_multiplier"] == 1.2


class TestMemoryAssignment:
    def test_code_memory_assigned_to_code_patterns(self, pool, storage):
        mid = storage.insert_memory({
            "content": "def process_items():\n    class DataHandler:\n        pass",
            "directory_context": "/proj",
            "heat": 1.0,
        })
        mem = storage.get_memory(mid)
        assigned = pool.assign_memory(mem)
        assert "code-patterns" in assigned

    def test_error_memory_assigned_to_errors(self, pool, storage):
        mid = storage.insert_memory({
            "content": "Got a TypeError exception when running the tests. Fixed the bug by checking types.",
            "directory_context": "/proj",
            "heat": 1.0,
        })
        mem = storage.get_memory(mid)
        assigned = pool.assign_memory(mem)
        assert "errors" in assigned

    def test_decision_memory_assigned_to_decisions(self, pool, storage):
        mid = storage.insert_memory({
            "content": "We decided to use Redis instead of Memcached. Chose Redis for its data structures.",
            "directory_context": "/proj",
            "heat": 1.0,
        })
        mem = storage.get_memory(mid)
        assigned = pool.assign_memory(mem)
        assert "decisions" in assigned

    def test_dependency_memory_assigned_to_dependencies(self, pool, storage):
        mid = storage.insert_memory({
            "content": "pip install flask==2.0. Updated the package dependency for the web framework.",
            "directory_context": "/proj",
            "heat": 1.0,
        })
        mem = storage.get_memory(mid)
        assigned = pool.assign_memory(mem)
        assert "dependencies" in assigned

    def test_generic_memory_defaults_to_code_patterns(self, pool, storage):
        mid = storage.insert_memory({
            "content": "The sky is blue today.",
            "directory_context": "/proj",
            "heat": 1.0,
        })
        mem = storage.get_memory(mid)
        assigned = pool.assign_memory(mem)
        assert "code-patterns" in assigned


class TestDomainConsolidation:
    def test_consolidation_runs_per_domain(self, pool, storage):
        mid = storage.insert_memory({
            "content": "def compute(): pass\nclass Handler: pass",
            "directory_context": "/proj",
            "heat": 0.8,
            "last_accessed": _hours_ago(10),
        })
        mem = storage.get_memory(mid)
        pool.assign_memory(mem)

        result = pool.consolidate_domain("code-patterns")
        assert result["process"] == "code-patterns"
        assert result["memories_processed"] >= 1

    def test_domain_decay_applied(self, pool, storage):
        """Errors domain (multiplier 0.7) should decay faster than decisions (1.5)."""
        mid_err = storage.insert_memory({
            "content": "RuntimeError exception in handler. Bug crash failure.",
            "directory_context": "/proj",
            "heat": 0.9,
            "last_accessed": _hours_ago(24),
        })
        mid_dec = storage.insert_memory({
            "content": "We decided to use PostgreSQL. Chose it for reliability. Selected approach.",
            "directory_context": "/proj",
            "heat": 0.9,
            "last_accessed": _hours_ago(24),
        })

        pool.assign_memory(storage.get_memory(mid_err))
        pool.assign_memory(storage.get_memory(mid_dec))

        pool.consolidate_domain("errors")
        pool.consolidate_domain("decisions")

        err_mem = storage.get_memory(mid_err)
        dec_mem = storage.get_memory(mid_dec)

        # Error decayed with multiplier 0.7 (faster), decision with 1.5 (slower)
        # So error memory should have lower heat
        assert err_mem["heat"] < dec_mem["heat"]

    def test_unknown_domain_returns_error(self, pool):
        result = pool.consolidate_domain("nonexistent")
        assert "error" in result

    def test_deleted_memory_cleaned_up(self, pool, storage):
        mid = storage.insert_memory({
            "content": "def helper(): pass\nfunction doStuff() {}",
            "directory_context": "/proj",
            "heat": 0.5,
        })
        mem = storage.get_memory(mid)
        pool.assign_memory(mem)
        storage.delete_memory(mid)

        result = pool.consolidate_domain("code-patterns")
        # Should process 0 memories since the one assigned was deleted
        assert result["memories_processed"] == 0


class TestConsensusRetrieval:
    def test_single_domain_retrieval(self, pool, storage):
        mid = storage.insert_memory({
            "content": "def process_data(): implemented data pipeline function",
            "directory_context": "/proj",
            "heat": 0.8,
        })
        mem = storage.get_memory(mid)
        pool.assign_memory(mem)

        results = pool.consensus_retrieve("process_data function", top_k=5)
        assert len(results) >= 1
        assert results[0]["id"] == mid

    def test_cross_domain_boost(self, pool, storage):
        """Memory in multiple domains gets boosted score."""
        # This memory has both error AND code keywords
        mid = storage.insert_memory({
            "content": "def fix_handler(): resolved TypeError exception in the function implementation",
            "directory_context": "/proj",
            "heat": 0.8,
        })
        mem = storage.get_memory(mid)
        assigned = pool.assign_memory(mem)
        # Should be in both code-patterns and errors
        assert len(assigned) >= 2

        results = pool.consensus_retrieve("fix handler error", top_k=5)
        assert len(results) >= 1
        # The memory should have voting_domains from multiple processes
        assert len(results[0]["voting_domains"]) >= 2

    def test_empty_pool_returns_empty(self, pool):
        results = pool.consensus_retrieve("anything", top_k=5)
        assert results == []

    def test_consensus_score_present(self, pool, storage):
        mid = storage.insert_memory({
            "content": "decided to use FastAPI framework for the API. Chose it over Flask.",
            "directory_context": "/proj",
            "heat": 0.9,
        })
        pool.assign_memory(storage.get_memory(mid))

        results = pool.consensus_retrieve("API framework decision", top_k=5)
        assert len(results) >= 1
        assert "consensus_score" in results[0]
        assert results[0]["consensus_score"] > 0


class TestCrossDomainMemory:
    def test_memory_in_multiple_domains(self, pool, storage):
        """A memory with both error and dependency keywords should land in both."""
        mid = storage.insert_memory({
            "content": "pip install requests failed with ImportError exception. Package dependency broken.",
            "directory_context": "/proj",
            "heat": 0.9,
        })
        mem = storage.get_memory(mid)
        assigned = pool.assign_memory(mem)
        assert "errors" in assigned
        assert "dependencies" in assigned

    def test_cross_domain_consolidation_independent(self, pool, storage):
        """Each domain consolidates its copy independently."""
        mid = storage.insert_memory({
            "content": "pip install numpy failed with ModuleNotFoundError. Fix dependency issue.",
            "directory_context": "/proj",
            "heat": 0.8,
            "last_accessed": _hours_ago(12),
        })
        pool.assign_memory(storage.get_memory(mid))

        err_stats = pool.consolidate_domain("errors")
        dep_stats = pool.consolidate_domain("dependencies")

        # Both should have processed the memory
        assert err_stats["memories_processed"] >= 1
        assert dep_stats["memories_processed"] >= 1


class TestProcessStats:
    def test_stats_returned_for_all_processes(self, pool):
        stats = pool.get_process_stats()
        assert len(stats) == 4
        for s in stats:
            assert "name" in s
            assert "memory_count" in s
            assert "avg_heat" in s
            assert "last_active" in s
            assert "domain" in s

    def test_stats_reflect_assignments(self, pool, storage):
        mid = storage.insert_memory({
            "content": "def compute(): implementing a new function method",
            "directory_context": "/proj",
            "heat": 0.7,
        })
        pool.assign_memory(storage.get_memory(mid))

        stats = pool.get_process_stats()
        code_stats = next(s for s in stats if s["name"] == "code-patterns")
        assert code_stats["memory_count"] >= 1
        assert code_stats["avg_heat"] > 0

    def test_empty_process_stats(self, pool):
        stats = pool.get_process_stats()
        for s in stats:
            assert s["memory_count"] == 0
            assert s["avg_heat"] == 0.0
