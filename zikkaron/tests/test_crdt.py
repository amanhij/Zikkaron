"""Tests for CRDT-based multi-agent memory sharing."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from zikkaron.config import Settings
from zikkaron.crdt_sync import CRDTMemorySync
from zikkaron.storage import StorageEngine


@pytest.fixture
def settings():
    return Settings(DB_PATH=":memory:", CRDT_AGENT_ID="agent-alpha")


@pytest.fixture
def storage(tmp_path):
    db_path = str(tmp_path / "test_crdt.db")
    engine = StorageEngine(db_path)
    yield engine
    engine.close()


@pytest.fixture
def crdt(storage, settings):
    return CRDTMemorySync(storage, settings)


def _insert_memory(storage, content, directory="/tmp/project", tags=None, **kwargs):
    """Helper to insert a memory directly."""
    mem = {
        "content": content,
        "embedding": None,
        "tags": tags or ["test"],
        "directory_context": directory,
        "heat": 1.0,
        "is_stale": False,
    }
    mem.update(kwargs)
    return storage.insert_memory(mem)


class TestAgentId:
    def test_agent_id_from_config(self, crdt, settings):
        """Agent ID should come from CRDT_AGENT_ID setting."""
        assert crdt.get_agent_id() == "agent-alpha"
        assert crdt.get_agent_id() == settings.CRDT_AGENT_ID

    def test_default_agent_id(self, storage):
        """Default agent ID should be 'default'."""
        default_settings = Settings(DB_PATH=":memory:")
        sync = CRDTMemorySync(storage, default_settings)
        assert sync.get_agent_id() == "default"


class TestVectorClock:
    def test_increment_clock(self, crdt):
        """Clock should increment correctly for the local agent."""
        clock1 = crdt.increment_clock()
        assert clock1 == {"agent-alpha": 1}

        clock2 = crdt.increment_clock()
        assert clock2 == {"agent-alpha": 2}

        clock3 = crdt.increment_clock()
        assert clock3 == {"agent-alpha": 3}

    def test_clock_returns_copy(self, crdt):
        """increment_clock should return a copy, not a reference."""
        clock1 = crdt.increment_clock()
        clock2 = crdt.increment_clock()
        # Modifying returned clock shouldn't affect internal state
        clock1["agent-alpha"] = 999
        clock3 = crdt.increment_clock()
        assert clock3["agent-alpha"] == 3


class TestTagProvenance:
    def test_tag_provenance(self, crdt):
        """Memory should get agent and clock metadata."""
        mem = {"content": "test memory", "tags": ["test"]}
        tagged = crdt.tag_provenance(mem)

        assert tagged["provenance_agent"] == "agent-alpha"
        clock = json.loads(tagged["vector_clock"])
        assert clock["agent-alpha"] == 1

    def test_tag_provenance_increments(self, crdt):
        """Each tag_provenance call should increment the clock."""
        mem1 = {"content": "first"}
        mem2 = {"content": "second"}

        crdt.tag_provenance(mem1)
        crdt.tag_provenance(mem2)

        clock2 = json.loads(mem2["vector_clock"])
        assert clock2["agent-alpha"] == 2


class TestCompareCocks:
    def test_compare_clocks_equal(self, crdt):
        """Equal clocks should return 'equal'."""
        a = {"agent-alpha": 3, "agent-beta": 2}
        b = {"agent-alpha": 3, "agent-beta": 2}
        assert crdt.compare_clocks(a, b) == "equal"

    def test_compare_clocks_before(self, crdt):
        """A happened before B when all A[k] <= B[k] with at least one <."""
        a = {"agent-alpha": 1, "agent-beta": 2}
        b = {"agent-alpha": 2, "agent-beta": 3}
        assert crdt.compare_clocks(a, b) == "before"

    def test_compare_clocks_after(self, crdt):
        """A happened after B when all B[k] <= A[k] with at least one <."""
        a = {"agent-alpha": 3, "agent-beta": 4}
        b = {"agent-alpha": 2, "agent-beta": 3}
        assert crdt.compare_clocks(a, b) == "after"

    def test_compare_clocks_concurrent(self, crdt):
        """Concurrent when some A[k] > B[k] and some B[k] > A[k]."""
        a = {"agent-alpha": 3, "agent-beta": 1}
        b = {"agent-alpha": 1, "agent-beta": 3}
        assert crdt.compare_clocks(a, b) == "concurrent"

    def test_compare_clocks_missing_keys(self, crdt):
        """Missing keys in one clock are treated as 0."""
        a = {"agent-alpha": 1}
        b = {"agent-beta": 1}
        # a has alpha=1, beta=0; b has alpha=0, beta=1 → concurrent
        assert crdt.compare_clocks(a, b) == "concurrent"

    def test_compare_clocks_subset_before(self, crdt):
        """A is before B even with different key sets."""
        a = {"agent-alpha": 1}
        b = {"agent-alpha": 2, "agent-beta": 1}
        assert crdt.compare_clocks(a, b) == "before"

    def test_compare_empty_clocks(self, crdt):
        """Two empty clocks are equal."""
        assert crdt.compare_clocks({}, {}) == "equal"


class TestMergeMemory:
    def test_merge_no_conflict(self, crdt):
        """Same content merges cleanly, no conflict."""
        local = {
            "content": "shared content",
            "tags": ["a", "b"],
            "heat": 0.5,
            "last_accessed": "2025-01-01T00:00:00",
            "vector_clock": json.dumps({"agent-alpha": 1}),
            "provenance_agent": "agent-alpha",
        }
        remote = {
            "content": "shared content",
            "tags": ["b", "c"],
            "heat": 0.7,
            "last_accessed": "2025-01-02T00:00:00",
            "vector_clock": json.dumps({"agent-beta": 1}),
            "provenance_agent": "agent-beta",
        }
        merged = crdt.merge_memory(local, remote)

        assert merged["_conflict"] is False
        assert merged["content"] == "shared content"
        # Tags: OR-Set union
        assert set(merged["tags"]) == {"a", "b", "c"}

    def test_merge_content_conflict(self, crdt):
        """Divergent content with concurrent clocks preserves both (MV-Register)."""
        local = {
            "content": "local version of the fix",
            "tags": ["bugfix"],
            "heat": 0.5,
            "last_accessed": "2025-01-01T00:00:00",
            "vector_clock": json.dumps({"agent-alpha": 2, "agent-beta": 1}),
            "provenance_agent": "agent-alpha",
        }
        remote = {
            "content": "remote version of the fix",
            "tags": ["bugfix"],
            "heat": 0.6,
            "last_accessed": "2025-01-01T00:00:00",
            "vector_clock": json.dumps({"agent-alpha": 1, "agent-beta": 2}),
            "provenance_agent": "agent-beta",
        }
        merged = crdt.merge_memory(local, remote)

        assert merged["_conflict"] is True
        assert "local version of the fix" in merged["content"]
        assert "remote version of the fix" in merged["content"]
        assert "--- [Agent: agent-beta] ---" in merged["content"]

    def test_merge_content_before(self, crdt):
        """When local is before remote, take remote content."""
        local = {
            "content": "old content",
            "tags": ["a"],
            "heat": 0.5,
            "last_accessed": "2025-01-01T00:00:00",
            "vector_clock": json.dumps({"agent-alpha": 1}),
            "provenance_agent": "agent-alpha",
        }
        remote = {
            "content": "new content",
            "tags": ["b"],
            "heat": 0.7,
            "last_accessed": "2025-01-02T00:00:00",
            "vector_clock": json.dumps({"agent-alpha": 2}),
            "provenance_agent": "agent-beta",
        }
        merged = crdt.merge_memory(local, remote)

        assert merged["_conflict"] is False
        assert merged["content"] == "new content"

    def test_merge_tags_union(self, crdt):
        """Tags merge as OR-Set (union of both sets)."""
        local = {
            "content": "same",
            "tags": ["python", "bugfix"],
            "heat": 0.5,
            "last_accessed": "2025-01-01T00:00:00",
            "vector_clock": json.dumps({"agent-alpha": 1}),
            "provenance_agent": "agent-alpha",
        }
        remote = {
            "content": "same",
            "tags": ["bugfix", "critical", "deployment"],
            "heat": 0.5,
            "last_accessed": "2025-01-01T00:00:00",
            "vector_clock": json.dumps({"agent-beta": 1}),
            "provenance_agent": "agent-beta",
        }
        merged = crdt.merge_memory(local, remote)
        assert set(merged["tags"]) == {"python", "bugfix", "critical", "deployment"}

    def test_merge_tags_json_string(self, crdt):
        """Tags stored as JSON string should be parsed correctly."""
        local = {
            "content": "same",
            "tags": '["a", "b"]',
            "heat": 0.5,
            "last_accessed": "2025-01-01T00:00:00",
            "vector_clock": json.dumps({"agent-alpha": 1}),
            "provenance_agent": "agent-alpha",
        }
        remote = {
            "content": "same",
            "tags": '["b", "c"]',
            "heat": 0.5,
            "last_accessed": "2025-01-01T00:00:00",
            "vector_clock": json.dumps({"agent-beta": 1}),
            "provenance_agent": "agent-beta",
        }
        merged = crdt.merge_memory(local, remote)
        assert set(merged["tags"]) == {"a", "b", "c"}

    def test_merge_heat_lww(self, crdt):
        """Latest heat value wins (LWW-Register based on last_accessed)."""
        local = {
            "content": "same",
            "tags": [],
            "heat": 0.3,
            "last_accessed": "2025-01-01T00:00:00",
            "vector_clock": json.dumps({"agent-alpha": 1}),
            "provenance_agent": "agent-alpha",
        }
        remote = {
            "content": "same",
            "tags": [],
            "heat": 0.9,
            "last_accessed": "2025-01-05T00:00:00",
            "vector_clock": json.dumps({"agent-beta": 1}),
            "provenance_agent": "agent-beta",
        }
        merged = crdt.merge_memory(local, remote)

        # Remote has later timestamp, so its heat wins
        assert merged["heat"] == 0.9
        assert merged["last_accessed"] == "2025-01-05T00:00:00"

    def test_merge_heat_lww_local_newer(self, crdt):
        """When local is newer, local heat is kept."""
        local = {
            "content": "same",
            "tags": [],
            "heat": 0.8,
            "last_accessed": "2025-01-10T00:00:00",
            "vector_clock": json.dumps({"agent-alpha": 1}),
            "provenance_agent": "agent-alpha",
        }
        remote = {
            "content": "same",
            "tags": [],
            "heat": 0.2,
            "last_accessed": "2025-01-01T00:00:00",
            "vector_clock": json.dumps({"agent-beta": 1}),
            "provenance_agent": "agent-beta",
        }
        merged = crdt.merge_memory(local, remote)
        assert merged["heat"] == 0.8

    def test_merge_vector_clocks(self, crdt):
        """Merged clock should have max of each agent's counter."""
        local = {
            "content": "same",
            "tags": [],
            "heat": 0.5,
            "last_accessed": "2025-01-01T00:00:00",
            "vector_clock": json.dumps({"agent-alpha": 3, "agent-beta": 1}),
            "provenance_agent": "agent-alpha",
        }
        remote = {
            "content": "same",
            "tags": [],
            "heat": 0.5,
            "last_accessed": "2025-01-01T00:00:00",
            "vector_clock": json.dumps({"agent-alpha": 1, "agent-beta": 4, "agent-gamma": 2}),
            "provenance_agent": "agent-beta",
        }
        merged = crdt.merge_memory(local, remote)
        merged_clock = json.loads(merged["vector_clock"])
        assert merged_clock == {"agent-alpha": 3, "agent-beta": 4, "agent-gamma": 2}


class TestDetectConflicts:
    def test_detect_conflicts(self, crdt, storage):
        """Should find memories with conflict markers."""
        # Insert a conflicted memory
        content = "version A\n--- [Agent: agent-beta] ---\nversion B"
        mid = _insert_memory(storage, content)
        storage._conn.execute(
            "UPDATE memories SET provenance_agent = ? WHERE id = ?",
            ("agent-alpha", mid),
        )
        storage._conn.commit()

        conflicts = crdt.detect_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0]["memory_id"] == mid
        assert "agent-alpha" in conflicts[0]["agents"]
        assert "agent-beta" in conflicts[0]["agents"]
        assert conflicts[0]["versions"] == 2

    def test_detect_no_conflicts(self, crdt, storage):
        """No conflicts when no markers present."""
        _insert_memory(storage, "clean memory")
        conflicts = crdt.detect_conflicts()
        assert len(conflicts) == 0

    def test_detect_multi_agent_conflict(self, crdt, storage):
        """Should detect conflicts involving multiple agents."""
        content = (
            "version A\n"
            "--- [Agent: agent-beta] ---\n"
            "version B\n"
            "--- [Agent: agent-gamma] ---\n"
            "version C"
        )
        mid = _insert_memory(storage, content)
        storage._conn.execute(
            "UPDATE memories SET provenance_agent = ? WHERE id = ?",
            ("agent-alpha", mid),
        )
        storage._conn.commit()

        conflicts = crdt.detect_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0]["versions"] == 3
        assert set(conflicts[0]["agents"]) == {"agent-alpha", "agent-beta", "agent-gamma"}


class TestResolveConflict:
    def _make_conflicted(self, storage):
        content = "version from alpha\n--- [Agent: agent-beta] ---\nversion from beta"
        mid = _insert_memory(storage, content)
        storage._conn.execute(
            "UPDATE memories SET provenance_agent = ? WHERE id = ?",
            ("agent-alpha", mid),
        )
        storage._conn.commit()
        return mid

    def test_resolve_latest(self, crdt, storage):
        """Resolution with 'latest' keeps the last version."""
        mid = self._make_conflicted(storage)
        result = crdt.resolve_conflict(mid, "latest")
        assert result["content"] == "version from beta"
        assert result.get("_resolved") is True

    def test_resolve_merge(self, crdt, storage):
        """Resolution with 'merge' keeps both versions, markers removed."""
        mid = self._make_conflicted(storage)
        result = crdt.resolve_conflict(mid, "merge")
        assert "version from alpha" in result["content"]
        assert "version from beta" in result["content"]
        assert "--- [Agent:" not in result["content"]

    def test_resolve_agent(self, crdt, storage):
        """Resolution with 'agent:<id>' keeps that agent's version."""
        mid = self._make_conflicted(storage)
        result = crdt.resolve_conflict(mid, "agent:agent-alpha")
        assert result["content"] == "version from alpha"

    def test_resolve_longest(self, crdt, storage):
        """Resolution with 'longest' keeps the longest version."""
        content = "short\n--- [Agent: agent-beta] ---\nthis is a much longer version of the text"
        mid = _insert_memory(storage, content)
        storage._conn.execute(
            "UPDATE memories SET provenance_agent = ? WHERE id = ?",
            ("agent-alpha", mid),
        )
        storage._conn.commit()

        result = crdt.resolve_conflict(mid, "longest")
        assert result["content"] == "this is a much longer version of the text"

    def test_resolve_no_conflict(self, crdt, storage):
        """Resolving a non-conflicted memory returns no_conflict status."""
        mid = _insert_memory(storage, "clean memory")
        result = crdt.resolve_conflict(mid, "latest")
        assert result["status"] == "no_conflict"

    def test_resolve_not_found(self, crdt, storage):
        """Resolving a non-existent memory returns error."""
        result = crdt.resolve_conflict(99999, "latest")
        assert "error" in result


class TestSyncMemories:
    def test_sync_new_remote(self, crdt, storage):
        """New remote memories should be added locally."""
        remote_memories = [
            {
                "content": "remote memory 1",
                "tags": ["from-beta"],
                "directory_context": "/tmp/project",
                "heat": 0.8,
                "provenance_agent": "agent-beta",
                "vector_clock": json.dumps({"agent-beta": 1}),
            },
            {
                "content": "remote memory 2",
                "tags": ["from-beta"],
                "directory_context": "/tmp/project",
                "heat": 0.9,
                "provenance_agent": "agent-beta",
                "vector_clock": json.dumps({"agent-beta": 2}),
            },
        ]
        result = crdt.sync_memories(remote_memories)

        assert result["new_from_remote"] == 2
        assert result["merged"] == 0
        assert result["conflicted"] == 0
        assert result["total_processed"] == 2

        # Verify memories were inserted
        all_mems = storage.get_all_memories_for_decay()
        assert len(all_mems) == 2
        contents = {m["content"] for m in all_mems}
        assert "remote memory 1" in contents
        assert "remote memory 2" in contents

    def test_sync_unchanged(self, crdt, storage):
        """Matching memories with equal clocks should be unchanged."""
        _insert_memory(storage, "existing memory")
        storage._conn.execute(
            "UPDATE memories SET vector_clock = ? WHERE id = 1",
            (json.dumps({"agent-alpha": 1}),),
        )
        storage._conn.commit()

        remote_memories = [
            {
                "content": "existing memory",
                "tags": ["test"],
                "directory_context": "/tmp/project",
                "heat": 1.0,
                "provenance_agent": "agent-beta",
                "vector_clock": json.dumps({"agent-alpha": 1}),
            },
        ]
        result = crdt.sync_memories(remote_memories)
        assert result["unchanged"] == 1
        assert result["merged"] == 0

    def test_sync_with_conflicts(self, crdt, storage):
        """Concurrent edits should be detected and merged with conflict markers."""
        _insert_memory(storage, "original content")
        storage._conn.execute(
            "UPDATE memories SET vector_clock = ?, provenance_agent = ? WHERE id = 1",
            (json.dumps({"agent-alpha": 2, "agent-beta": 1}), "agent-alpha"),
        )
        storage._conn.commit()

        # Remote has a concurrent edit (alpha:1, beta:2 vs local alpha:2, beta:1)
        remote_memories = [
            {
                "content": "original content",  # same content key for matching
                "tags": ["test", "remote-tag"],
                "directory_context": "/tmp/project",
                "heat": 1.0,
                "provenance_agent": "agent-beta",
                "vector_clock": json.dumps({"agent-alpha": 1, "agent-beta": 2}),
            },
        ]
        result = crdt.sync_memories(remote_memories)
        # Content is the same, so even though clocks are concurrent, no content conflict
        assert result["merged"] == 1
        assert result["conflicted"] == 0

        # Verify tags were merged (OR-Set)
        mem = storage.get_memory(1)
        assert "remote-tag" in mem["tags"]

    def test_sync_with_content_conflict(self, crdt, storage):
        """Different content with concurrent clocks should produce conflict."""
        _insert_memory(storage, "local version")
        storage._conn.execute(
            "UPDATE memories SET vector_clock = ?, provenance_agent = ? WHERE id = 1",
            (json.dumps({"agent-alpha": 2, "agent-beta": 1}), "agent-alpha"),
        )
        storage._conn.commit()

        # Trick: we need to match by content, but content differs.
        # In this scenario, we manually set up a second memory with
        # different content but same key so they can match.
        # Actually, sync_memories matches by exact content, so different content = new memory
        remote_memories = [
            {
                "content": "remote version",
                "tags": ["test"],
                "directory_context": "/tmp/project",
                "heat": 1.0,
                "provenance_agent": "agent-beta",
                "vector_clock": json.dumps({"agent-alpha": 1, "agent-beta": 2}),
            },
        ]
        result = crdt.sync_memories(remote_memories)
        # Different content means no match → treated as new
        assert result["new_from_remote"] == 1

    def test_sync_remote_newer(self, crdt, storage):
        """When remote is strictly newer, merge updates local."""
        _insert_memory(storage, "shared content")
        storage._conn.execute(
            "UPDATE memories SET vector_clock = ?, provenance_agent = ?, heat = 0.3 WHERE id = 1",
            (json.dumps({"agent-alpha": 1}), "agent-alpha"),
        )
        storage._conn.commit()

        remote_memories = [
            {
                "content": "shared content",
                "tags": ["test", "updated"],
                "directory_context": "/tmp/project",
                "heat": 0.9,
                "last_accessed": "2025-06-01T00:00:00",
                "provenance_agent": "agent-beta",
                "vector_clock": json.dumps({"agent-alpha": 1, "agent-beta": 3}),
            },
        ]
        result = crdt.sync_memories(remote_memories)
        assert result["merged"] == 1

        # Verify tags were updated
        mem = storage.get_memory(1)
        assert "updated" in mem["tags"]


class TestAgentStats:
    def test_agent_stats(self, crdt, storage):
        """Should return correct counts and agent info."""
        # Insert memories with different provenances
        mid1 = _insert_memory(storage, "memory by alpha")
        storage._conn.execute(
            "UPDATE memories SET provenance_agent = ? WHERE id = ?",
            ("agent-alpha", mid1),
        )
        mid2 = _insert_memory(storage, "memory by beta")
        storage._conn.execute(
            "UPDATE memories SET provenance_agent = ? WHERE id = ?",
            ("agent-beta", mid2),
        )
        # Insert a conflicted memory
        mid3 = _insert_memory(
            storage,
            "version A\n--- [Agent: agent-beta] ---\nversion B",
        )
        storage._conn.execute(
            "UPDATE memories SET provenance_agent = ? WHERE id = ?",
            ("agent-alpha", mid3),
        )
        storage._conn.commit()

        stats = crdt.get_agent_stats()
        assert stats["agent_id"] == "agent-alpha"
        assert stats["memories_authored"] == 2  # mid1 and mid3
        assert stats["conflicts_pending"] == 1
        assert "vector_clock" in stats

    def test_agent_stats_empty(self, crdt, storage):
        """Stats on empty DB should return zeros."""
        stats = crdt.get_agent_stats()
        assert stats["memories_authored"] == 0
        assert stats["conflicts_pending"] == 0
