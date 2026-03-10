import json

import pytest

from zikkaron.storage import StorageEngine, _FTS_STOP_WORDS


@pytest.fixture
def storage(tmp_path):
    db_path = str(tmp_path / "test_memory.db")
    engine = StorageEngine(db_path)
    yield engine
    engine.close()


def _make_memory(content="test memory", directory="/tmp/project", **kwargs):
    base = {
        "content": content,
        "directory_context": directory,
        "tags": ["test"],
    }
    base.update(kwargs)
    return base


class TestSchemaCreation:
    def test_all_tables_exist(self, storage):
        tables = storage._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = sorted(t["name"] for t in tables)
        expected = sorted([
            "episodes",
            "entities",
            "relationships",
            "memories",
            "consolidation_log",
            "file_hashes",
            "memory_clusters",
            "prospective_memories",
            "narrative_entries",
            "astrocyte_processes",
        ])
        for name in expected:
            assert name in table_names, f"Table {name} not found"

    def test_fts_virtual_table_exists(self, storage):
        tables = storage._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        ).fetchall()
        assert len(tables) == 1

    def test_v2_memory_columns_exist(self, storage):
        cols = {
            row["name"]
            for row in storage._conn.execute("PRAGMA table_info(memories)").fetchall()
        }
        v2_cols = [
            "surprise_score", "importance", "emotional_valence", "confidence",
            "access_count", "useful_count", "embedding_model", "contextual_prefix",
            "cluster_id", "is_prospective", "trigger_condition", "narrative_weight",
        ]
        for col in v2_cols:
            assert col in cols, f"Column memories.{col} not found"

    def test_v2_entity_columns_exist(self, storage):
        cols = {
            row["name"]
            for row in storage._conn.execute("PRAGMA table_info(entities)").fetchall()
        }
        for col in ("causal_weight", "domain"):
            assert col in cols, f"Column entities.{col} not found"

    def test_v2_relationship_columns_exist(self, storage):
        cols = {
            row["name"]
            for row in storage._conn.execute("PRAGMA table_info(relationships)").fetchall()
        }
        for col in ("event_time", "record_time", "is_causal", "confidence"):
            assert col in cols, f"Column relationships.{col} not found"

    def test_v2_indexes_exist(self, storage):
        indexes = {
            row["name"]
            for row in storage._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
            if row["name"]
        }
        for idx in (
            "idx_memories_cluster",
            "idx_memories_surprise",
            "idx_relationships_type",
            "idx_prospective_active",
        ):
            assert idx in indexes, f"Index {idx} not found"

    def test_migration_is_idempotent(self, storage):
        """Running migrate again should not raise."""
        storage._migrate_schema()
        storage._migrate_schema()


class TestMemoryCRUD:
    def test_insert_and_get_memory(self, storage):
        mem = _make_memory(content="pytest is great", tags=["python", "testing"])
        mem_id = storage.insert_memory(mem)
        assert mem_id is not None

        retrieved = storage.get_memory(mem_id)
        assert retrieved is not None
        assert retrieved["content"] == "pytest is great"
        assert retrieved["directory_context"] == "/tmp/project"
        assert retrieved["tags"] == ["python", "testing"]
        assert retrieved["heat"] == 1.0
        assert retrieved["is_stale"] is False

    def test_get_nonexistent_memory(self, storage):
        assert storage.get_memory(9999) is None

    def test_delete_memory(self, storage):
        mem_id = storage.insert_memory(_make_memory())
        storage.delete_memory(mem_id)
        assert storage.get_memory(mem_id) is None


class TestFTSSearch:
    def test_fts_search(self, storage):
        storage.insert_memory(_make_memory(content="fastapi server configuration"))
        storage.insert_memory(_make_memory(content="database migration script"))
        storage.insert_memory(_make_memory(content="react component rendering"))

        results = storage.search_memories_fts("fastapi")
        assert len(results) == 1
        assert results[0]["content"] == "fastapi server configuration"

    def test_fts_search_no_results(self, storage):
        storage.insert_memory(_make_memory(content="python coding"))
        results = storage.search_memories_fts("javascript")
        assert len(results) == 0

    def test_fts_respects_min_heat(self, storage):
        storage.insert_memory(_make_memory(content="hot memory about fastapi", heat=0.8))
        storage.insert_memory(_make_memory(content="cold memory about fastapi", heat=0.01))

        results = storage.search_memories_fts("fastapi", min_heat=0.1)
        assert len(results) == 1
        assert results[0]["heat"] == 0.8


class TestFTSPreprocessing:
    def test_fts_camelcase_splitting(self, storage):
        storage.insert_memory(_make_memory(content="DatabaseConnection pool timeout"))
        results = storage.search_memories_fts("DatabaseConnection")
        assert len(results) == 1
        assert results[0]["content"] == "DatabaseConnection pool timeout"

        # Also findable via a split sub-term
        results2 = storage.search_memories_fts("Database")
        assert len(results2) == 1

    def test_fts_snake_case_splitting(self, storage):
        storage.insert_memory(_make_memory(content="auth_middleware handles tokens"))
        results = storage.search_memories_fts("auth_middleware")
        assert len(results) == 1

        # Individual parts are searchable
        results2 = storage.search_memories_fts("auth")
        assert len(results2) == 1
        results3 = storage.search_memories_fts("middleware")
        assert len(results3) == 1

    def test_fts_stop_words_expanded(self, storage):
        # Coding-domain stop words should be in the set
        coding_stops = {"use", "using", "used", "just", "get", "code", "file", "thing"}
        assert coding_stops.issubset(_FTS_STOP_WORDS)

        # A query of only stop words should fall back to original query
        storage.insert_memory(_make_memory(content="just use the code"))
        result = storage.search_memories_fts("just use the code")
        assert len(result) == 1


class TestMemoryHeatFiltering:
    def test_get_memories_by_heat(self, storage):
        storage.insert_memory(_make_memory(content="hot", heat=0.9))
        storage.insert_memory(_make_memory(content="warm", heat=0.5))
        storage.insert_memory(_make_memory(content="cold", heat=0.02))

        hot = storage.get_memories_by_heat(min_heat=0.7)
        assert len(hot) == 1
        assert hot[0]["content"] == "hot"

        warm_plus = storage.get_memories_by_heat(min_heat=0.4)
        assert len(warm_plus) == 2

    def test_update_memory_heat(self, storage):
        mem_id = storage.insert_memory(_make_memory(heat=1.0))
        storage.update_memory_heat(mem_id, 0.3)
        updated = storage.get_memory(mem_id)
        assert updated["heat"] == 0.3

    def test_update_memory_staleness(self, storage):
        mem_id = storage.insert_memory(_make_memory())
        assert storage.get_memory(mem_id)["is_stale"] is False

        storage.update_memory_staleness(mem_id, True)
        assert storage.get_memory(mem_id)["is_stale"] is True

    def test_get_stale_memories(self, storage):
        storage.insert_memory(_make_memory(content="fresh"))
        stale_id = storage.insert_memory(_make_memory(content="stale"))
        storage.update_memory_staleness(stale_id, True)

        stale = storage.get_stale_memories()
        assert len(stale) == 1
        assert stale[0]["content"] == "stale"


class TestEntities:
    def test_insert_and_get_entity(self, storage):
        entity_id = storage.insert_entity({
            "name": "storage.py",
            "type": "file",
        })
        assert entity_id is not None

        retrieved = storage.get_entity_by_name("storage.py")
        assert retrieved is not None
        assert retrieved["name"] == "storage.py"
        assert retrieved["type"] == "file"
        assert retrieved["heat"] == 1.0
        assert retrieved["archived"] is False

    def test_get_nonexistent_entity(self, storage):
        assert storage.get_entity_by_name("nonexistent") is None

    def test_archive_entity(self, storage):
        entity_id = storage.insert_entity({"name": "old_func", "type": "function"})
        storage.archive_entity(entity_id)
        entity = storage.get_entity_by_name("old_func")
        assert entity["archived"] is True

    def test_get_all_entities_excludes_archived(self, storage):
        storage.insert_entity({"name": "active", "type": "file"})
        archived_id = storage.insert_entity({"name": "archived", "type": "file"})
        storage.archive_entity(archived_id)

        active = storage.get_all_entities()
        assert len(active) == 1
        assert active[0]["name"] == "active"

        all_entities = storage.get_all_entities(include_archived=True)
        assert len(all_entities) == 2

    def test_update_entity_heat(self, storage):
        entity_id = storage.insert_entity({"name": "func", "type": "function"})
        storage.update_entity_heat(entity_id, 0.5)
        entity = storage.get_entity_by_name("func")
        assert entity["heat"] == 0.5


class TestFileHashOperations:
    def test_upsert_and_get_file_hash(self, storage):
        storage.upsert_file_hash("/path/to/file.py", "abc123")
        assert storage.get_file_hash("/path/to/file.py") == "abc123"

    def test_upsert_updates_existing(self, storage):
        storage.upsert_file_hash("/path/to/file.py", "abc123")
        storage.upsert_file_hash("/path/to/file.py", "def456")
        assert storage.get_file_hash("/path/to/file.py") == "def456"

    def test_get_nonexistent_hash(self, storage):
        assert storage.get_file_hash("/no/such/file") is None

    def test_get_memories_by_file_hash(self, storage):
        storage.insert_memory(_make_memory(content="linked", file_hash="hash1"))
        storage.insert_memory(_make_memory(content="unlinked", file_hash="hash2"))

        results = storage.get_memories_by_file_hash("hash1")
        assert len(results) == 1
        assert results[0]["content"] == "linked"


class TestMemoryStats:
    def test_empty_stats(self, storage):
        stats = storage.get_memory_stats()
        assert stats["total_memories"] == 0
        assert stats["active_count"] == 0
        assert stats["archived_count"] == 0
        assert stats["stale_count"] == 0
        assert stats["avg_heat"] == 0.0
        assert stats["last_consolidation"] is None

    def test_stats_with_data(self, storage):
        storage.insert_memory(_make_memory(content="active hot", heat=0.8))
        storage.insert_memory(_make_memory(content="active warm", heat=0.5))
        cold_id = storage.insert_memory(_make_memory(content="cold", heat=0.01))
        stale_id = storage.insert_memory(_make_memory(content="stale", heat=0.6))
        storage.update_memory_staleness(stale_id, True)

        storage.insert_consolidation_log({
            "memories_added": 4,
            "duration_ms": 120,
        })

        stats = storage.get_memory_stats()
        assert stats["total_memories"] == 4
        assert stats["stale_count"] == 1
        assert stats["archived_count"] == 1  # cold < 0.05
        assert stats["last_consolidation"] is not None


class TestDirectoryMemories:
    def test_get_memories_for_directory(self, storage):
        storage.insert_memory(_make_memory(content="proj a", directory="/proj/a"))
        storage.insert_memory(_make_memory(content="proj b", directory="/proj/b"))

        results = storage.get_memories_for_directory("/proj/a")
        assert len(results) == 1
        assert results[0]["content"] == "proj a"


class TestRelationships:
    def test_insert_relationship(self, storage):
        src = storage.insert_entity({"name": "main.py", "type": "file"})
        tgt = storage.insert_entity({"name": "utils.py", "type": "file"})
        rel_id = storage.insert_relationship({
            "source_entity_id": src,
            "target_entity_id": tgt,
            "relationship_type": "imports",
        })
        assert rel_id is not None


class TestEpisodes:
    def test_insert_episode(self, storage):
        ep_id = storage.insert_episode({
            "session_id": "sess-001",
            "directory": "/home/user/project",
            "raw_content": "git status\n# output here",
        })
        assert ep_id is not None


class TestWALMode:
    def test_wal_mode(self, storage):
        mode = storage._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"


class TestConsolidationLog:
    def test_insert_consolidation_log(self, storage):
        log_id = storage.insert_consolidation_log({
            "memories_added": 10,
            "memories_updated": 3,
            "memories_archived": 2,
            "memories_deleted": 1,
            "duration_ms": 500,
        })
        assert log_id is not None


class TestMemoryClusters:
    def test_insert_and_get_cluster(self, storage):
        cid = storage.insert_cluster({
            "name": "python-debugging",
            "level": 0,
            "summary": "Memories about debugging Python code",
        })
        assert cid is not None
        cluster = storage.get_cluster(cid)
        assert cluster["name"] == "python-debugging"
        assert cluster["level"] == 0
        assert cluster["summary"] == "Memories about debugging Python code"
        assert cluster["member_count"] == 0
        assert cluster["heat"] == 1.0

    def test_get_nonexistent_cluster(self, storage):
        assert storage.get_cluster(9999) is None

    def test_get_clusters_by_level(self, storage):
        storage.insert_cluster({"name": "leaf-1", "level": 0})
        storage.insert_cluster({"name": "leaf-2", "level": 0})
        storage.insert_cluster({"name": "root-1", "level": 2})

        leaves = storage.get_clusters_by_level(0)
        assert len(leaves) == 2
        roots = storage.get_clusters_by_level(2)
        assert len(roots) == 1
        assert roots[0]["name"] == "root-1"

    def test_update_cluster(self, storage):
        cid = storage.insert_cluster({"name": "old-name", "summary": "old"})
        storage.update_cluster(cid, {"name": "new-name", "member_count": 5})
        cluster = storage.get_cluster(cid)
        assert cluster["name"] == "new-name"
        assert cluster["member_count"] == 5

    def test_update_cluster_ignores_invalid_fields(self, storage):
        cid = storage.insert_cluster({"name": "test"})
        storage.update_cluster(cid, {"nonexistent_field": "nope"})
        cluster = storage.get_cluster(cid)
        assert cluster["name"] == "test"


class TestProspectiveMemories:
    def test_insert_and_get_active(self, storage):
        pm_id = storage.insert_prospective_memory({
            "content": "Remind about testing",
            "trigger_condition": "pytest",
            "trigger_type": "keyword_match",
            "target_directory": "/home/user/project",
        })
        assert pm_id is not None

        active = storage.get_active_prospective_memories()
        assert len(active) == 1
        assert active[0]["content"] == "Remind about testing"
        assert active[0]["trigger_type"] == "keyword_match"
        assert active[0]["is_active"] is True
        assert active[0]["triggered_count"] == 0

    def test_trigger_prospective_memory(self, storage):
        pm_id = storage.insert_prospective_memory({
            "content": "Check docs",
            "trigger_condition": "docs/",
            "trigger_type": "directory_match",
        })
        storage.trigger_prospective_memory(pm_id)

        active = storage.get_active_prospective_memories()
        assert active[0]["triggered_count"] == 1
        assert active[0]["triggered_at"] is not None

    def test_trigger_increments_count(self, storage):
        pm_id = storage.insert_prospective_memory({
            "content": "Check docs",
            "trigger_condition": "docs/",
            "trigger_type": "directory_match",
        })
        storage.trigger_prospective_memory(pm_id)
        storage.trigger_prospective_memory(pm_id)

        active = storage.get_active_prospective_memories()
        assert active[0]["triggered_count"] == 2

    def test_inactive_not_returned(self, storage):
        storage.insert_prospective_memory({
            "content": "inactive",
            "trigger_condition": "x",
            "trigger_type": "keyword_match",
            "is_active": False,
        })
        assert len(storage.get_active_prospective_memories()) == 0


class TestNarrativeEntries:
    def test_insert_and_get_narratives(self, storage):
        nid = storage.insert_narrative_entry({
            "directory_context": "/home/user/project",
            "summary": "Set up project structure and CI",
            "period_start": "2026-03-01T00:00:00",
            "period_end": "2026-03-01T23:59:59",
            "key_decisions": ["Use FastAPI", "SQLite WAL"],
            "key_events": ["Init repo", "First test pass"],
        })
        assert nid is not None

        entries = storage.get_narratives_for_directory("/home/user/project")
        assert len(entries) == 1
        assert entries[0]["summary"] == "Set up project structure and CI"
        assert entries[0]["key_decisions"] == ["Use FastAPI", "SQLite WAL"]
        assert entries[0]["key_events"] == ["Init repo", "First test pass"]
        assert entries[0]["heat"] == 1.0

    def test_narratives_filtered_by_directory(self, storage):
        storage.insert_narrative_entry({
            "directory_context": "/proj/a",
            "summary": "A stuff",
            "period_start": "2026-03-01T00:00:00",
            "period_end": "2026-03-01T23:59:59",
        })
        storage.insert_narrative_entry({
            "directory_context": "/proj/b",
            "summary": "B stuff",
            "period_start": "2026-03-01T00:00:00",
            "period_end": "2026-03-01T23:59:59",
        })
        results = storage.get_narratives_for_directory("/proj/a")
        assert len(results) == 1
        assert results[0]["summary"] == "A stuff"


class TestAstrocyteProcesses:
    def test_insert_and_get_processes(self, storage):
        pid = storage.insert_astrocyte_process({
            "name": "consolidator",
            "domain": "memory-management",
            "specialization": "heat decay",
            "memory_ids": [1, 2, 3],
            "entity_ids": [10, 20],
        })
        assert pid is not None

        procs = storage.get_astrocyte_processes()
        assert len(procs) == 1
        assert procs[0]["name"] == "consolidator"
        assert procs[0]["domain"] == "memory-management"
        assert procs[0]["memory_ids"] == [1, 2, 3]
        assert procs[0]["entity_ids"] == [10, 20]
        assert procs[0]["heat"] == 1.0

    def test_update_astrocyte_process(self, storage):
        pid = storage.insert_astrocyte_process({
            "name": "proc1",
            "domain": "test",
        })
        storage.update_astrocyte_process(pid, {
            "heat": 0.5,
            "memory_ids": [4, 5],
            "specialization": "clustering",
        })
        procs = storage.get_astrocyte_processes()
        assert procs[0]["heat"] == 0.5
        assert procs[0]["memory_ids"] == [4, 5]
        assert procs[0]["specialization"] == "clustering"

    def test_update_ignores_invalid_fields(self, storage):
        pid = storage.insert_astrocyte_process({
            "name": "proc1",
            "domain": "test",
        })
        storage.update_astrocyte_process(pid, {"bad_field": "nope"})
        procs = storage.get_astrocyte_processes()
        assert procs[0]["name"] == "proc1"


class TestV2MemoryDefaults:
    def test_new_memory_has_v2_defaults(self, storage):
        mem_id = storage.insert_memory(_make_memory())
        mem = storage.get_memory(mem_id)
        assert mem["surprise_score"] == 0.0
        assert mem["importance"] == 0.5
        assert mem["emotional_valence"] == 0.0
        assert mem["confidence"] == 1.0
        assert mem["access_count"] == 0
        assert mem["useful_count"] == 0
        assert mem["embedding_model"] is None
        assert mem["cluster_id"] is None
        assert mem["is_prospective"] is False
        assert mem["narrative_weight"] == 0.0


class TestContextManager:
    def test_context_manager(self, tmp_path):
        db_path = str(tmp_path / "ctx_test.db")
        with StorageEngine(db_path) as engine:
            engine.insert_memory(_make_memory())
