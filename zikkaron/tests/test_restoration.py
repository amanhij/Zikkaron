"""Tests for Hippocampal Replay restoration engine."""

import json
import os
import tempfile

import pytest

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.storage import StorageEngine
from zikkaron.restoration import HippocampalReplay


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def engines(temp_db):
    settings = Settings(DB_PATH=temp_db)
    storage = StorageEngine(temp_db)
    embeddings = EmbeddingEngine()
    replay = HippocampalReplay(
        storage=storage,
        embeddings=embeddings,
        settings=settings,
    )
    return storage, embeddings, replay


class TestCheckpoints:
    def test_create_checkpoint(self, engines):
        storage, embeddings, replay = engines
        result = replay.create_checkpoint(
            directory="/test/project",
            current_task="Implementing feature X",
            files_being_edited=["src/main.py", "src/utils.py"],
            key_decisions=["Use async for IO"],
            next_steps=["Write tests"],
        )
        assert result["status"] == "created"
        assert result["checkpoint_id"] > 0

    def test_checkpoint_supersedes(self, engines):
        storage, embeddings, replay = engines
        replay.create_checkpoint(directory="/test", current_task="Task 1")
        replay.create_checkpoint(directory="/test", current_task="Task 2")

        active = storage.get_active_checkpoint()
        assert active is not None
        assert active["current_task"] == "Task 2"

        # Only one active checkpoint
        count = storage._conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE is_active = 1"
        ).fetchone()[0]
        assert count == 1

    def test_epoch_tracking(self, engines):
        storage, embeddings, replay = engines
        assert storage.get_current_epoch() == 0

        replay.create_checkpoint(directory="/test", current_task="T1")
        assert storage.get_current_epoch() == 0

        new_epoch = storage.increment_epoch()
        assert new_epoch == 1


class TestAnchor:
    def test_anchor_memory(self, engines):
        storage, embeddings, replay = engines
        mid = replay.anchor_memory(
            content="Always use PostgreSQL not SQLite",
            context="/test/project",
            tags=["database", "decision"],
            reason="Architecture decision",
        )
        assert mid > 0

        mem = storage.get_memory(mid)
        assert mem["is_protected"] == 1
        assert mem["importance"] == 1.0
        assert "_anchor" in mem["tags"]

    def test_anchor_heat(self, engines):
        storage, embeddings, replay = engines
        mid = replay.anchor_memory(
            content="Critical fact",
            context="/test",
            tags=[],
        )
        mem = storage.get_memory(mid)
        assert mem["heat"] == 1.0


class TestPreCompactDrain:
    def test_drain_creates_epoch(self, engines):
        storage, embeddings, replay = engines
        result = replay.pre_compact_drain("/test")
        assert result["status"] == "drained"
        assert result["epoch"] == 1

    def test_drain_auto_checkpoint(self, engines):
        storage, embeddings, replay = engines
        result = replay.pre_compact_drain("/test")
        assert result["auto_checkpoint_created"] is True

        active = storage.get_active_checkpoint()
        assert active is not None

    def test_drain_preserves_existing_checkpoint(self, engines):
        storage, embeddings, replay = engines
        replay.create_checkpoint(directory="/test", current_task="My task")
        result = replay.pre_compact_drain("/test")

        # Should update existing, not create new auto
        assert result["auto_checkpoint_created"] is False


class TestRestore:
    def test_restore_empty(self, engines):
        storage, embeddings, replay = engines
        result = replay.restore("/test")
        assert "formatted" in result
        assert result["anchored_memories"] == 0
        assert result["hot_memories"] == 0

    def test_restore_with_checkpoint(self, engines):
        storage, embeddings, replay = engines
        replay.create_checkpoint(
            directory="/test",
            current_task="Building feature X",
            files_being_edited=["main.py"],
        )
        result = replay.restore("/test")
        assert result["checkpoint"] is not None
        assert "Building feature X" in result["formatted"]

    def test_restore_includes_anchored(self, engines):
        storage, embeddings, replay = engines
        replay.anchor_memory(
            content="Use React not Vue",
            context="/test",
            tags=["framework"],
            reason="Team decision",
        )
        result = replay.restore("/test")
        assert result["anchored_memories"] >= 1
        assert "React" in result["formatted"]

    def test_full_drain_restore_cycle(self, engines):
        storage, embeddings, replay = engines

        # Simulate a session
        replay.create_checkpoint(
            directory="/test",
            current_task="Refactoring auth module",
            key_decisions=["Switch to JWT"],
        )
        replay.anchor_memory(
            content="API key stored in .env",
            context="/test",
            tags=["security"],
        )

        # Simulate compaction
        replay.pre_compact_drain("/test")

        # Restore
        result = replay.restore("/test")
        assert result["checkpoint"] is not None
        assert result["anchored_memories"] >= 1
        assert "Refactoring auth module" in result["formatted"]
        assert "API key" in result["formatted"]


class TestCLISubcommands:
    """Test the drain/restore CLI subcommands."""

    def test_cli_drain(self, temp_db):
        import subprocess
        result = subprocess.run(
            ["python", "-m", "zikkaron", "drain", "/test/project", "--db-path", temp_db],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "drained"
        assert data["epoch"] == 1

    def test_cli_restore(self, temp_db):
        import subprocess
        # First drain to create a checkpoint
        subprocess.run(
            ["python", "-m", "zikkaron", "drain", "/test/project", "--db-path", temp_db],
            capture_output=True, text=True, timeout=120,
        )
        # Then restore
        result = subprocess.run(
            ["python", "-m", "zikkaron", "restore", "/test/project", "--db-path", temp_db],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0
        assert "Zikkaron Context Restoration" in result.stdout

    def test_cli_restore_empty_db(self, temp_db):
        import subprocess
        result = subprocess.run(
            ["python", "-m", "zikkaron", "restore", "/test/project", "--db-path", temp_db],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0
        # Should still output restoration header even with empty DB
        assert "Zikkaron Context Restoration" in result.stdout


class TestAutoCheckpoint:
    def test_tool_call_tracking(self, engines):
        storage, embeddings, replay = engines
        assert not replay.should_auto_checkpoint()

        for _ in range(50):
            replay.record_tool_call()

        assert replay.should_auto_checkpoint()

    def test_reset_after_checkpoint(self, engines):
        storage, embeddings, replay = engines
        for _ in range(50):
            replay.record_tool_call()
        assert replay.should_auto_checkpoint()

        replay.create_checkpoint(directory="/test", current_task="T")
        assert not replay.should_auto_checkpoint()
