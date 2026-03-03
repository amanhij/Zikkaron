"""Tests for the autobiographical narrative engine."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from zikkaron.config import Settings
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.narrative import NarrativeEngine
from zikkaron.storage import StorageEngine


@pytest.fixture
def storage(tmp_path):
    engine = StorageEngine(str(tmp_path / "test_narrative.db"))
    yield engine
    engine.close()


@pytest.fixture
def settings():
    return Settings(
        DB_PATH=":memory:",
        NARRATIVE_INTERVAL_HOURS=24,
    )


@pytest.fixture
def knowledge_graph(storage, settings):
    return KnowledgeGraph(storage, settings)


@pytest.fixture
def narrative(storage, knowledge_graph, settings):
    return NarrativeEngine(storage, knowledge_graph, settings)


def _recent_timestamp(hours_ago: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _make_embedding() -> bytes:
    vec = np.ones(384, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tobytes()


class TestGenerateNarrative:
    def test_generate_narrative(self, narrative, storage):
        """Narrative generated for directory with memories."""
        # Insert recent memories
        for i in range(3):
            storage.insert_memory({
                "content": f"Implemented feature {i} for the API server",
                "embedding": _make_embedding(),
                "directory_context": "/home/proj",
                "heat": 0.8,
                "created_at": _recent_timestamp(hours_ago=2),
                "last_accessed": _recent_timestamp(hours_ago=1),
            })

        entry = narrative.generate_narrative("/home/proj", period_hours=24)

        assert entry["id"] is not None
        assert "/home/proj" in entry["summary"]
        assert "3 memories recorded" in entry["summary"]
        assert entry["directory_context"] == "/home/proj"

    def test_narrative_with_decisions(self, narrative, storage):
        """Narrative captures decision keywords."""
        storage.insert_memory({
            "content": "decided to use FastAPI instead of Flask for the backend",
            "embedding": _make_embedding(),
            "directory_context": "/proj",
            "heat": 0.9,
            "created_at": _recent_timestamp(hours_ago=1),
            "last_accessed": _recent_timestamp(),
        })

        entry = narrative.generate_narrative("/proj", period_hours=24)
        assert "Key decisions" in entry["summary"]
        assert len(entry["key_decisions"]) >= 1

    def test_narrative_with_events(self, narrative, storage):
        """Narrative captures notable events from high-importance memories."""
        mid = storage.insert_memory({
            "content": "Fixed critical authentication bug in login handler",
            "embedding": _make_embedding(),
            "directory_context": "/proj",
            "heat": 0.9,
            "created_at": _recent_timestamp(hours_ago=1),
            "last_accessed": _recent_timestamp(),
        })
        storage.update_memory_scores(mid, importance=0.9)

        entry = narrative.generate_narrative("/proj", period_hours=24)
        assert "Notable events" in entry["summary"]

    def test_empty_directory_narrative(self, narrative, storage):
        """Narrative for empty directory should still work."""
        entry = narrative.generate_narrative("/empty", period_hours=24)
        assert "0 memories recorded" in entry["summary"]


class TestProjectStory:
    def test_project_story(self, narrative, storage):
        """Multiple narratives combine into coherent story."""
        # Create first narrative entry manually
        storage.insert_narrative_entry({
            "directory_context": "/proj",
            "summary": "In /proj, during the last 24 hours: 5 memories recorded.",
            "period_start": _recent_timestamp(hours_ago=48),
            "period_end": _recent_timestamp(hours_ago=24),
            "key_decisions": ["Use FastAPI"],
            "key_events": ["Fixed auth bug"],
        })

        # Create second narrative entry
        storage.insert_narrative_entry({
            "directory_context": "/proj",
            "summary": "In /proj, during the last 24 hours: 3 memories recorded.",
            "period_start": _recent_timestamp(hours_ago=24),
            "period_end": _recent_timestamp(),
            "key_decisions": ["Add caching"],
            "key_events": ["Deployed v2"],
        })

        story = narrative.get_project_story("/proj")
        assert "5 memories recorded" in story
        assert "3 memories recorded" in story

    def test_empty_project_story(self, narrative):
        """Empty project returns informative message."""
        story = narrative.get_project_story("/nonexistent")
        assert "No narrative entries found" in story

    def test_narrative_chronological(self, narrative, storage):
        """Story entries are in chronological order."""
        # Insert entries in reverse order
        storage.insert_narrative_entry({
            "directory_context": "/proj",
            "summary": "Second period: implemented caching",
            "period_start": _recent_timestamp(hours_ago=24),
            "period_end": _recent_timestamp(),
            "key_decisions": [],
            "key_events": [],
        })
        storage.insert_narrative_entry({
            "directory_context": "/proj",
            "summary": "First period: set up project",
            "period_start": _recent_timestamp(hours_ago=48),
            "period_end": _recent_timestamp(hours_ago=24),
            "key_decisions": [],
            "key_events": [],
        })

        story = narrative.get_project_story("/proj")
        lines = story.split("\n\n")
        assert len(lines) == 2
        # First entry should be the earlier one
        assert "First period" in lines[0]
        assert "Second period" in lines[1]


class TestAutoNarrate:
    def test_auto_narrate(self, narrative, storage, settings):
        """Narratives generated during sleep for active directories."""
        # Create active memories in two directories
        for directory in ["/proj-a", "/proj-b"]:
            storage.insert_memory({
                "content": f"Working on {directory}",
                "embedding": _make_embedding(),
                "directory_context": directory,
                "heat": 0.8,
                "created_at": _recent_timestamp(hours_ago=1),
                "last_accessed": _recent_timestamp(),
            })

        stats = narrative.auto_narrate()

        assert stats["directories_checked"] >= 2
        assert stats["narratives_generated"] >= 2

        # Verify narratives were created
        story_a = narrative.get_project_story("/proj-a")
        assert "No narrative entries found" not in story_a

    def test_auto_narrate_skips_recent(self, narrative, storage, settings):
        """Auto-narrate skips directories with recent narrative entries."""
        storage.insert_memory({
            "content": "Active memory",
            "embedding": _make_embedding(),
            "directory_context": "/proj",
            "heat": 0.8,
            "created_at": _recent_timestamp(),
            "last_accessed": _recent_timestamp(),
        })

        # Create a recent narrative entry
        storage.insert_narrative_entry({
            "directory_context": "/proj",
            "summary": "Recent narrative",
            "period_start": _recent_timestamp(hours_ago=12),
            "period_end": _recent_timestamp(),
            "key_decisions": [],
            "key_events": [],
        })

        stats = narrative.auto_narrate()
        assert stats["narratives_generated"] == 0

    def test_auto_narrate_cold_directories_skipped(self, narrative, storage):
        """Directories with only cold memories (heat < 0.3) are skipped."""
        storage.insert_memory({
            "content": "Old cold memory",
            "embedding": _make_embedding(),
            "directory_context": "/old-proj",
            "heat": 0.1,
            "created_at": _recent_timestamp(hours_ago=100),
            "last_accessed": _recent_timestamp(hours_ago=100),
        })

        stats = narrative.auto_narrate()
        assert stats["directories_checked"] == 0
        assert stats["narratives_generated"] == 0
