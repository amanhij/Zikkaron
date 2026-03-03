"""Tests for the prospective memory system."""

from datetime import datetime, timezone

import pytest

from zikkaron.config import Settings
from zikkaron.prospective import ProspectiveMemoryEngine
from zikkaron.storage import StorageEngine


@pytest.fixture
def storage(tmp_path):
    engine = StorageEngine(str(tmp_path / "test_prospective.db"))
    yield engine
    engine.close()


@pytest.fixture
def settings():
    return Settings(DB_PATH=":memory:")


@pytest.fixture
def prospective(storage, settings):
    return ProspectiveMemoryEngine(storage, settings)


class TestCreateTrigger:
    def test_create_trigger(self, prospective, storage):
        """Trigger created successfully with valid params."""
        pm_id = prospective.create_trigger(
            content="Update tests after refactoring",
            trigger_condition="refactor",
            trigger_type="keyword_match",
            target_directory="/home/proj",
        )
        assert pm_id > 0

        active = storage.get_active_prospective_memories()
        assert len(active) == 1
        assert active[0]["content"] == "Update tests after refactoring"
        assert active[0]["trigger_type"] == "keyword_match"

    def test_invalid_trigger_type_raises(self, prospective):
        """Invalid trigger_type should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid trigger_type"):
            prospective.create_trigger(
                content="test",
                trigger_condition="cond",
                trigger_type="invalid_type",
            )


class TestDirectoryTrigger:
    def test_directory_trigger(self, prospective):
        """Trigger fires when context directory contains target_directory."""
        prospective.create_trigger(
            content="Check database migrations",
            trigger_condition="/home/proj/db",
            trigger_type="directory_match",
            target_directory="/home/proj/db",
        )

        context = {
            "directory": "/home/proj/db/migrations",
            "content": "something unrelated",
            "entities": [],
            "current_time": datetime.now(timezone.utc),
        }
        triggered = prospective.check_triggers(context)
        assert len(triggered) == 1
        assert triggered[0]["content"] == "Check database migrations"

    def test_directory_no_match(self, prospective):
        """Trigger does not fire for non-matching directory."""
        prospective.create_trigger(
            content="Check database migrations",
            trigger_condition="/home/proj/db",
            trigger_type="directory_match",
            target_directory="/home/proj/db",
        )

        context = {
            "directory": "/home/other/frontend",
            "content": "",
            "entities": [],
            "current_time": datetime.now(timezone.utc),
        }
        triggered = prospective.check_triggers(context)
        assert len(triggered) == 0


class TestKeywordTrigger:
    def test_keyword_trigger(self, prospective):
        """Trigger fires on matching keywords in content."""
        prospective.create_trigger(
            content="Remember to update the API docs",
            trigger_condition="API docs",
            trigger_type="keyword_match",
        )

        context = {
            "directory": "/proj",
            "content": "Working on the API endpoint and updating docs",
            "entities": [],
            "current_time": datetime.now(timezone.utc),
        }
        triggered = prospective.check_triggers(context)
        assert len(triggered) == 1

    def test_keyword_no_match(self, prospective):
        """Trigger does not fire when keywords are absent."""
        prospective.create_trigger(
            content="Remember to update the API docs",
            trigger_condition="API docs",
            trigger_type="keyword_match",
        )

        context = {
            "directory": "/proj",
            "content": "Working on frontend styling",
            "entities": [],
            "current_time": datetime.now(timezone.utc),
        }
        triggered = prospective.check_triggers(context)
        assert len(triggered) == 0


class TestEntityTrigger:
    def test_entity_trigger(self, prospective):
        """Trigger fires when matching entity appears in context."""
        prospective.create_trigger(
            content="Review StorageEngine changes",
            trigger_condition="StorageEngine",
            trigger_type="entity_match",
        )

        context = {
            "directory": "/proj",
            "content": "",
            "entities": ["StorageEngine", "EmbeddingEngine"],
            "current_time": datetime.now(timezone.utc),
        }
        triggered = prospective.check_triggers(context)
        assert len(triggered) == 1


class TestTimeTrigger:
    def test_time_trigger_hour_match(self, prospective):
        """Time-based trigger fires on matching hour:minute."""
        prospective.create_trigger(
            content="Run daily backup",
            trigger_condition="14:30",
            trigger_type="time_based",
        )

        context = {
            "directory": "/proj",
            "content": "",
            "entities": [],
            "current_time": datetime(2026, 3, 2, 14, 30, tzinfo=timezone.utc),
        }
        triggered = prospective.check_triggers(context)
        assert len(triggered) == 1

    def test_time_trigger_no_match(self, prospective):
        """Time-based trigger does not fire at wrong time."""
        prospective.create_trigger(
            content="Run daily backup",
            trigger_condition="14:30",
            trigger_type="time_based",
        )

        context = {
            "directory": "/proj",
            "content": "",
            "entities": [],
            "current_time": datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc),
        }
        triggered = prospective.check_triggers(context)
        assert len(triggered) == 0


class TestTriggerDeactivation:
    def test_trigger_deactivates_after_5_fires(self, prospective, storage):
        """Trigger deactivates after triggered_count > 5."""
        pm_id = prospective.create_trigger(
            content="Persistent reminder",
            trigger_condition="test",
            trigger_type="keyword_match",
        )

        context = {
            "directory": "/proj",
            "content": "this is a test of the system",
            "entities": [],
            "current_time": datetime.now(timezone.utc),
        }

        # Fire 6 times
        for i in range(6):
            prospective.check_triggers(context)

        # 7th time: trigger should be deactivated
        triggered = prospective.check_triggers(context)
        assert len(triggered) == 0

        active = storage.get_active_prospective_memories()
        assert all(pm["id"] != pm_id for pm in active)


class TestAutoCreateFromContent:
    def test_auto_create_from_todo(self, prospective, storage):
        """Content with TODO generates prospective memory."""
        content = "TODO: add validation to the input parser"
        ids = prospective.auto_create_from_content(content, "/home/proj")

        assert len(ids) >= 1
        active = storage.get_active_prospective_memories()
        assert len(active) >= 1
        assert any("validation" in pm["content"].lower() for pm in active)

    def test_auto_create_from_fixme(self, prospective, storage):
        """Content with FIXME generates prospective memory."""
        content = "FIXME: handle edge case for empty arrays"
        ids = prospective.auto_create_from_content(content, "/proj")
        assert len(ids) >= 1

    def test_auto_create_from_remember_to(self, prospective, storage):
        """Content with 'remember to' generates prospective memory."""
        content = "remember to update the changelog before release"
        ids = prospective.auto_create_from_content(content, "/proj")
        assert len(ids) >= 1

    def test_no_auto_create_from_normal_content(self, prospective, storage):
        """Normal content without future phrases generates no triggers."""
        content = "The server is running correctly with all tests passing."
        ids = prospective.auto_create_from_content(content, "/proj")
        assert len(ids) == 0


class TestNoFalseTriggers:
    def test_unrelated_context_no_trigger(self, prospective):
        """Unrelated context does not fire any triggers."""
        prospective.create_trigger(
            content="Check API when working on backend",
            trigger_condition="backend API",
            trigger_type="keyword_match",
            target_directory="/backend",
        )
        prospective.create_trigger(
            content="Review migrations",
            trigger_condition="/database",
            trigger_type="directory_match",
            target_directory="/database",
        )
        prospective.create_trigger(
            content="Check Redis config",
            trigger_condition="Redis",
            trigger_type="entity_match",
        )

        context = {
            "directory": "/frontend/components",
            "content": "Updated the CSS styles for the button component",
            "entities": ["ButtonComponent", "StyleSheet"],
            "current_time": datetime.now(timezone.utc),
        }
        triggered = prospective.check_triggers(context)
        assert len(triggered) == 0
