import pytest

from zikkaron.config import Settings
from zikkaron.sensory_buffer import SensoryBuffer
from zikkaron.storage import StorageEngine


@pytest.fixture
def storage(tmp_path):
    db_path = str(tmp_path / "test_buffer.db")
    engine = StorageEngine(db_path)
    yield engine
    engine.close()


@pytest.fixture
def settings():
    return Settings(
        MAX_EPISODE_TOKENS=100,   # 400 chars max per episode
        OVERLAP_TOKENS=25,        # 100 chars overlap
        DB_PATH=":memory:",
    )


@pytest.fixture
def buffer(storage, settings):
    return SensoryBuffer(storage, settings)


class TestStartSession:
    def test_session_id_is_set(self, buffer):
        sid = buffer.start_session()
        assert sid is not None
        assert len(sid) == 32  # uuid4 hex
        assert buffer.session_id == sid

    def test_episode_initialized(self, buffer):
        buffer.start_session()
        ep = buffer.get_current_episode()
        assert ep is not None
        assert ep["raw_content"] == ""
        assert ep["session_id"] == buffer.session_id
        assert ep["overlap_start"] is None
        assert ep["overlap_end"] is None


class TestCaptureContent:
    def test_capture_appends(self, buffer):
        buffer.start_session()
        buffer.capture("hello ", "/tmp")
        buffer.capture("world", "/tmp")
        ep = buffer.get_current_episode()
        assert ep["raw_content"] == "hello world"
        assert ep["directory"] == "/tmp"

    def test_capture_auto_starts_session(self, buffer):
        assert buffer.session_id is None
        buffer.capture("auto start", "/tmp")
        assert buffer.session_id is not None
        assert buffer.get_current_episode()["raw_content"] == "auto start"


class TestEpisodeRotation:
    def test_rotation_saves_old_and_starts_new(self, buffer, storage):
        buffer.start_session()
        sid = buffer.session_id
        # 400 chars max, so 450 chars should trigger rotation
        content = "x" * 450
        buffer.capture(content, "/proj")

        saved = storage.get_session_episodes(sid)
        assert len(saved) == 1
        assert saved[0]["raw_content"] == content
        assert saved[0]["directory"] == "/proj"

        # New episode should exist with overlap content
        ep = buffer.get_current_episode()
        assert ep is not None
        assert len(ep["raw_content"]) == 100  # overlap_chars

    def test_rotation_preserves_session_id(self, buffer):
        buffer.start_session()
        sid = buffer.session_id
        buffer.capture("y" * 450, "/proj")
        assert buffer.session_id == sid
        assert buffer.get_current_episode()["session_id"] == sid


class TestOverlapContinuity:
    def test_overlap_matches_end_of_old_episode(self, buffer, storage):
        buffer.start_session()
        sid = buffer.session_id
        content = "A" * 300 + "B" * 150
        buffer.capture(content, "/proj")

        saved = storage.get_session_episodes(sid)
        old_content = saved[0]["raw_content"]
        new_ep = buffer.get_current_episode()

        # The overlap (last 100 chars of old) should be the start of new
        assert old_content[-100:] == new_ep["raw_content"][:100]

    def test_overlap_positions_recorded(self, buffer):
        buffer.start_session()
        content = "Z" * 450
        buffer.capture(content, "/proj")
        ep = buffer.get_current_episode()
        assert ep["overlap_start"] == 450 - 100
        assert ep["overlap_end"] == 450


class TestFlush:
    def test_flush_saves_to_db(self, buffer, storage):
        buffer.start_session()
        sid = buffer.session_id
        buffer.capture("partial content", "/proj")
        ep_id = buffer.flush()

        assert ep_id is not None
        saved = storage.get_session_episodes(sid)
        assert len(saved) == 1
        assert saved[0]["raw_content"] == "partial content"

    def test_flush_resets_episode(self, buffer):
        buffer.start_session()
        buffer.capture("data", "/proj")
        buffer.flush()
        ep = buffer.get_current_episode()
        assert ep["raw_content"] == ""

    def test_flush_empty_returns_none(self, buffer):
        buffer.start_session()
        assert buffer.flush() is None

    def test_flush_no_session_returns_none(self, buffer):
        assert buffer.flush() is None


class TestMultipleCaptures:
    def test_content_accumulates(self, buffer):
        buffer.start_session()
        for i in range(10):
            buffer.capture(f"line {i}\n", "/proj")
        ep = buffer.get_current_episode()
        for i in range(10):
            assert f"line {i}" in ep["raw_content"]

    def test_multiple_rotations(self, buffer, storage):
        buffer.start_session()
        sid = buffer.session_id
        # Each capture is 200 chars, max is 400, so every ~2 captures rotates
        for _ in range(6):
            buffer.capture("a" * 200, "/proj")

        saved = storage.get_session_episodes(sid)
        # Should have saved multiple episodes
        assert len(saved) >= 2

        # Current episode should still be active
        assert buffer.get_current_episode() is not None
        assert buffer.get_current_episode()["raw_content"] != ""
