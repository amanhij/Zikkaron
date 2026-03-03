import uuid
from datetime import datetime, timezone

from zikkaron.config import Settings
from zikkaron.storage import StorageEngine


class SensoryBuffer:
    def __init__(self, storage: StorageEngine, settings: Settings):
        self._storage = storage
        self._settings = settings
        self._max_chars = settings.MAX_EPISODE_TOKENS * 4  # token ≈ 4 chars
        self._overlap_chars = settings.OVERLAP_TOKENS * 4
        self.session_id: str | None = None
        self.current_episode: dict | None = None

    def start_session(self) -> str:
        self.session_id = uuid.uuid4().hex
        self.current_episode = {
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "directory": "",
            "raw_content": "",
            "overlap_start": None,
            "overlap_end": None,
        }
        return self.session_id

    def capture(self, content: str, directory: str) -> None:
        if self.current_episode is None:
            self.start_session()
        self.current_episode["directory"] = directory
        self.current_episode["raw_content"] += content
        if len(self.current_episode["raw_content"]) > self._max_chars:
            self._rotate_episode()

    def flush(self) -> int | None:
        if self.current_episode is None or not self.current_episode["raw_content"]:
            return None
        ep_id = self._storage.insert_episode(self.current_episode)
        self.current_episode = {
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "directory": self.current_episode["directory"],
            "raw_content": "",
            "overlap_start": None,
            "overlap_end": None,
        }
        return ep_id

    def get_current_episode(self) -> dict | None:
        return self.current_episode

    def get_session_episodes(self, session_id: str) -> list[dict]:
        return self._storage.get_session_episodes(session_id)

    def _rotate_episode(self) -> None:
        old_content = self.current_episode["raw_content"]
        old_directory = self.current_episode["directory"]

        # Save old episode
        self._storage.insert_episode(self.current_episode)

        # Extract overlap from the end of old content
        overlap = old_content[-self._overlap_chars:]
        overlap_start = len(old_content) - len(overlap)
        overlap_end = len(old_content)

        # Start new episode with overlap as seed
        self.current_episode = {
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "directory": old_directory,
            "raw_content": overlap,
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
        }
