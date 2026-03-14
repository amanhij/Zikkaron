import uuid
from collections import deque
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
        # Action stream: lightweight log of all tool invocations for pattern extraction
        self._action_stream: deque[dict] = deque(maxlen=200)

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

    def capture_action(self, tool: str, directory: str, summary: str, result_type: str) -> None:
        """Record a tool invocation in the action stream.

        Action stream entries are lightweight structured records that capture
        what happened during the session. They feed into the sensory buffer
        as formatted text and can be used by consolidation to extract patterns
        like 'user tends to recall X before editing Y'.
        """
        if not self._settings.ACTION_STREAM_ENABLED:
            return

        action = {
            "tool": tool,
            "directory": directory,
            "summary": summary[:200],
            "result_type": result_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._action_stream.append(action)

        # Also inject into the sensory buffer as structured text
        action_text = f"[ACTION:{tool}] {result_type}: {summary[:150]}"
        self.capture(action_text, directory)

    def get_recent_actions(self, n: int = 20) -> list[dict]:
        """Return the last N action stream entries."""
        return list(self._action_stream)[-n:]

    def get_action_summary(self) -> str:
        """Generate a summary of recent actions for checkpoint context."""
        if not self._action_stream:
            return ""

        recent = list(self._action_stream)[-10:]
        lines = []
        for a in recent:
            lines.append(f"- {a['tool']}: {a['summary'][:80]}")
        return "Recent actions:\n" + "\n".join(lines)

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
