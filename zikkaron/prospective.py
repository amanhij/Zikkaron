"""Prospective memory system — future-oriented triggers that fire on matching context."""

import re
from datetime import datetime, timezone

from zikkaron.config import Settings
from zikkaron.storage import StorageEngine

VALID_TRIGGER_TYPES = frozenset({
    "directory_match", "keyword_match", "entity_match", "time_based",
})

# Future-oriented phrases that signal prospective intent
_PROSPECTIVE_PATTERNS = [
    (re.compile(r"\bTODO\b[:\s]*(.+?)(?:\n|$)", re.IGNORECASE), "keyword_match"),
    (re.compile(r"\bFIXME\b[:\s]*(.+?)(?:\n|$)", re.IGNORECASE), "keyword_match"),
    (re.compile(r"remember to\s+(.+?)(?:\.|$)", re.IGNORECASE), "keyword_match"),
    (re.compile(r"don'?t forget\s+(.+?)(?:\.|$)", re.IGNORECASE), "keyword_match"),
    (re.compile(r"next time\s+(.+?)(?:\.|$)", re.IGNORECASE), "keyword_match"),
    (re.compile(r"when we\s+(.+?)(?:\.|$)", re.IGNORECASE), "keyword_match"),
    (re.compile(r"later\s+(.+?)(?:\.|$)", re.IGNORECASE), "keyword_match"),
    (re.compile(r"eventually\s+(.+?)(?:\.|$)", re.IGNORECASE), "keyword_match"),
    (re.compile(r"should also\s+(.+?)(?:\.|$)", re.IGNORECASE), "keyword_match"),
]

# Simple cron-like time matching for time_based triggers
# Format: "HH:MM" or "weekday:N" (0=Monday..6=Sunday)
_TIME_HOUR_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_TIME_WEEKDAY_RE = re.compile(r"^weekday:(\d)$")

MAX_TRIGGER_COUNT = 5


class ProspectiveMemoryEngine:
    """Future-oriented memory triggers that fire when matching context appears."""

    def __init__(self, storage: StorageEngine, settings: Settings) -> None:
        self._storage = storage
        self._settings = settings

    def create_trigger(
        self,
        content: str,
        trigger_condition: str,
        trigger_type: str,
        target_directory: str | None = None,
    ) -> int:
        """Create a prospective memory trigger.

        Returns the prospective memory ID.
        """
        if trigger_type not in VALID_TRIGGER_TYPES:
            raise ValueError(
                f"Invalid trigger_type: {trigger_type}. "
                f"Must be one of {sorted(VALID_TRIGGER_TYPES)}"
            )

        return self._storage.insert_prospective_memory({
            "content": content,
            "trigger_condition": trigger_condition,
            "trigger_type": trigger_type,
            "target_directory": target_directory,
            "is_active": True,
        })

    def check_triggers(self, context: dict) -> list[dict]:
        """Check all active triggers against the given context.

        context keys:
          - directory (str)
          - content (str)
          - entities (list[str])
          - current_time (datetime)

        Returns list of triggered prospective memories.
        Deactivates triggers that have fired more than MAX_TRIGGER_COUNT times.
        """
        active = self._storage.get_active_prospective_memories()
        triggered = []

        directory = context.get("directory", "")
        content = context.get("content", "")
        entities = context.get("entities", [])
        current_time = context.get("current_time", datetime.now(timezone.utc))

        for pm in active:
            if self._matches(pm, directory, content, entities, current_time):
                self._storage.trigger_prospective_memory(pm["id"])

                # Re-read to get updated count
                new_count = pm["triggered_count"] + 1

                # Deactivate if over limit
                if new_count > MAX_TRIGGER_COUNT:
                    self._storage._conn.execute(
                        "UPDATE prospective_memories SET is_active = 0 WHERE id = ?",
                        (pm["id"],),
                    )
                    self._storage._conn.commit()

                pm["triggered_count"] = new_count
                triggered.append(pm)

        return triggered

    def auto_create_from_content(self, content: str, directory: str) -> list[int]:
        """Scan content for future-oriented phrases and auto-create triggers.

        Returns list of created prospective memory IDs.
        """
        created_ids = []

        for pattern, default_type in _PROSPECTIVE_PATTERNS:
            for match in pattern.finditer(content):
                actionable = match.group(1).strip()
                if not actionable or len(actionable) < 5:
                    continue

                # Extract keywords from the actionable phrase for trigger_condition
                keywords = " ".join(
                    w for w in actionable.split()
                    if len(w) > 2 and w.lower() not in {"the", "and", "for", "with"}
                )
                if not keywords:
                    continue

                pm_id = self.create_trigger(
                    content=actionable,
                    trigger_condition=keywords,
                    trigger_type=default_type,
                    target_directory=directory,
                )
                created_ids.append(pm_id)

        return created_ids

    def _matches(
        self,
        pm: dict,
        directory: str,
        content: str,
        entities: list[str],
        current_time: datetime,
    ) -> bool:
        """Check if a single prospective memory matches the context."""
        trigger_type = pm["trigger_type"]
        condition = pm["trigger_condition"]

        if trigger_type == "directory_match":
            target_dir = pm.get("target_directory") or condition
            return target_dir != "" and target_dir in directory

        elif trigger_type == "keyword_match":
            keywords = condition.lower().split()
            content_lower = content.lower()
            return any(kw in content_lower for kw in keywords)

        elif trigger_type == "entity_match":
            entity_name = condition.lower()
            return any(entity_name == e.lower() for e in entities)

        elif trigger_type == "time_based":
            return self._matches_time(condition, current_time)

        return False

    @staticmethod
    def _matches_time(condition: str, current_time: datetime) -> bool:
        """Check if current_time matches a cron-like time condition."""
        m = _TIME_HOUR_RE.match(condition)
        if m:
            hour, minute = int(m.group(1)), int(m.group(2))
            return current_time.hour == hour and current_time.minute == minute

        m = _TIME_WEEKDAY_RE.match(condition)
        if m:
            weekday = int(m.group(1))
            return current_time.weekday() == weekday

        return False
