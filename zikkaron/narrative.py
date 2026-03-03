"""Autobiographical narrative engine — generates project stories from memory history."""

import json
import logging
import re
from collections import Counter
from datetime import datetime, timedelta, timezone

from zikkaron.config import Settings
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.storage import StorageEngine

logger = logging.getLogger(__name__)

# Keywords that signal a decision
_DECISION_KEYWORDS = frozenset({
    "decided", "chose", "choosing", "switched", "migrated", "replaced",
    "using", "adopted", "selected", "picked", "went with",
})

# Keywords that signal notable events
_EVENT_KEYWORDS = frozenset({
    "error", "fix", "fixed", "bug", "resolved", "broke", "crash",
    "deployed", "released", "implemented", "completed", "refactored",
})


class NarrativeEngine:
    """Generates autobiographical narrative entries from memory history."""

    def __init__(
        self,
        storage: StorageEngine,
        knowledge_graph: KnowledgeGraph,
        settings: Settings,
    ) -> None:
        self._storage = storage
        self._graph = knowledge_graph
        self._settings = settings

    def generate_narrative(
        self,
        directory: str,
        period_hours: int | None = None,
    ) -> dict:
        """Generate a narrative entry for a directory over a time period.

        Returns the inserted narrative entry dict.
        """
        if period_hours is None:
            period_hours = self._settings.NARRATIVE_INTERVAL_HOURS

        now = datetime.now(timezone.utc)
        period_start = now - timedelta(hours=period_hours)

        # Collect memories for this directory within the time window
        all_memories = self._storage.get_memories_for_directory(directory, min_heat=0.0)
        period_memories = [
            m for m in all_memories
            if datetime.fromisoformat(m["created_at"]) >= period_start
        ]

        count = len(period_memories)
        decisions = self._extract_decisions(period_memories)
        events = self._extract_events(period_memories)
        top_entities = self._get_top_entities(period_memories)
        high_heat_topics = self._get_high_heat_topics(directory)

        # Build summary
        period_desc = f"the last {period_hours} hours"
        parts = [f"In {directory}, during {period_desc}: {count} memories recorded."]

        if decisions:
            parts.append(f"Key decisions: {', '.join(decisions[:5])}.")
        if events:
            parts.append(f"Notable events: {', '.join(events[:5])}.")
        if top_entities:
            parts.append(f"Main entities involved: {', '.join(top_entities[:5])}.")
        if high_heat_topics:
            parts.append(f"Current focus areas: {', '.join(high_heat_topics[:5])}.")

        summary = " ".join(parts)

        entry = {
            "directory_context": directory,
            "summary": summary,
            "period_start": period_start.isoformat(),
            "period_end": now.isoformat(),
            "key_decisions": decisions[:10],
            "key_events": events[:10],
        }

        entry_id = self._storage.insert_narrative_entry(entry)
        entry["id"] = entry_id
        return entry

    def get_project_story(self, directory: str, max_entries: int = 10) -> str:
        """Retrieve all narrative entries for a directory and combine into a story."""
        entries = self._storage.get_narratives_for_directory(directory, limit=max_entries)
        if not entries:
            return f"No narrative entries found for {directory}."

        # Sort chronologically (ascending by period_start)
        entries.sort(key=lambda e: e["period_start"])

        parts = []
        for entry in entries:
            parts.append(entry["summary"])

        return "\n\n".join(parts)

    def auto_narrate(self) -> dict:
        """Auto-generate narratives for active directories during sleep-time compute.

        For each directory with memories having heat > 0.3:
          If no narrative entry in the last NARRATIVE_INTERVAL_HOURS, generate one.
        """
        stats = {"directories_checked": 0, "narratives_generated": 0}
        interval = self._settings.NARRATIVE_INTERVAL_HOURS
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=interval)

        # Find active directories
        active_dirs = self._get_active_directories(min_heat=0.3)
        stats["directories_checked"] = len(active_dirs)

        for directory in active_dirs:
            # Check if there's a recent narrative
            existing = self._storage.get_narratives_for_directory(directory, limit=1)
            if existing:
                latest_end = datetime.fromisoformat(existing[0]["period_end"])
                if latest_end >= cutoff:
                    continue

            try:
                self.generate_narrative(directory, period_hours=interval)
                stats["narratives_generated"] += 1
            except Exception:
                logger.exception("Failed to generate narrative for %s", directory)

        return stats

    def _extract_decisions(self, memories: list[dict]) -> list[str]:
        """Extract decision descriptions from memories."""
        decisions = []
        for mem in memories:
            content_lower = mem["content"].lower()
            if any(kw in content_lower for kw in _DECISION_KEYWORDS):
                # Use first sentence or first 100 chars as decision summary
                summary = mem["content"].split(".")[0][:100].strip()
                if summary:
                    decisions.append(summary)

            # Also check entity type = "decision"
            tags = mem.get("tags", [])
            if isinstance(tags, str):
                tags = json.loads(tags)
            if "decision" in tags:
                summary = mem["content"].split(".")[0][:100].strip()
                if summary and summary not in decisions:
                    decisions.append(summary)

        return decisions

    def _extract_events(self, memories: list[dict]) -> list[str]:
        """Extract notable events from memories."""
        events = []
        for mem in memories:
            content_lower = mem["content"].lower()
            importance = mem.get("importance", 0.5)

            # High-importance memories are notable events
            if importance > 0.7:
                summary = mem["content"].split(".")[0][:100].strip()
                if summary:
                    events.append(summary)
                continue

            # Check for event keywords
            if any(kw in content_lower for kw in _EVENT_KEYWORDS):
                summary = mem["content"].split(".")[0][:100].strip()
                if summary and summary not in events:
                    events.append(summary)

        return events

    def _get_top_entities(self, memories: list[dict]) -> list[str]:
        """Get most frequently mentioned entities from memory content."""
        entity_counts: Counter = Counter()
        for mem in memories:
            # Extract entities from content using simple patterns
            entities = self._graph.extract_entities_typed(mem["content"], "")
            for name, etype, _ in entities:
                entity_counts[name] += 1

        return [name for name, _ in entity_counts.most_common(10)]

    def _get_high_heat_topics(self, directory: str) -> list[str]:
        """Get topics from high-heat memories in this directory."""
        hot_memories = self._storage.get_memories_for_directory(
            directory, min_heat=0.7
        )
        topics = []
        for mem in hot_memories[:10]:
            # First meaningful sentence
            summary = mem["content"].split(".")[0][:60].strip()
            if summary:
                topics.append(summary)
        return topics

    def _get_active_directories(self, min_heat: float = 0.3) -> list[str]:
        """Get directories that have memories above the heat threshold."""
        rows = self._storage._conn.execute(
            "SELECT DISTINCT directory_context FROM memories WHERE heat >= ?",
            (min_heat,),
        ).fetchall()
        return [row[0] for row in rows]
