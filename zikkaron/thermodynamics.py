"""Neuroscience-inspired memory thermodynamics — surprise, importance, valence, and enhanced decay."""

import logging
import re
from datetime import datetime, timezone

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.storage import StorageEngine

logger = logging.getLogger(__name__)

# -- Keyword sets for heuristic scoring --

_ERROR_KEYWORDS = re.compile(
    r"\b(error|exception|traceback|failed|failure|bug|crash|broken|timeout|"
    r"denied|rejected|deprecated)\b",
    re.IGNORECASE,
)

_SUCCESS_KEYWORDS = re.compile(
    r"\b(fixed|resolved|working|success|passed|deployed|completed|shipped|merged|approved)\b",
    re.IGNORECASE,
)

_DECISION_KEYWORDS = re.compile(
    r"\b(decided|chose|switched|migrated|selected|picked|opted)\b",
    re.IGNORECASE,
)

_ARCHITECTURE_KEYWORDS = re.compile(
    r"\b(design|pattern|refactor|architecture|restructur|modular|decouple|abstract)\b",
    re.IGNORECASE,
)

_CODE_BLOCK_RE = re.compile(r"```|`[^`]+`")
_FILE_PATH_RE = re.compile(r"(?:\.{0,2}/)?(?:[\w@.-]+/)+[\w@.-]+\.\w+")


class MemoryThermodynamics:
    """Advanced heat calculations inspired by neuroscience principles.

    Computes surprise, importance, emotional valence, synaptic boosts,
    enhanced decay, and metamemory tracking for the Zikkaron memory system.
    """

    def __init__(
        self,
        storage: StorageEngine,
        embeddings: EmbeddingEngine,
        settings: Settings,
    ) -> None:
        self._storage = storage
        self._embeddings = embeddings
        self._settings = settings

    # -- a. Surprise Scoring --

    def compute_surprise(self, content: str, directory: str) -> float:
        """Compute how novel this content is relative to existing memories.

        Encodes the content, finds the top 5 most similar existing memories,
        and returns surprise = 1.0 - max_similarity.
        Returns 0.5 (moderate) if no existing memories are found.
        """
        query_embedding = self._embeddings.encode(content)
        if query_embedding is None:
            return 0.5

        vec_hits = self._storage.search_vectors(
            query_embedding, top_k=5, min_heat=0.0
        )
        if not vec_hits:
            return 0.5

        max_similarity = 0.0
        for mid, _distance in vec_hits:
            mem = self._storage.get_memory(mid)
            if mem and mem.get("embedding"):
                sim = self._embeddings.similarity(query_embedding, mem["embedding"])
                max_similarity = max(max_similarity, sim)

        surprise = 1.0 - max_similarity
        return max(0.0, min(1.0, surprise))

    def apply_surprise_boost(self, base_heat: float, surprise: float) -> float:
        """Apply surprise boost to initial heat. Capped at 1.0."""
        boosted = base_heat + (surprise * self._settings.SURPRISE_BOOST)
        return min(boosted, 1.0)

    # -- b. Importance Scoring --

    def compute_importance(self, content: str, tags: list[str]) -> float:
        """Heuristic importance scoring based on content signals. No LLM needed.

        Scores:
          - error/exception/traceback keywords: +0.2
          - decision keywords: +0.3
          - architecture keywords: +0.2
          - 3+ tags: +0.1
          - content > 500 chars: +0.1
          - code blocks or file paths: +0.1
        Normalized to 0.0–1.0.
        """
        score = 0.0

        if _ERROR_KEYWORDS.search(content):
            score += 0.2
        if _DECISION_KEYWORDS.search(content):
            score += 0.3
        if _ARCHITECTURE_KEYWORDS.search(content):
            score += 0.2
        if len(tags) >= 3:
            score += 0.1
        if len(content) > 500:
            score += 0.1
        if _CODE_BLOCK_RE.search(content) or _FILE_PATH_RE.search(content):
            score += 0.1

        return min(score, 1.0)

    # -- c. Emotional Valence --

    def compute_valence(self, content: str) -> float:
        """Compute emotional valence from content keywords.

        Frustration signals push toward -1.0, satisfaction signals push toward +1.0.
        Returns a value in [-1.0, +1.0].
        """
        frustration_count = len(_ERROR_KEYWORDS.findall(content))
        satisfaction_count = len(_SUCCESS_KEYWORDS.findall(content))

        total = frustration_count + satisfaction_count
        if total == 0:
            return 0.0

        # Net valence: positive for satisfaction, negative for frustration
        raw = (satisfaction_count - frustration_count) / total
        return max(-1.0, min(1.0, raw))

    # -- d. Synaptic Tagging and Capture --

    def synaptic_boost(self, memory_id: int, event_heat: float) -> int:
        """Boost nearby memories when a high-importance event occurs.

        Finds all memories created within SYNAPTIC_WINDOW_MINUTES of the
        given memory and boosts their heat by SYNAPTIC_BOOST * event_heat.

        Returns the number of memories boosted.
        """
        mem = self._storage.get_memory(memory_id)
        if mem is None:
            return 0

        nearby = self._storage.get_memories_in_time_window(
            mem["created_at"], self._settings.SYNAPTIC_WINDOW_MINUTES
        )

        boost = self._settings.SYNAPTIC_BOOST * event_heat
        boosted_count = 0
        for m in nearby:
            if m["id"] == memory_id:
                continue
            new_heat = min(m["heat"] + boost, 1.0)
            if new_heat != m["heat"]:
                self._storage.update_memory_heat(m["id"], new_heat)
                boosted_count += 1

        return boosted_count

    # -- e. Enhanced Decay Formula --

    def compute_decay(self, memory: dict, hours_elapsed: float) -> float:
        """Compute decayed heat using importance, valence, and confidence modifiers.

        Base: heat * (DECAY_FACTOR ^ hours_elapsed)
        - High importance (>0.7): use IMPORTANCE_DECAY_FACTOR (slower decay)
        - High |valence|: multiply effective factor by (1 + |valence| * EMOTIONAL_DECAY_RESISTANCE)
        - High confidence: multiply effective factor by (1 + confidence * 0.1)
        """
        importance = memory.get("importance", 0.5)
        valence = memory.get("emotional_valence", 0.0)
        confidence = memory.get("confidence", 1.0)

        if importance > 0.7:
            decay_factor = self._settings.IMPORTANCE_DECAY_FACTOR
        else:
            decay_factor = self._settings.DECAY_FACTOR

        # Emotional resistance makes the factor closer to 1.0 (slower decay)
        emotional_modifier = 1.0 + abs(valence) * self._settings.EMOTIONAL_DECAY_RESISTANCE
        effective_factor = 1.0 - (1.0 - decay_factor) * (1.0 / emotional_modifier)

        # Confidence modifier
        confidence_modifier = 1.0 + confidence * 0.1
        effective_factor = 1.0 - (1.0 - effective_factor) * (1.0 / confidence_modifier)

        # Clamp factor to valid range
        effective_factor = max(0.0, min(effective_factor, 1.0))

        new_heat = memory["heat"] * (effective_factor ** hours_elapsed)
        return new_heat

    # -- f. Metamemory --

    def record_access(self, memory_id: int, was_useful: bool) -> None:
        """Track memory access and usefulness for metamemory."""
        mem = self._storage.get_memory(memory_id)
        if mem is None:
            return

        access_count = mem.get("access_count", 0) + 1
        useful_count = mem.get("useful_count", 0) + (1 if was_useful else 0)

        # Update confidence after enough data points
        if access_count > 3:
            confidence = useful_count / access_count
        else:
            confidence = mem.get("confidence", 1.0)

        self._storage.update_memory_metamemory(
            memory_id, access_count, useful_count, confidence
        )

    def get_reliability(self, memory_id: int) -> float:
        """Return the confidence score for a memory.

        If access_count < 3, returns 1.0 (benefit of the doubt).
        """
        mem = self._storage.get_memory(memory_id)
        if mem is None:
            return 1.0

        if mem.get("access_count", 0) < 3:
            return 1.0

        return mem.get("confidence", 1.0)

    # -- g. Session Coherence --

    def apply_session_coherence(self, heat: float, created_at: str) -> float:
        """Boost heat for memories created within the current session window.

        Memories created recently (within SESSION_COHERENCE_WINDOW_HOURS) get
        a heat bonus that decreases linearly as they age. This prevents the
        'I just told you this 10 minutes ago' problem by keeping active session
        context elevated.

        Returns boosted heat (capped at 1.0).
        """
        try:
            mem_dt = datetime.fromisoformat(created_at)
            if mem_dt.tzinfo is None:
                mem_dt = mem_dt.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            hours = (now - mem_dt).total_seconds() / 3600.0
            window = self._settings.SESSION_COHERENCE_WINDOW_HOURS

            if hours < window:
                freshness = 1.0 - (hours / window)
                bonus = self._settings.SESSION_COHERENCE_BONUS * freshness
                return min(heat + bonus, 1.0)
        except (ValueError, TypeError):
            pass

        return heat
