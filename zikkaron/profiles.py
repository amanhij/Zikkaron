"""Memobase-style structured user profiles + Hindsight derived beliefs."""
import json
import logging
import re
from datetime import datetime, timezone

from zikkaron.config import Settings
from zikkaron.storage import StorageEngine

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r'\b([A-Z][a-z]+)\b')

# Pattern: {Name} likes/loves/enjoys/prefers {X}
_INTEREST_RE = re.compile(
    r'\b([A-Z][a-z]+)\s+(?:likes?|loves?|enjoys?|prefers?)\s+(.+?)(?:\.|,|$)',
    re.IGNORECASE,
)

# Pattern: {Name} is {adjective}
_TRAIT_RE = re.compile(
    r'\b([A-Z][a-z]+)\s+is\s+([a-z]+)\b',
)

# Pattern: {Name} went to/visited/traveled to {place}
_TRAVEL_RE = re.compile(
    r'\b([A-Z][a-z]+)\s+(?:went\s+to|visited|traveled\s+to|travelled\s+to)\s+(.+?)(?:\.|,|$)',
    re.IGNORECASE,
)

# Pattern: {Name} works as/is a {job}
_CAREER_RE = re.compile(
    r'\b([A-Z][a-z]+)\s+(?:works\s+as\s+(?:a\s+|an\s+)?|is\s+a(?:n)?\s+)(.+?)(?:\.|,|$)',
    re.IGNORECASE,
)

# Pattern: {Name} believes/thinks {X}
_OPINION_RE = re.compile(
    r'\b([A-Z][a-z]+)\s+(?:believes?|thinks?)\s+(.+?)(?:\.|,|$)',
    re.IGNORECASE,
)

# Pattern: {Name} wants to/hopes to {X}
_GOAL_RE = re.compile(
    r'\b([A-Z][a-z]+)\s+(?:wants?\s+to|hopes?\s+to)\s+(.+?)(?:\.|,|$)',
    re.IGNORECASE,
)

# Pattern: {Name} always/usually/often {X}
_HABIT_RE = re.compile(
    r'\b([A-Z][a-z]+)\s+(?:always|usually|often)\s+(.+?)(?:\.|,|$)',
    re.IGNORECASE,
)

# Common non-name words that match [A-Z][a-z]+ at sentence starts
_NON_NAMES = frozenset({
    "The", "This", "That", "These", "Those", "There", "They", "Then",
    "What", "When", "Where", "Which", "While", "Who", "Why", "How",
    "Here", "His", "Her", "Its", "Our", "Your", "Their", "Some",
    "Many", "Most", "Each", "Every", "Any", "All", "Both", "Few",
    "She", "But", "And", "For", "Not", "Yet", "Also", "Just",
    "Very", "Really", "Always", "Usually", "Often", "Never",
    "Because", "Since", "After", "Before", "During", "Until",
})


def _normalize(text: str) -> str:
    """Lowercase, replace spaces with underscores, truncate to 50 chars."""
    return text.strip().lower().replace(" ", "_")[:50]


def _is_name(word: str) -> bool:
    return word not in _NON_NAMES and _NAME_RE.fullmatch(word) is not None


class ProfileExtractor:
    """Rule-based extraction of structured user profile attributes from memory content."""

    def __init__(self, storage: StorageEngine, settings: Settings) -> None:
        self._storage = storage
        self._settings = settings

    def extract_and_store(self, content: str, memory_id: int, directory_context: str) -> None:
        attrs = self._extract_attributes(content)
        for entity_name, attr_type, attr_key, attr_value, confidence in attrs:
            self._storage.insert_profile(
                entity_name=entity_name,
                attribute_type=attr_type,
                attribute_key=attr_key,
                attribute_value=attr_value,
                memory_id=memory_id,
                confidence=confidence,
                directory_context=directory_context,
            )
        logger.debug(
            "Profile extraction: %d attributes from memory %d",
            len(attrs), memory_id,
        )

    def _extract_attributes(
        self, content: str,
    ) -> list[tuple[str, str, str, str, float]]:
        results: list[tuple[str, str, str, str, float]] = []

        for m in _INTEREST_RE.finditer(content):
            name, value = m.group(1), m.group(2).strip()
            if _is_name(name) and value:
                results.append((name, "interest", _normalize(value), value, 0.7))

        for m in _TRAIT_RE.finditer(content):
            name, adj = m.group(1), m.group(2).strip()
            if _is_name(name) and adj:
                results.append((name, "trait", adj.lower(), adj, 0.7))

        for m in _TRAVEL_RE.finditer(content):
            name, place = m.group(1), m.group(2).strip()
            if _is_name(name) and place:
                results.append((name, "interest", "travel", place, 0.6))

        for m in _CAREER_RE.finditer(content):
            name, job = m.group(1), m.group(2).strip()
            if _is_name(name) and job:
                results.append((name, "career", _normalize(job), job, 0.7))

        for m in _OPINION_RE.finditer(content):
            name, opinion = m.group(1), m.group(2).strip()
            if _is_name(name) and opinion:
                results.append((name, "opinion", _normalize(opinion), opinion, 0.6))

        for m in _GOAL_RE.finditer(content):
            name, goal = m.group(1), m.group(2).strip()
            if _is_name(name) and goal:
                results.append((name, "goal", _normalize(goal), goal, 0.6))

        for m in _HABIT_RE.finditer(content):
            name, habit = m.group(1), m.group(2).strip()
            if _is_name(name) and habit:
                results.append((name, "habit", _normalize(habit), habit, 0.6))

        return results

    def generate_profile_summary(
        self, entity_name: str, directory_context: str,
    ) -> str | None:
        profiles = self._storage.get_profiles_for_entity(entity_name, directory_context)
        if len(profiles) < 3:
            return None

        by_type: dict[str, list[str]] = {}
        for p in profiles:
            by_type.setdefault(p["attribute_type"], []).append(p["attribute_value"])

        parts: list[str] = []

        if "trait" in by_type:
            traits = ", ".join(by_type["trait"])
            parts.append(f"{entity_name} is {traits}")

        if "interest" in by_type:
            interests = ", ".join(by_type["interest"])
            parts.append(f"{entity_name} enjoys {interests}")

        if "career" in by_type:
            career = by_type["career"][0]
            parts.append(f"{entity_name} works as {career}")

        if "opinion" in by_type:
            opinions = ", ".join(by_type["opinion"])
            parts.append(f"{entity_name} believes {opinions}")

        if "goal" in by_type:
            goals = ", ".join(by_type["goal"])
            parts.append(f"{entity_name} wants to {goals}")

        if "habit" in by_type:
            habits = ", ".join(by_type["habit"])
            parts.append(f"{entity_name} {habits}")

        return ". ".join(parts) + "." if parts else None


ACTIVITY_CATEGORIES = {
    "camping": "outdoor", "hiking": "outdoor", "fishing": "outdoor",
    "kayaking": "outdoor", "climbing": "outdoor", "cycling": "outdoor",
    "painting": "creative", "drawing": "creative", "pottery": "creative",
    "sculpture": "creative", "photography": "creative", "writing": "creative",
    "piano": "music", "violin": "music", "guitar": "music",
    "singing": "music", "drums": "music",
    "running": "fitness", "swimming": "fitness", "yoga": "fitness",
    "pilates": "fitness", "weightlifting": "fitness",
    "cooking": "domestic", "baking": "domestic", "gardening": "domestic",
    "sewing": "domestic", "knitting": "domestic",
}

_CATEGORY_LABELS = {
    "outdoor": "outdoor activities",
    "creative": "creative activities",
    "music": "music",
    "fitness": "fitness",
    "domestic": "domestic activities",
}


class BeliefDeriver:
    """Derives higher-level beliefs from memory content and profile attributes."""

    def __init__(self, storage: StorageEngine, settings: Settings) -> None:
        self._storage = storage
        self._settings = settings

    def derive_from_memory(
        self, content: str, memory_id: int, directory_context: str,
    ) -> None:
        content_lower = content.lower()

        # Check for activity mentions that map to categories
        found_categories: dict[str, list[str]] = {}
        for activity, category in ACTIVITY_CATEGORIES.items():
            if activity in content_lower:
                found_categories.setdefault(category, []).append(activity)

        # Extract name from content
        names = [m.group(1) for m in _NAME_RE.finditer(content) if _is_name(m.group(1))]
        subject = names[0] if names else "unknown"

        for category, activities in found_categories.items():
            label = _CATEGORY_LABELS.get(category, category)
            belief_content = f"enjoys {label} (evidence: {', '.join(activities)})"
            self._storage.insert_belief(
                belief_type="preference",
                subject=subject,
                content=belief_content,
                evidence_memory_ids=[memory_id],
                confidence=0.6,
                directory_context=directory_context,
            )

        # Trait/preference patterns
        for m in _INTEREST_RE.finditer(content):
            name, value = m.group(1), m.group(2).strip()
            if _is_name(name) and value:
                self._storage.insert_belief(
                    belief_type="preference",
                    subject=name,
                    content=f"likes {value}",
                    evidence_memory_ids=[memory_id],
                    confidence=0.6,
                    directory_context=directory_context,
                )

        for m in _TRAIT_RE.finditer(content):
            name, adj = m.group(1), m.group(2).strip()
            if _is_name(name) and adj:
                self._storage.insert_belief(
                    belief_type="trait",
                    subject=name,
                    content=f"is {adj}",
                    evidence_memory_ids=[memory_id],
                    confidence=0.6,
                    directory_context=directory_context,
                )

    def derive_from_profiles(
        self, entity_name: str, directory_context: str,
    ) -> None:
        profiles = self._storage.get_profiles_for_entity(entity_name, directory_context)
        if not profiles:
            return

        by_type: dict[str, list[dict]] = {}
        for p in profiles:
            by_type.setdefault(p["attribute_type"], []).append(p)

        # Group interests by activity category
        if "interest" in by_type:
            category_interests: dict[str, list[str]] = {}
            for p in by_type["interest"]:
                value_lower = p["attribute_value"].lower()
                category = ACTIVITY_CATEGORIES.get(value_lower)
                if category:
                    category_interests.setdefault(category, []).append(p["attribute_value"])

            for category, values in category_interests.items():
                if len(values) >= 3:
                    label = _CATEGORY_LABELS.get(category, category)
                    evidence_ids = []
                    for p in by_type["interest"]:
                        if p["attribute_value"] in values:
                            ids = json.loads(p["evidence_memory_ids"]) if isinstance(
                                p["evidence_memory_ids"], str,
                            ) else p["evidence_memory_ids"]
                            evidence_ids.extend(ids)
                    self._storage.insert_belief(
                        belief_type="preference",
                        subject=entity_name,
                        content=f"enjoys {label} (interests: {', '.join(values)})",
                        evidence_memory_ids=list(set(evidence_ids)),
                        confidence=0.7,
                        directory_context=directory_context,
                    )

        # Personality summary from traits
        if "trait" in by_type and len(by_type["trait"]) >= 2:
            trait_values = [p["attribute_value"] for p in by_type["trait"]]
            evidence_ids = []
            for p in by_type["trait"]:
                ids = json.loads(p["evidence_memory_ids"]) if isinstance(
                    p["evidence_memory_ids"], str,
                ) else p["evidence_memory_ids"]
                evidence_ids.extend(ids)
            self._storage.insert_belief(
                belief_type="summary",
                subject=entity_name,
                content=f"personality traits: {', '.join(trait_values)}",
                evidence_memory_ids=list(set(evidence_ids)),
                confidence=0.7,
                directory_context=directory_context,
            )
