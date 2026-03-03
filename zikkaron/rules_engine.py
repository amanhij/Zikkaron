"""Neuro-symbolic rules engine for hard constraints and soft preferences over retrieval."""

import fnmatch
import logging
from typing import Any

from zikkaron.config import Settings
from zikkaron.storage import StorageEngine

logger = logging.getLogger(__name__)

# Valid condition operators
VALID_OPERATORS = {
    "==", "!=", "contains", "not_contains",
    ">", "<", ">=", "<=", "matches",
}

# Fields that are numeric (use float comparison)
NUMERIC_FIELDS = {
    "heat", "importance", "surprise_score", "confidence",
    "emotional_valence", "plasticity", "stability", "excitability",
    "narrative_weight", "access_count", "useful_count",
    "compression_level", "reconsolidation_count",
}


def _parse_condition(condition: str) -> tuple[str, str, str]:
    """Parse a condition string into (field, operator, value).

    Supports conditions like:
        "language == typescript"
        "tag contains architecture"
        "importance > 0.7"
        "content not_contains password"
        "directory_context matches /project/*"
    """
    # Try multi-word operators first (not_contains)
    for op in ("not_contains",):
        if f" {op} " in condition:
            parts = condition.split(f" {op} ", 1)
            return parts[0].strip(), op, parts[1].strip()

    # Try two-char operators (>=, <=, ==, !=)
    for op in (">=", "<=", "==", "!="):
        if f" {op} " in condition:
            parts = condition.split(f" {op} ", 1)
            return parts[0].strip(), op, parts[1].strip()

    # Single-char operators (>, <)
    for op in (">", "<"):
        if f" {op} " in condition:
            parts = condition.split(f" {op} ", 1)
            return parts[0].strip(), op, parts[1].strip()

    # Word operators (contains, matches)
    for op in ("contains", "matches"):
        if f" {op} " in condition:
            parts = condition.split(f" {op} ", 1)
            return parts[0].strip(), op, parts[1].strip()

    raise ValueError(f"Cannot parse condition: {condition!r}")


def _parse_action(action: str) -> tuple[str, float]:
    """Parse an action string into (action_type, value).

    "filter" -> ("filter", 0.0)
    "boost:0.3" -> ("boost", 0.3)
    "penalty:0.2" -> ("penalty", 0.2)
    """
    if action == "filter":
        return "filter", 0.0
    if action.startswith("boost:"):
        return "boost", float(action.split(":", 1)[1])
    if action.startswith("penalty:"):
        return "penalty", float(action.split(":", 1)[1])
    raise ValueError(f"Invalid action: {action!r}")


def _get_field_value(memory: dict, field: str) -> Any:
    """Get a field value from a memory dict.

    Supports direct memory fields plus special 'tag' field which checks tags list.
    """
    if field == "tag" or field == "tags":
        return memory.get("tags", [])
    if field in memory:
        return memory[field]
    # Check if the field matches a tag name (e.g., "language" might be a tag prefix)
    tags = memory.get("tags", [])
    for tag in tags:
        if ":" in tag:
            key, val = tag.split(":", 1)
            if key.strip() == field:
                return val.strip()
        elif "=" in tag:
            key, val = tag.split("=", 1)
            if key.strip() == field:
                return val.strip()
    return None


class RulesEngine:
    """Neuro-symbolic rules engine combining hard logical constraints
    and soft preferences with neural retrieval results."""

    def __init__(self, storage: StorageEngine, settings: Settings) -> None:
        self._storage = storage
        self._settings = settings

    def add_rule(
        self,
        rule_type: str,
        scope: str,
        condition: str,
        action: str,
        priority: int = 0,
        scope_value: str | None = None,
    ) -> int:
        """Add a new rule.

        Args:
            rule_type: "hard" (must satisfy) or "soft" (preference)
            scope: "global", "directory", or "file"
            condition: Condition string (e.g., "importance > 0.7")
            action: Action string (e.g., "filter", "boost:0.3")
            priority: Higher = applied first (default 0)
            scope_value: Directory path or file pattern for scoped rules

        Returns:
            Rule ID
        """
        if rule_type not in ("hard", "soft"):
            raise ValueError(f"rule_type must be 'hard' or 'soft', got {rule_type!r}")
        if scope not in ("global", "directory", "file"):
            raise ValueError(f"scope must be 'global', 'directory', or 'file', got {scope!r}")

        # Validate condition parses correctly
        _parse_condition(condition)

        # Validate action parses correctly
        _parse_action(action)

        # Hard rules must use "filter" action
        if rule_type == "hard":
            action_type, _ = _parse_action(action)
            if action_type != "filter":
                raise ValueError("Hard rules must use 'filter' action")

        rule_id = self._storage.insert_rule({
            "rule_type": rule_type,
            "scope": scope,
            "scope_value": scope_value,
            "condition": condition,
            "action": action,
            "priority": priority,
            "is_active": True,
        })
        return rule_id

    def get_applicable_rules(self, directory: str) -> list[dict]:
        """Get all active rules that apply to the given directory.

        Returns rules where:
        - scope == "global", OR
        - scope == "directory" AND scope_value is a prefix of directory, OR
        - scope == "file" AND scope_value matches directory as a glob pattern

        Sorted by priority descending (highest first).
        """
        # Get global rules
        global_rules = self._storage.get_rules_for_scope("global")

        # Get directory-scoped rules: need to check prefix match
        # Query all active directory rules and filter by prefix
        rows = self._storage._conn.execute(
            "SELECT * FROM memory_rules WHERE scope = 'directory' AND is_active = 1 "
            "ORDER BY priority DESC"
        ).fetchall()
        dir_rules = []
        for row in rows:
            rule = self._storage._row_to_dict(row)
            sv = rule.get("scope_value") or ""
            if directory.startswith(sv):
                dir_rules.append(rule)

        # Get file-scoped rules: glob pattern matching
        rows = self._storage._conn.execute(
            "SELECT * FROM memory_rules WHERE scope = 'file' AND is_active = 1 "
            "ORDER BY priority DESC"
        ).fetchall()
        file_rules = []
        for row in rows:
            rule = self._storage._row_to_dict(row)
            sv = rule.get("scope_value") or ""
            if fnmatch.fnmatch(directory, sv):
                file_rules.append(rule)

        # Combine and sort by priority descending
        all_rules = global_rules + dir_rules + file_rules
        all_rules.sort(key=lambda r: r.get("priority", 0), reverse=True)
        return all_rules

    def apply_rules(self, memories: list[dict], directory: str) -> list[dict]:
        """Apply rules to filter and re-rank memories.

        Hard rules filter out non-matching memories.
        Soft rules adjust _retrieval_score via boost/penalty.

        Args:
            memories: List of memory dicts (must have _retrieval_score)
            directory: Current directory context

        Returns:
            Filtered and re-ranked list of memories
        """
        rules = self.get_applicable_rules(directory)
        if not rules:
            return memories

        result = list(memories)

        for rule in rules:
            rule_type = rule["rule_type"]
            condition = rule["condition"]
            action = rule["action"]

            if rule_type == "hard":
                # Filter: keep only memories that satisfy the condition
                result = [
                    m for m in result
                    if self.evaluate_condition(condition, m)
                ]
            elif rule_type == "soft":
                action_type, action_value = _parse_action(action)
                for m in result:
                    if self.evaluate_condition(condition, m):
                        score = m.get("_retrieval_score", 0.0)
                        if action_type == "boost":
                            m["_retrieval_score"] = score + action_value
                        elif action_type == "penalty":
                            m["_retrieval_score"] = score - action_value

        # Re-sort by score after soft rule adjustments
        result.sort(key=lambda m: m.get("_retrieval_score", 0.0), reverse=True)
        return result

    def evaluate_condition(self, condition: str, memory: dict) -> bool:
        """Evaluate a condition against a memory.

        Args:
            condition: Condition string like "importance > 0.7"
            memory: Memory dict

        Returns:
            True if the condition is satisfied
        """
        try:
            field, operator, value = _parse_condition(condition)
        except ValueError:
            logger.warning("Failed to parse condition: %s", condition)
            return True  # Unparseable conditions pass by default

        field_value = _get_field_value(memory, field)

        # Handle None field values
        if field_value is None:
            # For numeric comparisons, treat None as 0
            if operator in (">", "<", ">=", "<="):
                field_value = 0.0
            # For string comparisons, treat None as empty string
            elif operator in ("==", "!=", "contains", "not_contains", "matches"):
                field_value = ""

        # Numeric comparisons
        if operator in (">", "<", ">=", "<="):
            try:
                num_field = float(field_value) if not isinstance(field_value, (int, float)) else field_value
                num_value = float(value)
            except (ValueError, TypeError):
                return False

            if operator == ">":
                return num_field > num_value
            elif operator == "<":
                return num_field < num_value
            elif operator == ">=":
                return num_field >= num_value
            elif operator == "<=":
                return num_field <= num_value

        # Equality
        if operator == "==":
            # Try numeric equality first
            if field in NUMERIC_FIELDS:
                try:
                    return float(field_value) == float(value)
                except (ValueError, TypeError):
                    pass
            return str(field_value).lower() == str(value).lower()

        if operator == "!=":
            if field in NUMERIC_FIELDS:
                try:
                    return float(field_value) != float(value)
                except (ValueError, TypeError):
                    pass
            return str(field_value).lower() != str(value).lower()

        # Contains
        if operator == "contains":
            if isinstance(field_value, list):
                # Check if value is in the list (case-insensitive)
                return any(
                    value.lower() in str(item).lower()
                    for item in field_value
                )
            return value.lower() in str(field_value).lower()

        if operator == "not_contains":
            if isinstance(field_value, list):
                return not any(
                    value.lower() in str(item).lower()
                    for item in field_value
                )
            return value.lower() not in str(field_value).lower()

        # Glob matching
        if operator == "matches":
            return fnmatch.fnmatch(str(field_value), value)

        return True  # Unknown operator passes by default

    def delete_rule(self, rule_id: int) -> bool:
        """Deactivate a rule (soft delete).

        Returns True if the rule existed and was deactivated.
        """
        # Check if rule exists
        row = self._storage._conn.execute(
            "SELECT id FROM memory_rules WHERE id = ?", (rule_id,)
        ).fetchone()
        if row is None:
            return False
        self._storage.update_rule(rule_id, {"is_active": False})
        return True

    def get_all_rules(self) -> list[dict]:
        """Return all active rules, sorted by scope then priority."""
        rows = self._storage._conn.execute(
            "SELECT * FROM memory_rules WHERE is_active = 1 "
            "ORDER BY scope, priority DESC"
        ).fetchall()
        return self._storage._rows_to_dicts(rows)
