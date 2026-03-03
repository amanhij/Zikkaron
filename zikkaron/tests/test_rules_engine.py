"""Tests for neuro-symbolic rules engine — hard constraints, soft preferences, and retrieval integration."""

import pytest

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.retrieval import HippoRetriever
from zikkaron.rules_engine import (
    RulesEngine,
    _parse_action,
    _parse_condition,
)
from zikkaron.storage import StorageEngine


@pytest.fixture
def settings():
    return Settings(DB_PATH=":memory:")


@pytest.fixture
def storage(tmp_path):
    db_path = str(tmp_path / "test_rules.db")
    engine = StorageEngine(db_path)
    yield engine
    engine.close()


@pytest.fixture
def embeddings():
    return EmbeddingEngine("all-MiniLM-L6-v2")


@pytest.fixture
def engine(storage, settings):
    return RulesEngine(storage, settings)


def _make_memory_dict(
    mid, content, tags=None, directory="/project", heat=1.0,
    importance=0.5, confidence=1.0, surprise_score=0.0,
    score=0.5, store_type="episodic", compression_level=0,
):
    """Create a memory dict for testing (not persisted)."""
    return {
        "id": mid,
        "content": content,
        "tags": tags or [],
        "directory_context": directory,
        "heat": heat,
        "importance": importance,
        "confidence": confidence,
        "surprise_score": surprise_score,
        "emotional_valence": 0.0,
        "store_type": store_type,
        "compression_level": compression_level,
        "_retrieval_score": score,
    }


def _insert_memory(storage, embeddings, content, directory="/project", tags=None, **kwargs):
    """Helper to insert a real memory into storage."""
    embedding = embeddings.encode(content)
    mem = {
        "content": content,
        "embedding": embedding,
        "tags": tags or ["test"],
        "directory_context": directory,
        "heat": 1.0,
        "is_stale": False,
        "embedding_model": embeddings.get_model_name(),
    }
    mem.update(kwargs)
    return storage.insert_memory(mem)


# -- Parser tests --


class TestParseCondition:
    def test_equals(self):
        assert _parse_condition("language == typescript") == ("language", "==", "typescript")

    def test_not_equals(self):
        assert _parse_condition("store_type != episodic") == ("store_type", "!=", "episodic")

    def test_contains(self):
        assert _parse_condition("tag contains architecture") == ("tag", "contains", "architecture")

    def test_not_contains(self):
        assert _parse_condition("content not_contains password") == ("content", "not_contains", "password")

    def test_greater_than(self):
        assert _parse_condition("importance > 0.7") == ("importance", ">", "0.7")

    def test_less_than(self):
        assert _parse_condition("heat < 0.3") == ("heat", "<", "0.3")

    def test_greater_equal(self):
        assert _parse_condition("confidence >= 0.5") == ("confidence", ">=", "0.5")

    def test_less_equal(self):
        assert _parse_condition("surprise_score <= 0.1") == ("surprise_score", "<=", "0.1")

    def test_matches(self):
        assert _parse_condition("directory_context matches /project/*") == (
            "directory_context", "matches", "/project/*"
        )

    def test_invalid(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_condition("invalid condition without operator")


class TestParseAction:
    def test_filter(self):
        assert _parse_action("filter") == ("filter", 0.0)

    def test_boost(self):
        assert _parse_action("boost:0.3") == ("boost", 0.3)

    def test_penalty(self):
        assert _parse_action("penalty:0.2") == ("penalty", 0.2)

    def test_invalid(self):
        with pytest.raises(ValueError, match="Invalid action"):
            _parse_action("unknown_action")


# -- Rule management tests --


class TestAddHardRule:
    def test_creates_hard_filter_rule(self, engine):
        rule_id = engine.add_rule(
            rule_type="hard",
            scope="global",
            condition="content not_contains password",
            action="filter",
            priority=100,
        )
        assert rule_id > 0
        rules = engine.get_all_rules()
        assert len(rules) == 1
        assert rules[0]["rule_type"] == "hard"
        assert rules[0]["condition"] == "content not_contains password"
        assert rules[0]["action"] == "filter"
        assert rules[0]["priority"] == 100

    def test_hard_rule_rejects_boost_action(self, engine):
        with pytest.raises(ValueError, match="Hard rules must use 'filter'"):
            engine.add_rule("hard", "global", "importance > 0.5", "boost:0.3")

    def test_invalid_rule_type(self, engine):
        with pytest.raises(ValueError, match="rule_type"):
            engine.add_rule("unknown", "global", "importance > 0.5", "filter")

    def test_invalid_scope(self, engine):
        with pytest.raises(ValueError, match="scope"):
            engine.add_rule("hard", "unknown", "importance > 0.5", "filter")

    def test_invalid_condition(self, engine):
        with pytest.raises(ValueError, match="Cannot parse"):
            engine.add_rule("hard", "global", "bad condition", "filter")


class TestAddSoftRule:
    def test_creates_soft_boost_rule(self, engine):
        rule_id = engine.add_rule(
            rule_type="soft",
            scope="global",
            condition="tag contains architecture",
            action="boost:0.3",
            priority=5,
        )
        assert rule_id > 0
        rules = engine.get_all_rules()
        assert len(rules) == 1
        assert rules[0]["rule_type"] == "soft"
        assert rules[0]["action"] == "boost:0.3"

    def test_creates_soft_penalty_rule(self, engine):
        rule_id = engine.add_rule(
            rule_type="soft",
            scope="directory",
            condition="heat < 0.3",
            action="penalty:0.1",
            priority=2,
            scope_value="/project",
        )
        assert rule_id > 0


# -- Condition evaluation tests --


class TestEvaluateEquals:
    def test_string_equals(self, engine):
        mem = _make_memory_dict(1, "test content", store_type="semantic")
        assert engine.evaluate_condition("store_type == semantic", mem)
        assert not engine.evaluate_condition("store_type == episodic", mem)

    def test_case_insensitive(self, engine):
        mem = _make_memory_dict(1, "test content", store_type="Semantic")
        assert engine.evaluate_condition("store_type == semantic", mem)

    def test_numeric_equals(self, engine):
        mem = _make_memory_dict(1, "test", importance=0.7)
        assert engine.evaluate_condition("importance == 0.7", mem)

    def test_not_equals(self, engine):
        mem = _make_memory_dict(1, "test", store_type="episodic")
        assert engine.evaluate_condition("store_type != semantic", mem)
        assert not engine.evaluate_condition("store_type != episodic", mem)


class TestEvaluateContains:
    def test_string_contains(self, engine):
        mem = _make_memory_dict(1, "use TypeScript for frontend")
        assert engine.evaluate_condition("content contains TypeScript", mem)
        assert not engine.evaluate_condition("content contains Python", mem)

    def test_list_contains(self, engine):
        mem = _make_memory_dict(1, "test", tags=["architecture", "design", "frontend"])
        assert engine.evaluate_condition("tags contains architecture", mem)
        assert not engine.evaluate_condition("tags contains backend", mem)

    def test_tag_alias(self, engine):
        mem = _make_memory_dict(1, "test", tags=["architecture"])
        assert engine.evaluate_condition("tag contains architecture", mem)

    def test_not_contains_string(self, engine):
        mem = _make_memory_dict(1, "safe content here")
        assert engine.evaluate_condition("content not_contains password", mem)
        assert not engine.evaluate_condition("content not_contains safe", mem)

    def test_not_contains_list(self, engine):
        mem = _make_memory_dict(1, "test", tags=["frontend"])
        assert engine.evaluate_condition("tags not_contains backend", mem)
        assert not engine.evaluate_condition("tags not_contains frontend", mem)


class TestEvaluateNumericGt:
    def test_greater_than(self, engine):
        mem = _make_memory_dict(1, "test", importance=0.8)
        assert engine.evaluate_condition("importance > 0.7", mem)
        assert not engine.evaluate_condition("importance > 0.9", mem)

    def test_less_than(self, engine):
        mem = _make_memory_dict(1, "test", heat=0.2)
        assert engine.evaluate_condition("heat < 0.3", mem)
        assert not engine.evaluate_condition("heat < 0.1", mem)

    def test_greater_equal(self, engine):
        mem = _make_memory_dict(1, "test", confidence=0.5)
        assert engine.evaluate_condition("confidence >= 0.5", mem)
        assert not engine.evaluate_condition("confidence >= 0.6", mem)

    def test_less_equal(self, engine):
        mem = _make_memory_dict(1, "test", surprise_score=0.1)
        assert engine.evaluate_condition("surprise_score <= 0.1", mem)
        assert not engine.evaluate_condition("surprise_score <= 0.05", mem)

    def test_none_field_treated_as_zero(self, engine):
        mem = {"id": 1, "content": "test"}  # no importance field
        assert engine.evaluate_condition("importance > -1", mem)
        assert not engine.evaluate_condition("importance > 0", mem)


class TestEvaluateMatches:
    def test_glob_match(self, engine):
        mem = _make_memory_dict(1, "test", directory="/project/web/src")
        assert engine.evaluate_condition("directory_context matches /project/web/*", mem)
        assert not engine.evaluate_condition("directory_context matches /project/api/*", mem)

    def test_glob_wildcard(self, engine):
        mem = _make_memory_dict(1, "test", directory="/project/file.ts")
        assert engine.evaluate_condition("directory_context matches *.ts", mem)
        assert not engine.evaluate_condition("directory_context matches *.py", mem)


class TestTagKeyValueExtraction:
    def test_colon_separated_tag(self, engine):
        mem = _make_memory_dict(1, "test", tags=["language:typescript", "framework:react"])
        assert engine.evaluate_condition("language == typescript", mem)
        assert not engine.evaluate_condition("language == python", mem)

    def test_equals_separated_tag(self, engine):
        mem = _make_memory_dict(1, "test", tags=["language=typescript"])
        assert engine.evaluate_condition("language == typescript", mem)


# -- Rule application tests --


class TestHardRuleFilters:
    def test_filters_non_matching(self, engine):
        engine.add_rule("hard", "global", "content not_contains password", "filter", 10)

        memories = [
            _make_memory_dict(1, "database connection string"),
            _make_memory_dict(2, "the password is secret123"),
            _make_memory_dict(3, "api key configuration"),
        ]
        result = engine.apply_rules(memories, "/project")
        assert len(result) == 2
        assert all("password" not in m["content"] for m in result)

    def test_multiple_hard_rules(self, engine):
        engine.add_rule("hard", "global", "importance > 0.3", "filter", 10)
        engine.add_rule("hard", "global", "heat > 0.5", "filter", 5)

        memories = [
            _make_memory_dict(1, "high both", importance=0.8, heat=0.9),
            _make_memory_dict(2, "high importance only", importance=0.8, heat=0.2),
            _make_memory_dict(3, "high heat only", importance=0.1, heat=0.9),
            _make_memory_dict(4, "low both", importance=0.1, heat=0.2),
        ]
        result = engine.apply_rules(memories, "/project")
        assert len(result) == 1
        assert result[0]["id"] == 1


class TestSoftRuleBoosts:
    def test_boost_increases_score(self, engine):
        engine.add_rule("soft", "global", "tag contains architecture", "boost:0.3", 5)

        memories = [
            _make_memory_dict(1, "arch doc", tags=["architecture"], score=0.5),
            _make_memory_dict(2, "code snippet", tags=["code"], score=0.6),
        ]
        result = engine.apply_rules(memories, "/project")
        # Memory 1 should get boosted from 0.5 to 0.8, now above memory 2
        assert result[0]["id"] == 1
        assert result[0]["_retrieval_score"] == pytest.approx(0.8, abs=0.01)
        assert result[1]["id"] == 2
        assert result[1]["_retrieval_score"] == pytest.approx(0.6, abs=0.01)

    def test_penalty_decreases_score(self, engine):
        engine.add_rule("soft", "global", "heat < 0.3", "penalty:0.2", 5)

        memories = [
            _make_memory_dict(1, "cold memory", heat=0.2, score=0.7),
            _make_memory_dict(2, "hot memory", heat=0.9, score=0.6),
        ]
        result = engine.apply_rules(memories, "/project")
        # Memory 1 penalized from 0.7 to 0.5, so memory 2 (0.6) is now first
        assert result[0]["id"] == 2
        assert result[1]["id"] == 1
        assert result[1]["_retrieval_score"] == pytest.approx(0.5, abs=0.01)

    def test_combined_hard_and_soft(self, engine):
        engine.add_rule("hard", "global", "content not_contains secret", "filter", 100)
        engine.add_rule("soft", "global", "importance > 0.7", "boost:0.2", 5)

        memories = [
            _make_memory_dict(1, "important doc", importance=0.9, score=0.5),
            _make_memory_dict(2, "secret info", importance=0.9, score=0.8),
            _make_memory_dict(3, "normal doc", importance=0.3, score=0.6),
        ]
        result = engine.apply_rules(memories, "/project")
        # Memory 2 filtered by hard rule, memory 1 boosted
        assert len(result) == 2
        assert result[0]["id"] == 1  # 0.5 + 0.2 = 0.7
        assert result[1]["id"] == 3  # 0.6 unchanged


# -- Scoping tests --


class TestDirectoryScopedRule:
    def test_applies_in_matching_directory(self, engine):
        engine.add_rule(
            "hard", "directory", "importance > 0.5", "filter",
            priority=10, scope_value="/critical-project",
        )

        memories = [
            _make_memory_dict(1, "important", importance=0.8, score=0.5),
            _make_memory_dict(2, "trivial", importance=0.2, score=0.6),
        ]

        # Rule applies for matching directory
        result = engine.apply_rules(memories, "/critical-project/src")
        assert len(result) == 1
        assert result[0]["id"] == 1

    def test_does_not_apply_outside_directory(self, engine):
        engine.add_rule(
            "hard", "directory", "importance > 0.5", "filter",
            priority=10, scope_value="/critical-project",
        )

        memories = [
            _make_memory_dict(1, "important", importance=0.8, score=0.5),
            _make_memory_dict(2, "trivial", importance=0.2, score=0.6),
        ]

        # Rule does NOT apply for different directory
        result = engine.apply_rules(memories, "/other-project/src")
        assert len(result) == 2


class TestGlobalRuleAppliesEverywhere:
    def test_applies_to_any_directory(self, engine):
        engine.add_rule("hard", "global", "content not_contains banned", "filter", 10)

        memories = [
            _make_memory_dict(1, "good content"),
            _make_memory_dict(2, "this is banned content"),
        ]

        # Applies in any directory
        for directory in ["/project/a", "/project/b", "/other", "/"]:
            result = engine.apply_rules(list(memories), directory)
            assert len(result) == 1
            assert result[0]["id"] == 1


class TestFileScopedRule:
    def test_file_glob_matching(self, engine):
        engine.add_rule(
            "soft", "file", "tag contains typescript", "boost:0.5",
            priority=5, scope_value="*.ts",
        )

        memories = [
            _make_memory_dict(1, "ts code", tags=["typescript"], score=0.3),
            _make_memory_dict(2, "other code", tags=["python"], score=0.5),
        ]

        # Matches .ts file
        result = engine.apply_rules(memories, "src/app.ts")
        assert result[0]["id"] == 1  # boosted to 0.8

        # Doesn't match .py file — use fresh memory dicts, no boost applied
        memories2 = [
            _make_memory_dict(1, "ts code", tags=["typescript"], score=0.3),
            _make_memory_dict(2, "other code", tags=["python"], score=0.5),
        ]
        result = engine.apply_rules(memories2, "src/app.py")
        # No rules match, so scores unchanged
        scores = {m["id"]: m["_retrieval_score"] for m in result}
        assert scores[1] == pytest.approx(0.3, abs=0.01)  # no boost
        assert scores[2] == pytest.approx(0.5, abs=0.01)


# -- Priority ordering tests --


class TestPriorityOrdering:
    def test_higher_priority_applied_first(self, engine):
        # Low priority: boost all code tags
        engine.add_rule("soft", "global", "tag contains code", "boost:0.3", 1)
        # High priority: filter out low importance
        engine.add_rule("hard", "global", "importance > 0.3", "filter", 10)

        memories = [
            _make_memory_dict(1, "important code", tags=["code"], importance=0.8, score=0.5),
            _make_memory_dict(2, "trivial code", tags=["code"], importance=0.1, score=0.6),
        ]
        result = engine.apply_rules(memories, "/project")
        # Memory 2 filtered out first (higher priority), then memory 1 gets boosted
        assert len(result) == 1
        assert result[0]["id"] == 1
        assert result[0]["_retrieval_score"] == pytest.approx(0.8, abs=0.01)

    def test_rules_sorted_by_priority(self, engine):
        engine.add_rule("soft", "global", "importance > 0.5", "boost:0.1", 1)
        engine.add_rule("soft", "global", "heat > 0.5", "boost:0.1", 10)
        engine.add_rule("soft", "global", "confidence > 0.5", "boost:0.1", 5)

        rules = engine.get_applicable_rules("/project")
        priorities = [r["priority"] for r in rules]
        assert priorities == sorted(priorities, reverse=True)


# -- Rule deletion tests --


class TestRuleDeletion:
    def test_deleted_rule_no_longer_applies(self, engine):
        rule_id = engine.add_rule("hard", "global", "importance > 0.5", "filter", 10)

        memories = [
            _make_memory_dict(1, "important", importance=0.8, score=0.5),
            _make_memory_dict(2, "trivial", importance=0.2, score=0.6),
        ]

        # Before deletion: filters
        result = engine.apply_rules(list(memories), "/project")
        assert len(result) == 1

        # Delete rule
        assert engine.delete_rule(rule_id) is True

        # After deletion: no filtering
        result = engine.apply_rules(list(memories), "/project")
        assert len(result) == 2

    def test_delete_nonexistent_rule(self, engine):
        assert engine.delete_rule(99999) is False

    def test_deleted_rule_not_in_get_all(self, engine):
        rule_id = engine.add_rule("hard", "global", "importance > 0.5", "filter")
        engine.delete_rule(rule_id)
        assert len(engine.get_all_rules()) == 0


# -- Integration with HippoRetriever --


class TestIntegrationRecall:
    def test_rules_affect_recall_results(self, storage, embeddings, settings):
        kg = KnowledgeGraph(storage, settings)
        retriever = HippoRetriever(storage, embeddings, kg, settings)
        rules_engine = RulesEngine(storage, settings)
        retriever.set_rules_engine(rules_engine)

        # Insert memories with different content
        _insert_memory(
            storage, embeddings,
            "TypeScript is the best language for web development",
            tags=["typescript", "web"],
        )
        _insert_memory(
            storage, embeddings,
            "Python is great for data science and machine learning",
            tags=["python", "ml"],
        )
        _insert_memory(
            storage, embeddings,
            "JavaScript was the original web scripting language",
            tags=["javascript", "web"],
        )

        # Add a hard rule: only keep memories containing "TypeScript" OR "JavaScript"
        rules_engine.add_rule(
            "hard", "global", "content contains web", "filter", 10,
        )

        # Recall should filter based on rules
        results = retriever.recall("programming languages", max_results=10, min_heat=0.0)
        # Python memory should be filtered out (no "web" in content... wait it does)
        # Actually Python memory says "data science" not "web", so it should be filtered
        for r in results:
            assert "web" in r["content"].lower()

    def test_recall_without_rules_engine(self, storage, embeddings, settings):
        """Verify recall works fine when no rules engine is set."""
        kg = KnowledgeGraph(storage, settings)
        retriever = HippoRetriever(storage, embeddings, kg, settings)
        # No rules engine set

        _insert_memory(storage, embeddings, "some test memory", tags=["test"])

        results = retriever.recall("test memory", max_results=5, min_heat=0.0)
        assert len(results) >= 1


# -- MCP tool tests --


class TestMcpAddRuleTool:
    def test_add_rule_tool(self, storage, settings):
        """Test that the server-level add_rule tool function works."""
        from zikkaron.server import add_rule as _add_rule_tool
        from zikkaron.server import get_rules as _get_rules_tool

        # Temporarily set the global _rules_engine
        import zikkaron.server as srv
        original = srv._rules_engine
        try:
            srv._rules_engine = RulesEngine(storage, settings)

            result = _add_rule_tool(
                rule_type="soft",
                scope="global",
                condition="tag contains architecture",
                action="boost:0.3",
                priority=5,
                scope_value="",
            )
            assert result["status"] == "created"
            assert "rule_id" in result

            # Verify via get_rules
            rules = _get_rules_tool()
            assert len(rules) == 1
            assert rules[0]["condition"] == "tag contains architecture"
        finally:
            srv._rules_engine = original

    def test_add_rule_tool_validation_error(self, storage, settings):
        """Test that invalid rules return error."""
        from zikkaron.server import add_rule as _add_rule_tool

        import zikkaron.server as srv
        original = srv._rules_engine
        try:
            srv._rules_engine = RulesEngine(storage, settings)

            result = _add_rule_tool(
                rule_type="invalid",
                scope="global",
                condition="importance > 0.5",
                action="filter",
            )
            assert result["status"] == "error"
        finally:
            srv._rules_engine = original

    def test_add_rule_tool_not_initialized(self):
        """Test that add_rule returns error when engine not initialized."""
        from zikkaron.server import add_rule as _add_rule_tool

        import zikkaron.server as srv
        original = srv._rules_engine
        try:
            srv._rules_engine = None
            result = _add_rule_tool(
                rule_type="hard",
                scope="global",
                condition="importance > 0.5",
                action="filter",
            )
            assert result["status"] == "error"
        finally:
            srv._rules_engine = original

    def test_get_rules_with_directory_filter(self, storage, settings):
        """Test get_rules with directory scoping."""
        from zikkaron.server import get_rules as _get_rules_tool

        import zikkaron.server as srv
        original = srv._rules_engine
        try:
            srv._rules_engine = RulesEngine(storage, settings)
            rules_eng = srv._rules_engine

            rules_eng.add_rule("hard", "global", "importance > 0.5", "filter", 10)
            rules_eng.add_rule(
                "soft", "directory", "tag contains web", "boost:0.2",
                priority=5, scope_value="/web-project",
            )

            # All rules
            all_rules = _get_rules_tool()
            assert len(all_rules) == 2

            # Only applicable to /web-project
            web_rules = _get_rules_tool(directory="/web-project/src")
            assert len(web_rules) == 2  # global + matching directory

            # Only global rules for other directory
            other_rules = _get_rules_tool(directory="/other-project")
            assert len(other_rules) == 1  # only global
        finally:
            srv._rules_engine = original


# -- Edge case tests --


class TestEdgeCases:
    def test_empty_memories_list(self, engine):
        engine.add_rule("hard", "global", "importance > 0.5", "filter", 10)
        result = engine.apply_rules([], "/project")
        assert result == []

    def test_no_rules(self, engine):
        memories = [_make_memory_dict(1, "test", score=0.5)]
        result = engine.apply_rules(memories, "/project")
        assert len(result) == 1
        assert result[0]["_retrieval_score"] == 0.5

    def test_all_filtered_out(self, engine):
        engine.add_rule("hard", "global", "importance > 0.99", "filter", 10)
        memories = [
            _make_memory_dict(1, "a", importance=0.5, score=0.5),
            _make_memory_dict(2, "b", importance=0.3, score=0.6),
        ]
        result = engine.apply_rules(memories, "/project")
        assert len(result) == 0

    def test_multiple_soft_rules_stack(self, engine):
        engine.add_rule("soft", "global", "importance > 0.5", "boost:0.1", 5)
        engine.add_rule("soft", "global", "heat > 0.5", "boost:0.1", 3)

        mem = _make_memory_dict(1, "test", importance=0.8, heat=0.9, score=0.5)
        result = engine.apply_rules([mem], "/project")
        # Both boosts apply: 0.5 + 0.1 + 0.1 = 0.7
        assert result[0]["_retrieval_score"] == pytest.approx(0.7, abs=0.01)
