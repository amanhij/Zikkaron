import numpy as np
import pytest

from zikkaron.config import Settings
from zikkaron.enrichment import (
    ConceptNetExpander,
    EnrichmentPipeline,
    EnrichmentResult,
    FPAFilter,
    HARDCODED_EXPANSIONS,
    LogicExpander,
)


class MockEmbeddingEngine:
    def encode_query(self, text):
        np.random.seed(abs(hash(text)) % (2**31))
        emb = np.random.randn(768).astype(np.float32)
        return emb / np.linalg.norm(emb)


def _settings(**overrides) -> Settings:
    defaults = {
        "INDEX_ENRICHMENT_ENABLED": True,
        "CONCEPTNET_ENRICHMENT_ENABLED": False,
        "COMET_ENRICHMENT_ENABLED": False,
        "DOC2QUERY_ENRICHMENT_ENABLED": False,
        "LOGIC_ENRICHMENT_ENABLED": False,
        "ENRICHMENT_MIN_CONTENT_LENGTH": 20,
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestLogicExpander:
    def test_logic_expander_hypernym(self):
        expander = LogicExpander()
        result = expander.expand("went camping at Yellowstone")
        assert any(
            term in result for term in ("national_park", "outdoor")
        ), f"Expected hypernym in {result}"

    def test_logic_expander_verb_nominalization(self):
        expander = LogicExpander()
        result = expander.expand("enjoys reading")
        assert any(
            "reading" in term for term in result
        ), f"Expected reading nominalization in {result}"


class TestConceptNetExpander:
    def test_hardcoded_fallback(self):
        expander = ConceptNetExpander()
        # Force lite and http unavailable so it falls back to hardcoded
        expander._lite_available = False
        expander._http_available = False
        settings = _settings()
        result = expander.expand("camping", settings)
        expected = HARDCODED_EXPANSIONS["camping"]
        assert len(result) > 0, "Expected hardcoded expansions"
        for term in result:
            assert term in expected, f"{term} not in hardcoded camping expansions"


class TestFPAFilter:
    def _make_embedding(self, text: str) -> bytes:
        engine = MockEmbeddingEngine()
        vec = engine.encode_query(text)
        return vec.tobytes()

    def test_fpa_filter_accepts_related(self):
        """Similar embeddings (same text) should pass the filter."""
        engine = MockEmbeddingEngine()
        original = engine.encode_query("camping outdoors nature")
        fpa = FPAFilter(engine)
        # Use the same text so cosine similarity is 1.0
        kept = fpa.filter(
            original.tobytes(),
            ["camping outdoors nature"],
            threshold=0.25,
        )
        assert len(kept) == 1

    def test_fpa_filter_rejects_unrelated(self):
        """Distant embeddings should be rejected by the filter."""

        class DistantEmbeddingEngine:
            """Returns orthogonal embeddings for different texts."""
            def __init__(self):
                self._call_count = 0

            def encode_query(self, text):
                # Return a vector with a single 1.0 at a unique position
                emb = np.zeros(768, dtype=np.float32)
                emb[self._call_count % 768] = 1.0
                self._call_count += 1
                return emb

        engine = DistantEmbeddingEngine()
        # Original embedding: [1, 0, 0, ...]
        original = engine.encode_query("original concept")
        fpa = FPAFilter(engine)
        # Next calls produce orthogonal vectors → cosine = 0
        kept = fpa.filter(
            original.tobytes(),
            ["unrelated1", "unrelated2", "unrelated3"],
            threshold=0.25,
        )
        assert len(kept) == 0, f"Expected all rejected, got {kept}"


class TestEnrichmentResult:
    def test_dataclass_defaults(self):
        result = EnrichmentResult()
        assert result.concepts == []
        assert result.comet_inferences == []
        assert result.queries == []
        assert result.logic_expansions == []
        assert result.enriched_content == ""
        assert result.model_versions == {}

    def test_dataclass_fields(self):
        result = EnrichmentResult(
            concepts=["a"],
            logic_expansions=["b"],
            enriched_content="test",
        )
        assert result.concepts == ["a"]
        assert result.logic_expansions == ["b"]
        assert result.enriched_content == "test"


class TestEnrichmentPipeline:
    def _dummy_embedding(self) -> bytes:
        vec = np.ones(768, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tobytes()

    def test_short_content_skips(self):
        settings = _settings(ENRICHMENT_MIN_CONTENT_LENGTH=20)
        pipeline = EnrichmentPipeline(settings)
        result = pipeline.enrich("short", self._dummy_embedding(), settings)
        assert result.concepts == []
        assert result.comet_inferences == []
        assert result.queries == []
        assert result.logic_expansions == []
        assert result.enriched_content == "short"

    def test_logic_only(self):
        settings = _settings(
            LOGIC_ENRICHMENT_ENABLED=True,
            ENRICHMENT_MIN_CONTENT_LENGTH=5,
        )
        pipeline = EnrichmentPipeline(settings)
        content = "went camping at Yellowstone last summer"
        result = pipeline.enrich(content, self._dummy_embedding(), settings)
        assert result.concepts == []
        assert result.comet_inferences == []
        assert result.queries == []
        assert len(result.logic_expansions) > 0, "Expected logic expansions"
        assert "[enrichment]" in result.enriched_content
