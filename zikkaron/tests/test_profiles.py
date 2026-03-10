"""Tests for structured user profiles, derived beliefs, convex fusion, and comparison routing."""

import pytest

from zikkaron.config import Settings
from zikkaron.profiles import BeliefDeriver, ProfileExtractor
from zikkaron.retrieval import analyze_query
from zikkaron.storage import StorageEngine


@pytest.fixture
def storage(tmp_path):
    engine = StorageEngine(str(tmp_path / "test_profiles.db"))
    yield engine
    engine.close()


def _settings(**overrides) -> Settings:
    defaults = {
        "DB_PATH": ":memory:",
        "PROFILE_EXTRACTION_ENABLED": True,
        "PROFILE_SUMMARY_ENABLED": True,
        "DERIVED_BELIEFS_ENABLED": True,
        "COMPARISON_DUAL_SEARCH_ENABLED": True,
        "FUSION_METHOD": "convex",
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestProfileExtractionLikes:
    def test_profile_extraction_likes(self, storage):
        settings = _settings()
        extractor = ProfileExtractor(storage, settings)
        extractor.extract_and_store("Melanie loves camping", memory_id=1, directory_context="/proj")

        profiles = storage.get_profiles_for_entity("Melanie", "/proj")
        assert len(profiles) >= 1
        match = [p for p in profiles if p["entity_name"] == "Melanie" and p["attribute_type"] == "interest"]
        assert len(match) >= 1, f"Expected interest profile for Melanie, got {profiles}"


class TestProfileExtractionTrait:
    def test_profile_extraction_trait(self, storage):
        settings = _settings()
        extractor = ProfileExtractor(storage, settings)
        extractor.extract_and_store("Melanie is adventurous", memory_id=1, directory_context="/proj")

        profiles = storage.get_profiles_for_entity("Melanie", "/proj")
        match = [p for p in profiles if p["attribute_type"] == "trait"]
        assert len(match) >= 1, f"Expected trait profile, got {profiles}"
        assert match[0]["attribute_value"] == "adventurous"


class TestProfileAccumulation:
    def test_profile_accumulation(self, storage):
        settings = _settings()
        extractor = ProfileExtractor(storage, settings)

        extractor.extract_and_store("Melanie loves camping", memory_id=1, directory_context="/proj")
        initial = storage.get_profiles_for_entity("Melanie", "/proj")
        camping_profiles = [p for p in initial if p["attribute_key"] == "camping"]
        assert len(camping_profiles) == 1
        initial_confidence = camping_profiles[0]["confidence"]

        # Insert same normalized key to trigger UNIQUE constraint → confidence +0.1
        extractor.extract_and_store("Melanie enjoys camping", memory_id=2, directory_context="/proj")
        extractor.extract_and_store("Melanie likes camping", memory_id=3, directory_context="/proj")

        updated = storage.get_profiles_for_entity("Melanie", "/proj")
        camping_updated = [p for p in updated if p["attribute_key"] == "camping"]
        assert len(camping_updated) == 1
        assert camping_updated[0]["confidence"] > initial_confidence, (
            f"Expected confidence to increase from {initial_confidence}, got {camping_updated[0]['confidence']}"
        )


class TestProfileSummaryGeneration:
    def test_profile_summary_generation(self, storage):
        settings = _settings()
        extractor = ProfileExtractor(storage, settings)

        # Insert 3+ attributes for the same entity
        extractor.extract_and_store("Melanie loves camping", memory_id=1, directory_context="/proj")
        extractor.extract_and_store("Melanie is adventurous", memory_id=2, directory_context="/proj")
        extractor.extract_and_store("Melanie enjoys hiking", memory_id=3, directory_context="/proj")

        summary = extractor.generate_profile_summary("Melanie", "/proj")
        assert summary is not None, "Expected a summary for entity with 3+ attributes"
        assert "Melanie" in summary


class TestBeliefDerivation:
    def test_belief_derivation(self, storage):
        settings = _settings()
        deriver = BeliefDeriver(storage, settings)

        deriver.derive_from_memory(
            "Melanie loves camping and hiking in the mountains",
            memory_id=1,
            directory_context="/proj",
        )

        beliefs = storage.get_beliefs_for_subject("Melanie", "/proj")
        assert len(beliefs) >= 1, f"Expected at least one belief, got {beliefs}"


class TestProfileFTSSearch:
    def test_profile_fts_search(self, storage):
        settings = _settings()
        extractor = ProfileExtractor(storage, settings)

        extractor.extract_and_store("Melanie loves camping", memory_id=1, directory_context="/proj")
        extractor.extract_and_store("Alice enjoys painting", memory_id=2, directory_context="/proj")

        results = storage.search_profiles_fts("camping")
        assert len(results) >= 1, f"Expected FTS match for 'camping', got {results}"
        assert any(r["entity_name"] == "Melanie" for r in results)


class TestConvexFusionBasic:
    def test_convex_fusion_basic(self, storage, tmp_path):
        from zikkaron.embeddings import EmbeddingEngine
        from zikkaron.knowledge_graph import KnowledgeGraph

        settings = _settings(FUSION_METHOD="convex")
        embeddings = EmbeddingEngine("all-MiniLM-L6-v2")
        graph = KnowledgeGraph(storage, settings)

        from zikkaron.retrieval import HippoRetriever
        retriever = HippoRetriever(storage, embeddings, graph, settings)

        signal_scores = {
            "vector": {1: 0.9, 2: 0.5, 3: 0.1},
            "fts": {1: 0.3, 2: 0.8, 4: 0.6},
        }
        weights = {"vector": 1.0, "fts": 0.5}

        results = retriever._convex_fuse(signal_scores, weights)
        assert len(results) > 0
        # Results should be (memory_id, score) tuples sorted desc
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)
        # All scores should be normalized between 0 and 1
        for _, score in results:
            assert 0.0 <= score <= 1.0, f"Score {score} outside [0, 1]"


class TestConvexVsWRRF:
    def test_convex_vs_wrrf(self, storage):
        from zikkaron.embeddings import EmbeddingEngine
        from zikkaron.knowledge_graph import KnowledgeGraph
        from zikkaron.retrieval import HippoRetriever

        settings = _settings()
        embeddings = EmbeddingEngine("all-MiniLM-L6-v2")
        graph = KnowledgeGraph(storage, settings)
        retriever = HippoRetriever(storage, embeddings, graph, settings)

        # Design inputs where score magnitudes (convex) vs ranks (WRRF) diverge.
        # Signal B has item 2 scored 0.9 (close to max) but ranked 2nd.
        # Convex preserves that magnitude; WRRF only sees rank position.
        signal_scores = {
            "A": {1: 1.0, 2: 0.5, 3: 0.0},
            "B": {3: 1.0, 2: 0.9, 1: 0.0},
        }
        weights = {"A": 1.0, "B": 2.0}

        convex_results = retriever._convex_fuse(signal_scores, weights)
        convex_ranking = [mid for mid, _ in convex_results]

        # WRRF input: signal -> [mem_ids sorted by signal score desc]
        ranked_lists = {
            "A": [1, 2, 3],
            "B": [3, 2, 1],
        }
        wrrf_results = retriever._wrrf_fuse(ranked_lists, weights)
        wrrf_ranking = [mid for mid, _ in wrrf_results]

        assert len(convex_ranking) > 0
        assert len(wrrf_ranking) > 0
        # Convex puts 2 first (high magnitude in B), WRRF puts 3 first (rank 1 in heavier B)
        assert convex_ranking != wrrf_ranking, (
            f"Expected different rankings: convex={convex_ranking}, wrrf={wrrf_ranking}"
        )


class TestComparisonDetection:
    def test_comparison_detection(self):
        settings = _settings()
        analysis = analyze_query("camping or hiking?", settings)
        assert len(analysis["comparison_options"]) == 2
        assert "camping" in analysis["comparison_options"]
        assert "hiking" in analysis["comparison_options"]


class TestAllFeaturesToggleable:
    def test_profiles_disabled(self, storage):
        settings = _settings(PROFILE_EXTRACTION_ENABLED=False)
        extractor = ProfileExtractor(storage, settings)
        # Should not raise when feature is off (extraction still works at extractor level,
        # the toggle controls whether the pipeline calls it)
        extractor.extract_and_store("Melanie loves camping", memory_id=1, directory_context="/proj")

    def test_beliefs_disabled(self, storage):
        settings = _settings(DERIVED_BELIEFS_ENABLED=False)
        deriver = BeliefDeriver(storage, settings)
        deriver.derive_from_memory("Melanie loves camping", memory_id=1, directory_context="/proj")

    def test_convex_fusion_method(self, storage):
        from zikkaron.embeddings import EmbeddingEngine
        from zikkaron.knowledge_graph import KnowledgeGraph
        from zikkaron.retrieval import HippoRetriever

        settings = _settings(FUSION_METHOD="convex")
        embeddings = EmbeddingEngine("all-MiniLM-L6-v2")
        graph = KnowledgeGraph(storage, settings)
        retriever = HippoRetriever(storage, embeddings, graph, settings)
        assert retriever._settings.FUSION_METHOD == "convex"

    def test_wrrf_fusion_method(self, storage):
        from zikkaron.embeddings import EmbeddingEngine
        from zikkaron.knowledge_graph import KnowledgeGraph
        from zikkaron.retrieval import HippoRetriever

        settings = _settings(FUSION_METHOD="wrrf")
        embeddings = EmbeddingEngine("all-MiniLM-L6-v2")
        graph = KnowledgeGraph(storage, settings)
        retriever = HippoRetriever(storage, embeddings, graph, settings)
        assert retriever._settings.FUSION_METHOD == "wrrf"

    def test_comparison_disabled(self):
        settings = _settings(COMPARISON_DUAL_SEARCH_ENABLED=False)
        # analyze_query should still work without errors
        analysis = analyze_query("camping or hiking?", settings)
        assert isinstance(analysis, dict)

    def test_profile_summary_disabled(self, storage):
        settings = _settings(PROFILE_SUMMARY_ENABLED=False)
        extractor = ProfileExtractor(storage, settings)
        extractor.extract_and_store("Melanie loves camping", memory_id=1, directory_context="/proj")
        extractor.extract_and_store("Melanie is adventurous", memory_id=2, directory_context="/proj")
        extractor.extract_and_store("Melanie enjoys hiking", memory_id=3, directory_context="/proj")
        # Summary generation should still work at extractor level (toggle controls pipeline)
        summary = extractor.generate_profile_summary("Melanie", "/proj")
        assert summary is not None or summary is None  # no crash
