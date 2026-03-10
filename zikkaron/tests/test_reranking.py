"""Tests for v22-v25 reranking features: GTE-Reranker, NLI, multi-passage, dual-vector."""

import pytest

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.retrieval import HippoRetriever, _question_to_statement
from zikkaron.storage import StorageEngine


@pytest.fixture
def storage(tmp_path):
    engine = StorageEngine(str(tmp_path / "test_reranking.db"))
    yield engine
    engine.close()


@pytest.fixture
def settings():
    return Settings(
        DB_PATH=":memory:",
        GTE_RERANKER_ENABLED=False,
        NLI_RERANKING_ENABLED=False,
        MULTI_PASSAGE_RERANKING_ENABLED=False,
        DUAL_VECTORS_ENABLED=False,
        CROSS_ENCODER_ENABLED=False,
    )


@pytest.fixture
def embeddings():
    return EmbeddingEngine("all-MiniLM-L6-v2")


@pytest.fixture
def graph(storage, settings):
    return KnowledgeGraph(storage, settings)


@pytest.fixture
def retriever(storage, embeddings, graph, settings):
    return HippoRetriever(storage, embeddings, graph, settings)


def _make_memory(storage, embeddings, content, directory="/proj", tags=None):
    embedding = embeddings.encode(content)
    return storage.insert_memory({
        "content": content,
        "embedding": embedding,
        "tags": tags or [],
        "directory_context": directory,
        "heat": 1.0,
        "is_stale": False,
        "file_hash": None,
        "embedding_model": embeddings.get_model_name(),
    })


# ── _question_to_statement ────────────────────────────────────────────


class TestQuestionToStatement:
    def test_question_to_statement_prefer(self):
        result = _question_to_statement("Would Melanie prefer national parks?")
        assert "prefers" in result

    def test_question_to_statement_enjoy(self):
        result = _question_to_statement("Does Melanie enjoy camping?")
        assert "enjoys" in result

    def test_question_to_statement_is(self):
        result = _question_to_statement("Is Melanie adventurous?")
        assert "is" in result

    def test_question_to_statement_default(self):
        result = _question_to_statement("Can Melanie cook?")
        assert not result.startswith("Can")


# ── _cluster_memories ─────────────────────────────────────────────────


class TestClusterMemories:
    def test_cluster_memories_basic(self, retriever):
        """Overlapping memories should be grouped together."""
        memories = [
            {"content": "Melanie loves camping trips in the mountains"},
            {"content": "Melanie enjoys camping trips with her family"},
            {"content": "Camping trips in the mountains are relaxing"},
        ]
        clusters = retriever._cluster_memories(memories)
        # With high overlap these should form fewer clusters than 3
        total = sum(len(c) for c in clusters)
        assert total == 3
        assert len(clusters) < 3

    def test_cluster_memories_no_overlap(self, retriever):
        """Unrelated memories should each be in their own cluster."""
        memories = [
            {"content": "quantum chromodynamics gauge symmetry"},
            {"content": "baroque harpsichord concerto allegro"},
            {"content": "submarine hydrothermal vent chemosynthesis"},
        ]
        clusters = retriever._cluster_memories(memories)
        assert len(clusters) == 3


# ── _score_single_pair ────────────────────────────────────────────────


class TestScoreSinglePair:
    def test_score_single_pair_fallback(self, retriever):
        """Without any CE model loaded, should return 0.0."""
        assert retriever._gte_reranker is None
        score = retriever._score_single_pair("test query", "test document")
        assert score == 0.0


# ── _dual_vector_search ──────────────────────────────────────────────


class TestDualVectorSearch:
    def test_dual_vector_disabled(self, retriever, embeddings):
        """With DUAL_VECTORS_ENABLED=False, returns empty list."""
        embedding = embeddings.encode("test")
        result = retriever._dual_vector_search(embedding, top_k=5)
        assert result == []


# ── _multi_passage_rerank ────────────────────────────────────────────


class TestMultiPassageRerank:
    def test_multi_passage_disabled(self, retriever):
        """With MULTI_PASSAGE_RERANKING_ENABLED=False, returns input unchanged."""
        memories = [
            {"content": "memory one", "_retrieval_score": 0.9},
            {"content": "memory two", "_retrieval_score": 0.5},
        ]
        result = retriever._multi_passage_rerank("query", memories, top_k=5)
        assert result == memories


# ── All features toggleable ──────────────────────────────────────────


class TestAllFeaturesToggleable:
    def test_all_reranking_features_toggleable(self, storage, embeddings, graph):
        """Every reranking feature disabled — recall should still work."""
        s = Settings(
            DB_PATH=":memory:",
            GTE_RERANKER_ENABLED=False,
            NLI_RERANKING_ENABLED=False,
            MULTI_PASSAGE_RERANKING_ENABLED=False,
            DUAL_VECTORS_ENABLED=False,
            CROSS_ENCODER_ENABLED=False,
            QUERY_ROUTING_ENABLED=False,
        )
        r = HippoRetriever(storage, embeddings, graph, s)
        _make_memory(storage, embeddings, "test memory for toggle check")
        results = r.recall("test", max_results=5)
        assert isinstance(results, list)
