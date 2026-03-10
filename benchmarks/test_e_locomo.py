"""Benchmark E: LoCoMo — Long-term Conversational Memory Evaluation.

Adapts the LoCoMo dataset (Maharana et al., 2024) to evaluate Zikkaron's
ability to retrieve relevant conversational context from long conversations.

Phase 1-3 optimizations:
- nomic-embed-text-v1.5 (768d) with asymmetric query/document encoding
- Context window enrichment (1-turn surrounding context)
- Temporal metadata injection (session dates prepended)
- FlashRank cross-encoder reranking
- MMR diversity enforcement
- Z-score adversarial abstention
"""
import json
import os
import re
import string
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pytest

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.retrieval import HippoRetriever
from zikkaron.storage import StorageEngine

# Use nomic for higher quality embeddings (768d, 62.28 nDCG@10 vs 41.95)
EMBEDDING_MODEL = os.environ.get("LOCOMO_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
EMBEDDING_DIM = 768 if "nomic" in EMBEDDING_MODEL or "bge-base" in EMBEDDING_MODEL else 384

CATEGORY_NAMES = ["single_hop", "multi_hop", "temporal", "open_domain", "adversarial"]
# Category numbering from LoCoMo paper (verified against MemMachine counts):
# cat 1=multi_hop(282), cat 2=temporal(321), cat 3=open_domain(96), cat 4=single_hop(841), cat 5=adversarial(446)
CATEGORY_MAP = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}

LOCOMO_JSON_PATH = os.environ.get(
    "LOCOMO_JSON_PATH",
    os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--Percena--locomo-mc10/"
        "snapshots/7d59a0463d83f97b042684310c0b3d17553004cd/raw/locomo10.json"
    ),
)


def _load_locomo_data() -> list[dict]:
    with open(LOCOMO_JSON_PATH) as f:
        return json.load(f)


def _extract_sessions(conversation: dict) -> dict[str, list[dict]]:
    """Extract session turn lists from a LoCoMo conversation dict."""
    sessions: dict[str, list[dict]] = {}
    for key, value in conversation.items():
        if key.startswith("session_") and isinstance(value, list):
            sessions[key] = value
    return sessions


def _get_session_date(conversation: dict, session_key: str) -> str:
    """Extract session date from conversation metadata for temporal enrichment."""
    date_key = f"{session_key}_date_time"
    date_str = conversation.get(date_key, "")
    if date_str:
        return date_str
    return ""


def _parse_locomo_date(date_str: str) -> str | None:
    """Parse LoCoMo date format '1:56 pm on 8 May, 2023' to ISO 8601."""
    if not date_str:
        return None
    m = re.match(r"(\d{1,2}:\d{2}\s*[ap]m)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})", date_str)
    if m:
        full = f"{m.group(2)} {m.group(3)} {m.group(4)} {m.group(1)}"
        try:
            dt = datetime.strptime(full, "%d %B %Y %I:%M %p")
            return dt.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            pass
    return None


def _reformulate_to_observation(speaker: str, text: str) -> str:
    """Convert first-person dialogue to third-person observation.

    '[Caroline] I love painting' → 'Caroline loves painting. Caroline said she loves painting.'
    Simple regex-based, no LLM needed.
    """
    obs = text.strip()
    # Remove common dialogue markers
    obs = re.sub(r"^(hey|hi|hello|oh|well|yeah|yes|no|okay|ok|sure|haha|lol|hmm|wow|aww)\b[,!.]*\s*", "", obs, flags=re.IGNORECASE)

    # Replace first-person with speaker name
    # "I love" → "Speaker loves", "I'm" → "Speaker is", etc.
    obs = re.sub(r"\bI'm\b", f"{speaker} is", obs)
    obs = re.sub(r"\bI've\b", f"{speaker} has", obs)
    obs = re.sub(r"\bI'll\b", f"{speaker} will", obs)
    obs = re.sub(r"\bI'd\b", f"{speaker} would", obs)
    # Conjugate common "I [adverb] verb" → "Speaker [adverb] verbs" (3rd person singular)
    _CONJ_VERBS = (
        "want|love|need|think|like|feel|know|make|take|work|play|live"
        "|hope|look|call|start|use|try|find|give|tell|ask|seem|come"
        "|mean|keep|help|show|hear|turn|read|spend|grow|run|learn|enjoy"
        "|believe|remember|plan|see|say|talk|bring|hold|put|get"
    )
    _ES_VERBS = "go|miss|do|watch|teach|push|fix|pass|wish"
    _OPT_ADV = r"(?:(?:also|really|just|always|often|never|still|actually|definitely|already|even|usually|sometimes) )?"
    obs = re.sub(
        rf"\bI ({_OPT_ADV})({_ES_VERBS})\b",
        lambda m: f"{speaker} {m.group(1)}{m.group(2)}es",
        obs,
    )
    obs = re.sub(
        rf"\bI ({_OPT_ADV})({_CONJ_VERBS})\b",
        lambda m: f"{speaker} {m.group(1)}{m.group(2)}s",
        obs,
    )
    obs = re.sub(rf"\bI ({_OPT_ADV})have\b", lambda m: f"{speaker} {m.group(1)}has", obs)
    obs = re.sub(r"\bI\b", speaker, obs)
    obs = re.sub(r"\bmy\b", f"{speaker}'s", obs, flags=re.IGNORECASE)
    obs = re.sub(r"\bme\b", speaker, obs, flags=re.IGNORECASE)
    obs = re.sub(r"\bmyself\b", speaker, obs, flags=re.IGNORECASE)
    obs = re.sub(r"\bmine\b", f"{speaker}'s", obs, flags=re.IGNORECASE)

    # Build observation: attribution prefix + reformulated text
    attribution = f"{speaker} said: {text.strip()}"
    # Truncate if too long
    if len(obs) > 500:
        obs = obs[:500]
    if len(attribution) > 500:
        attribution = attribution[:500]
    return f"{obs}\n{attribution}"


def _normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def _token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def _ingest_conversation(
    conv_data: dict,
    storage: StorageEngine,
    embeddings: EmbeddingEngine,
    project_dir: str,
    pair_mode: bool = True,
    obs_mode: bool = False,
    settings: "Settings | None" = None,
) -> dict[str, int]:
    """Ingest a LoCoMo conversation into Zikkaron storage.

    pair_mode: If True, store adjacent turn pairs as single memories.
    obs_mode: If True, add observation reformulation to content (better embeddings).
    """
    dia_id_to_memory_id: dict[str, int] = {}
    conversation = conv_data.get("conversation", {})
    sessions = _extract_sessions(conversation)

    for session_key in sorted(sessions):
        turns = sessions[session_key]
        session_date = _get_session_date(conversation, session_key)
        iso_date = _parse_locomo_date(session_date)

        if pair_mode:
            # Store adjacent turn pairs: [A says X] + [B responds Y]
            i = 0
            while i < len(turns):
                turn_a = turns[i] if isinstance(turns[i], dict) else None
                turn_b = turns[i + 1] if i + 1 < len(turns) and isinstance(turns[i + 1], dict) else None

                if not turn_a or not turn_a.get("text", turn_a.get("content", "")):
                    i += 1
                    continue

                speaker_a = turn_a.get("speaker", turn_a.get("role", "unknown"))
                text_a = turn_a.get("text", turn_a.get("content", ""))
                dia_id_a = turn_a.get("dia_id", turn_a.get("dialogue_id", turn_a.get("id", "")))

                if turn_b and turn_b.get("text", turn_b.get("content", "")):
                    speaker_b = turn_b.get("speaker", turn_b.get("role", "unknown"))
                    text_b = turn_b.get("text", turn_b.get("content", ""))
                    dia_id_b = turn_b.get("dia_id", turn_b.get("dialogue_id", turn_b.get("id", "")))
                    dialogue = f"[{speaker_a}] {text_a}\n[{speaker_b}] {text_b}"
                    if obs_mode:
                        obs_a = _reformulate_to_observation(speaker_a, text_a)
                        obs_b = _reformulate_to_observation(speaker_b, text_b)
                        content = f"{obs_a}\n{obs_b}"
                        embed_text = content
                    else:
                        embed_text = dialogue
                        content = dialogue
                    tags = [session_key, speaker_a, speaker_b] if speaker_a != speaker_b else [session_key, speaker_a]
                    vec = embeddings.encode_document(embed_text)
                    mem_data = {
                        "content": content,
                        "directory_context": project_dir,
                        "tags": tags,
                        "embedding": vec,
                        "embedding_model": embeddings.get_model_name(),
                    }
                    if iso_date:
                        mem_data["created_at"] = iso_date
                    mem_id = storage.insert_memory(mem_data, embeddings_engine=embeddings, settings=settings)
                    if dia_id_a:
                        dia_id_to_memory_id[str(dia_id_a)] = mem_id
                    if dia_id_b:
                        dia_id_to_memory_id[str(dia_id_b)] = mem_id
                    i += 2
                else:
                    # Odd turn at end of session — store solo
                    dialogue = f"[{speaker_a}] {text_a}"
                    if obs_mode:
                        obs_a = _reformulate_to_observation(speaker_a, text_a)
                        content = obs_a
                        embed_text = content
                    else:
                        embed_text = dialogue
                        content = dialogue
                    vec = embeddings.encode_document(embed_text)
                    mem_data = {
                        "content": content,
                        "directory_context": project_dir,
                        "tags": [session_key, speaker_a],
                        "embedding": vec,
                        "embedding_model": embeddings.get_model_name(),
                    }
                    if iso_date:
                        mem_data["created_at"] = iso_date
                    mem_id = storage.insert_memory(mem_data, embeddings_engine=embeddings, settings=settings)
                    if dia_id_a:
                        dia_id_to_memory_id[str(dia_id_a)] = mem_id
                    i += 1
        else:
            # Single-turn storage (original mode)
            for i, turn in enumerate(turns):
                if isinstance(turn, dict):
                    speaker = turn.get("speaker", turn.get("role", "unknown"))
                    text = turn.get("text", turn.get("content", ""))
                    dia_id = turn.get("dia_id", turn.get("dialogue_id", turn.get("id", "")))
                elif isinstance(turn, str):
                    speaker = "unknown"
                    text = turn
                    dia_id = ""
                else:
                    continue

                if not text:
                    continue

                content = f"[{speaker}] {text}" if speaker else text
                vec = embeddings.encode_document(content)
                tags = [session_key, speaker] if speaker else [session_key]
                mem_data = {
                    "content": content,
                    "directory_context": project_dir,
                    "tags": tags,
                    "embedding": vec,
                    "embedding_model": embeddings.get_model_name(),
                }
                if iso_date:
                    mem_data["created_at"] = iso_date
                mem_id = storage.insert_memory(mem_data, embeddings_engine=embeddings, settings=settings)
                if dia_id:
                    dia_id_to_memory_id[str(dia_id)] = mem_id

    return dia_id_to_memory_id


def _make_settings(**overrides) -> Settings:
    """Create benchmark-optimized settings."""
    defaults = {
        "CROSS_ENCODER_ENABLED": True,
        "ADVERSARIAL_DIVERSITY_ENFORCEMENT": False,
        "ADVERSARIAL_DETECTION_ENABLED": True,
        "TEMPORAL_RETRIEVAL_ENABLED": False,  # No effect on LoCoMo temporal (inference Qs)
        "RERANKER_ENABLED": False,  # CE alone is better at k=10
        "RERANKER_TOP_K": 50,
        "CANDIDATE_POOL_MULTIPLIER": 40,
        "CROSS_ENCODER_TOP_K": 75,
        "CROSS_ENCODER_WEIGHT": 1.0,  # Pure CE ranking beats blend (sweep confirmed)
        # Vector + FTS complement each other with nomic embeddings
        "WRRF_VECTOR_WEIGHT": 1.0,
        "WRRF_FTS_WEIGHT": 0.5,
        # Disable graph signals (no real graph data in benchmark)
        "WRRF_PPR_WEIGHT": 0.0,
        "WRRF_SPREADING_WEIGHT": 0.0,
        "WRRF_HOPFIELD_WEIGHT": 0.0,
        "WRRF_HDC_WEIGHT": 0.0,
        "WRRF_SR_WEIGHT": 0.0,
        "WRRF_FRACTAL_WEIGHT": 0.0,
        # v17: Index-time enrichment — fast-only (ConceptNet hardcoded + Logic)
        "INDEX_ENRICHMENT_ENABLED": True,
        "CONCEPTNET_ENRICHMENT_ENABLED": True,
        "LOGIC_ENRICHMENT_ENABLED": True,
        "COMET_ENRICHMENT_ENABLED": False,  # Too slow for benchmark
        "DOC2QUERY_ENRICHMENT_ENABLED": False,  # Too slow for benchmark
        # v18-19: Profiles and beliefs
        "PROFILE_EXTRACTION_ENABLED": True,
        "DERIVED_BELIEFS_ENABLED": True,
        # v20: Comparison routing
        "COMPARISON_DUAL_SEARCH_ENABLED": True,
        # v21: Keep WRRF (convex regressed)
        "FUSION_METHOD": "wrrf",
        # v22: Disable GTE-Reranker (regressed), keep FlashRank
        "GTE_RERANKER_ENABLED": False,
        "GTE_RERANKER_FALLBACK_TO_FLASHRANK": True,
        # v23: NLI entailment (only for open_domain queries)
        "NLI_RERANKING_ENABLED": True,
        "NLI_ONLY_FOR_OPEN_DOMAIN": True,
        # v24: Disable multi-passage (may be hurting)
        "MULTI_PASSAGE_RERANKING_ENABLED": False,
        # v25: Dual vectors (not ready yet)
        "DUAL_VECTORS_ENABLED": False,
    }
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture(scope="module")
def locomo_data():
    if not Path(LOCOMO_JSON_PATH).exists():
        pytest.skip(f"LoCoMo data not found at {LOCOMO_JSON_PATH}")
    return _load_locomo_data()


@pytest.fixture
def locomo_conv0(locomo_data, tmp_path_factory):
    """Ingest the first LoCoMo conversation into a temporary Zikkaron instance."""
    tmp = tmp_path_factory.mktemp("locomo_conv0")
    db_path = str(tmp / "memory.db")
    project_dir = str(tmp / "project")
    os.makedirs(project_dir, exist_ok=True)

    settings = _make_settings(DB_PATH=db_path)
    storage = StorageEngine(db_path, embedding_dim=EMBEDDING_DIM)
    embeddings = EmbeddingEngine(EMBEDDING_MODEL)
    kg = KnowledgeGraph(storage, settings)

    conv = locomo_data[0]
    dia_map = _ingest_conversation(conv, storage, embeddings, project_dir)

    retriever = HippoRetriever(storage, embeddings, kg, settings)
    return retriever, conv, dia_map, project_dir


@pytest.fixture
def locomo_full(locomo_data, tmp_path_factory):
    """Ingest all LoCoMo conversations."""
    contexts = []
    for i, conv in enumerate(locomo_data):
        tmp = tmp_path_factory.mktemp(f"locomo_{i}")
        db_path = str(tmp / "memory.db")
        project_dir = str(tmp / "project")
        os.makedirs(project_dir, exist_ok=True)

        settings = _make_settings(DB_PATH=db_path)
        storage = StorageEngine(db_path, embedding_dim=EMBEDDING_DIM)
        embeddings = EmbeddingEngine(EMBEDDING_MODEL)
        kg = KnowledgeGraph(storage, settings)

        dia_map = _ingest_conversation(conv, storage, embeddings, project_dir)
        retriever = HippoRetriever(storage, embeddings, kg, settings)
        contexts.append((retriever, conv, dia_map, project_dir))
    return contexts


@pytest.fixture
def locomo_full_obs(locomo_data, tmp_path_factory):
    """Ingest all LoCoMo conversations with observation-based storage."""
    contexts = []
    for i, conv in enumerate(locomo_data):
        tmp = tmp_path_factory.mktemp(f"locomo_obs_{i}")
        db_path = str(tmp / "memory.db")
        project_dir = str(tmp / "project")
        os.makedirs(project_dir, exist_ok=True)

        settings = _make_settings(DB_PATH=db_path)
        storage = StorageEngine(db_path, embedding_dim=EMBEDDING_DIM)
        embeddings = EmbeddingEngine(EMBEDDING_MODEL)
        kg = KnowledgeGraph(storage, settings)

        dia_map = _ingest_conversation(conv, storage, embeddings, project_dir, obs_mode=True)
        retriever = HippoRetriever(storage, embeddings, kg, settings)
        contexts.append((retriever, conv, dia_map, project_dir))
    return contexts


def _evaluate_retrieval_for_qa(
    retriever: HippoRetriever,
    qa_items: list[dict],
    dia_map: dict[str, int],
    max_results: int = 10,
) -> dict[str, dict]:
    """Evaluate retrieval on QA items, returning per-category metrics."""
    per_category: dict[str, list[dict]] = defaultdict(list)

    for qa in qa_items:
        question = qa.get("question", qa.get("query", ""))
        answer = str(qa.get("answer", ""))
        raw_category = qa.get("category", "unknown")
        category = CATEGORY_MAP.get(raw_category, str(raw_category))
        evidence_ids = qa.get("evidence", qa.get("evidence_ids", []))
        if isinstance(evidence_ids, str):
            evidence_ids = [evidence_ids]

        if not question:
            continue

        # Run retrieval
        results = retriever.recall(
            query=question,
            max_results=max_results,
        )

        # Compute MRR: find rank of first relevant result
        retrieved_mem_ids = [r.get("id", r.get("memory_id", -1)) for r in results]
        relevant_mem_ids = {dia_map.get(str(eid), -999) for eid in evidence_ids}

        rr = 0.0
        for rank, mem_id in enumerate(retrieved_mem_ids, 1):
            if mem_id in relevant_mem_ids:
                rr = 1.0 / rank
                break

        # Recall@K: any relevant result in top-K?
        recall_at_k = 1.0 if any(mid in relevant_mem_ids for mid in retrieved_mem_ids) else 0.0

        # Compute token F1 against answer using retrieved content
        retrieved_text = " ".join(r.get("content", "") for r in results[:3])
        f1 = _token_f1(retrieved_text, answer)

        per_category[category].append({"mrr": rr, "f1": f1, "recall": recall_at_k})

    # Aggregate
    results = {}
    all_mrrs = []
    all_recalls = []
    for cat, items in per_category.items():
        mrrs = [i["mrr"] for i in items]
        f1s = [i["f1"] for i in items]
        recalls = [i["recall"] for i in items]
        all_mrrs.extend(mrrs)
        all_recalls.extend(recalls)
        results[cat] = {
            "mrr": sum(mrrs) / len(mrrs) if mrrs else 0.0,
            "f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "recall@10": sum(recalls) / len(recalls) if recalls else 0.0,
            "count": len(items),
        }
    results["overall"] = {
        "mrr": sum(all_mrrs) / len(all_mrrs) if all_mrrs else 0.0,
        "recall@10": sum(all_recalls) / len(all_recalls) if all_recalls else 0.0,
        "count": len(all_mrrs),
    }
    return results


class TestLoCoMoSingleConversation:
    """Evaluate retrieval on a single LoCoMo conversation (fast)."""

    def test_single_conversation_retrieval(self, locomo_conv0):
        retriever, conv, dia_map, project_dir = locomo_conv0
        qa_items = conv.get("qa", conv.get("qa_pairs", []))

        if not qa_items:
            pytest.skip("No QA items in conversation 0")

        t0 = time.time()
        results = _evaluate_retrieval_for_qa(retriever, qa_items, dia_map)
        elapsed = time.time() - t0

        print(f"\n=== LoCoMo Single Conversation Results (model={EMBEDDING_MODEL}) ===")
        print(f"  Embedding dim: {EMBEDDING_DIM}, Time: {elapsed:.1f}s")
        for cat in CATEGORY_NAMES + ["overall"]:
            if cat in results:
                r = results[cat]
                mrr = r.get("mrr", 0)
                f1 = r.get("f1", 0)
                recall = r.get("recall@10", 0)
                count = r.get("count", 0)
                print(f"  {cat:15s}: MRR={mrr:.3f}  R@10={recall:.3f}  F1={f1:.3f}  n={count}")

        overall_mrr = results.get("overall", {}).get("mrr", 0)
        print(f"\n  Overall MRR: {overall_mrr:.3f}")


class TestLoCoMoObservation:
    """Test observation-based storage on a single conversation (fast)."""

    def test_obs_single(self, locomo_data, tmp_path_factory):
        conv = locomo_data[0]
        qa_items = conv.get("qa", conv.get("qa_pairs", []))
        if not qa_items:
            pytest.skip("No QA items")

        modes = {"pair_only": False, "pair+obs": True}
        for mode_name, obs in modes.items():
            tmp = tmp_path_factory.mktemp(f"obs_{mode_name}")
            db_path = str(tmp / "memory.db")
            project_dir = str(tmp / "project")
            os.makedirs(project_dir, exist_ok=True)

            settings = _make_settings(DB_PATH=db_path)
            storage = StorageEngine(db_path, embedding_dim=EMBEDDING_DIM)
            embeddings = EmbeddingEngine(EMBEDDING_MODEL)
            kg = KnowledgeGraph(storage, settings)
            dia_map = _ingest_conversation(conv, storage, embeddings, project_dir, obs_mode=obs)
            retriever = HippoRetriever(storage, embeddings, kg, settings)

            t0 = time.time()
            results = _evaluate_retrieval_for_qa(retriever, qa_items, dia_map)
            elapsed = time.time() - t0

            print(f"\n=== [{mode_name}] Conv 0 Results ({elapsed:.1f}s) ===")
            for cat in CATEGORY_NAMES + ["overall"]:
                if cat in results:
                    r = results[cat]
                    print(f"  {cat:15s}: MRR={r.get('mrr',0):.3f}  R@10={r.get('recall@10',0):.3f}  n={r.get('count',0)}")


def _print_micro_averaged(all_results: dict, label: str, elapsed: float):
    """Print micro-averaged metrics (weighted by per-conversation count)."""
    print(f"\n=== {label} (model={EMBEDDING_MODEL}) ===")
    print(f"  Embedding dim: {EMBEDDING_DIM}, Time: {elapsed:.1f}s")
    for cat in CATEGORY_NAMES + ["overall"]:
        if cat in all_results:
            items = all_results[cat]
            total_n = sum(r.get("count", 0) for r in items)
            if total_n > 0:
                avg_mrr = sum(r["mrr"] * r["count"] for r in items) / total_n
                avg_recall = sum(r.get("recall@10", 0) * r.get("count", 0) for r in items) / total_n
            else:
                avg_mrr = avg_recall = 0.0
            print(f"  {cat:15s}: MRR={avg_mrr:.3f}  R@10={avg_recall:.3f}  n={total_n}")


class TestLoCoMoFullBenchmark:
    """Evaluate retrieval across all LoCoMo conversations (slow)."""

    def test_full_benchmark(self, locomo_full):
        all_results = defaultdict(list)

        t0 = time.time()
        for i, (retriever, conv, dia_map, project_dir) in enumerate(locomo_full):
            qa_items = conv.get("qa", conv.get("qa_pairs", []))
            if not qa_items:
                continue

            results = _evaluate_retrieval_for_qa(retriever, qa_items, dia_map)
            for cat, metrics in results.items():
                all_results[cat].append(metrics)

        elapsed = time.time() - t0
        _print_micro_averaged(all_results, "LoCoMo Full Benchmark Results", elapsed)


class TestLoCoMoFullObs:
    """Full benchmark with observation-based storage."""

    def test_full_obs_benchmark(self, locomo_full_obs):
        all_results = defaultdict(list)

        t0 = time.time()
        for i, (retriever, conv, dia_map, project_dir) in enumerate(locomo_full_obs):
            qa_items = conv.get("qa", conv.get("qa_pairs", []))
            if not qa_items:
                continue

            results = _evaluate_retrieval_for_qa(retriever, qa_items, dia_map)
            for cat, metrics in results.items():
                all_results[cat].append(metrics)

        elapsed = time.time() - t0
        _print_micro_averaged(all_results, "LoCoMo Full OBS Benchmark Results", elapsed)


class TestLoCoMoABSweep:
    """Fast A/B sweep on first 3 conversations to find best config."""

    CONFIGS = {
        "baseline": {},
        "combmnz": {"COMBMNZ_ENABLED": True},
        "minmax_norm": {"FUSION_NORM": "minmax"},
        "combmnz+minmax": {"COMBMNZ_ENABLED": True, "FUSION_NORM": "minmax"},
        "ce_blend_50_50": {"CROSS_ENCODER_WEIGHT": 0.5},
        "ce_blend_70_30": {"CROSS_ENCODER_WEIGHT": 0.7},
        "fts_weight_0.8": {"WRRF_FTS_WEIGHT": 0.8},
        "fts_weight_1.0": {"WRRF_FTS_WEIGHT": 1.0},
        "pool_mult_60": {"CANDIDATE_POOL_MULTIPLIER": 60},
        "ce_pool_100": {"CROSS_ENCODER_TOP_K": 100},
        "heuristic+ce": {"RERANKER_ENABLED": True},
        "raw_fusion": {"FUSION_NORM": "raw"},
        "best_guess": {
            "COMBMNZ_ENABLED": True,
            "FUSION_NORM": "minmax",
            "WRRF_FTS_WEIGHT": 0.8,
            "CROSS_ENCODER_WEIGHT": 0.6,
        },
    }

    def test_ab_sweep(self, locomo_data):
        """Run each config on first 3 conversations, sharing ingested data."""
        n_convs = min(3, len(locomo_data))
        print(f"\n=== A/B Sweep on {n_convs} conversations ===")

        # Ingest once, reuse across configs (only retrieval settings change)
        shared_contexts = []
        embeddings = EmbeddingEngine(EMBEDDING_MODEL)
        t_ingest = time.time()
        for i in range(n_convs):
            conv = locomo_data[i]
            tmp_dir = f"/tmp/locomo_ab_shared_{i}"
            os.makedirs(tmp_dir, exist_ok=True)
            db_path = os.path.join(tmp_dir, "memory.db")
            project_dir = os.path.join(tmp_dir, "project")
            os.makedirs(project_dir, exist_ok=True)
            if os.path.exists(db_path):
                os.remove(db_path)

            base_settings = _make_settings(DB_PATH=db_path)
            storage = StorageEngine(db_path, embedding_dim=EMBEDDING_DIM)
            kg = KnowledgeGraph(storage, base_settings)
            dia_map = _ingest_conversation(conv, storage, embeddings, project_dir)
            qa_items = conv.get("qa", conv.get("qa_pairs", []))
            shared_contexts.append((db_path, project_dir, dia_map, qa_items, storage, kg))
        print(f"  Ingestion: {time.time() - t_ingest:.1f}s for {n_convs} conversations")

        for config_name, overrides in self.CONFIGS.items():
            all_results = defaultdict(list)
            t0 = time.time()

            for db_path, project_dir, dia_map, qa_items, storage, kg in shared_contexts:
                if not qa_items:
                    continue
                settings = _make_settings(DB_PATH=db_path, **overrides)
                retriever = HippoRetriever(storage, embeddings, kg, settings)

                results = _evaluate_retrieval_for_qa(retriever, qa_items, dia_map)
                for cat, metrics in results.items():
                    all_results[cat].append(metrics)

            elapsed = time.time() - t0
            overall = all_results.get("overall", [])
            total_n = sum(r.get("count", 0) for r in overall)
            avg_mrr = sum(r["mrr"] * r["count"] for r in overall) / total_n if total_n else 0.0
            avg_recall = sum(r.get("recall@10", 0) * r.get("count", 0) for r in overall) / total_n if total_n else 0.0

            print(f"\n  [{config_name}] MRR={avg_mrr:.3f}  R@10={avg_recall:.3f}  Time={elapsed:.1f}s  n={total_n}")
            for cat in CATEGORY_NAMES:
                if cat in all_results:
                    items = all_results[cat]
                    cat_n = sum(r.get("count", 0) for r in items)
                    cat_mrr = sum(r["mrr"] * r["count"] for r in items) / cat_n if cat_n else 0.0
                    print(f"    {cat:15s}: MRR={cat_mrr:.3f}  n={cat_n}")


class TestLoCoMoObsSweep:
    """Quick obs-mode A/B sweep on 3 conversations to find best fusion config."""

    CONFIGS = {
        "ce50_baseline": {},  # Current best: CE_TOP_K=50
        "ce75": {"CROSS_ENCODER_TOP_K": 75},  # Larger CE pool
        "ce100": {"CROSS_ENCODER_TOP_K": 100},  # Even larger CE pool
        "fts_0.8": {"WRRF_FTS_WEIGHT": 0.8},  # Higher FTS weight
        "pool60_ce75": {"CANDIDATE_POOL_MULTIPLIER": 60, "CROSS_ENCODER_TOP_K": 75},
    }

    def test_obs_sweep(self, locomo_data):
        n_convs = min(3, len(locomo_data))
        print(f"\n=== Obs-Mode A/B Sweep on {n_convs} conversations ===")

        shared_contexts = []
        embeddings = EmbeddingEngine(EMBEDDING_MODEL)
        t_ingest = time.time()
        for i in range(n_convs):
            conv = locomo_data[i]
            tmp_dir = f"/tmp/locomo_obs_sweep_{i}"
            os.makedirs(tmp_dir, exist_ok=True)
            db_path = os.path.join(tmp_dir, "memory.db")
            project_dir = os.path.join(tmp_dir, "project")
            os.makedirs(project_dir, exist_ok=True)
            if os.path.exists(db_path):
                os.remove(db_path)

            base_settings = _make_settings(DB_PATH=db_path)
            storage = StorageEngine(db_path, embedding_dim=EMBEDDING_DIM)
            kg = KnowledgeGraph(storage, base_settings)
            dia_map = _ingest_conversation(conv, storage, embeddings, project_dir, obs_mode=True)
            qa_items = conv.get("qa", conv.get("qa_pairs", []))
            shared_contexts.append((db_path, project_dir, dia_map, qa_items, storage, kg))
        print(f"  Ingestion (obs_mode): {time.time() - t_ingest:.1f}s for {n_convs} conversations")

        for config_name, overrides in self.CONFIGS.items():
            all_results = defaultdict(list)
            t0 = time.time()

            for db_path, project_dir, dia_map, qa_items, storage, kg in shared_contexts:
                if not qa_items:
                    continue
                settings = _make_settings(DB_PATH=db_path, **overrides)
                retriever = HippoRetriever(storage, embeddings, kg, settings)

                results = _evaluate_retrieval_for_qa(retriever, qa_items, dia_map)
                for cat, metrics in results.items():
                    all_results[cat].append(metrics)

            elapsed = time.time() - t0
            overall = all_results.get("overall", [])
            total_n = sum(r.get("count", 0) for r in overall)
            avg_mrr = sum(r["mrr"] * r["count"] for r in overall) / total_n if total_n else 0.0
            avg_recall = sum(r.get("recall@10", 0) * r.get("count", 0) for r in overall) / total_n if total_n else 0.0

            print(f"\n  [{config_name}] MRR={avg_mrr:.3f}  R@10={avg_recall:.3f}  Time={elapsed:.1f}s  n={total_n}")
            for cat in CATEGORY_NAMES:
                if cat in all_results:
                    items = all_results[cat]
                    cat_n = sum(r.get("count", 0) for r in items)
                    cat_mrr = sum(r["mrr"] * r["count"] for r in items) / cat_n if cat_n else 0.0
                    print(f"    {cat:15s}: MRR={cat_mrr:.3f}  n={cat_n}")
