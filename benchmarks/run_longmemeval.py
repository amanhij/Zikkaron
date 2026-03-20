"""LongMemEval benchmark for Zikkaron.

Evaluates long-term interactive memory across 500 questions covering:
  - Information Extraction (single-session user/assistant/preference)
  - Multi-Session Reasoning
  - Temporal Reasoning
  - Knowledge Updates
  - Abstention (false-premise detection)

Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
Paper: Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term
       Interactive Memory" (ICLR 2025, arXiv:2410.10813)

Usage:
  # Phase 1: Retrieval-only (no LLM, fast, free)
  python benchmarks/run_longmemeval.py --retrieval-only

  # Phase 2: Full QA with Claude as reader + judge
  python benchmarks/run_longmemeval.py

  # Subset for quick testing
  python benchmarks/run_longmemeval.py --max-questions 20 --retrieval-only

  # Specific question types only
  python benchmarks/run_longmemeval.py --types temporal-reasoning,knowledge-update
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.retrieval import HippoRetriever
from zikkaron.storage import StorageEngine
from zikkaron.thermodynamics import MemoryThermodynamics
from zikkaron.predictive_coding import PredictiveCodingGate
from zikkaron.curation import MemoryCurator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

DATASET_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
DATASET_DIR = Path(__file__).parent / "data" / "longmemeval"

QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]

# Map question types to the 5 core abilities for reporting
ABILITY_MAP = {
    "single-session-user": "Information Extraction",
    "single-session-assistant": "Information Extraction",
    "single-session-preference": "Information Extraction",
    "multi-session": "Multi-Session Reasoning",
    "temporal-reasoning": "Temporal Reasoning",
    "knowledge-update": "Knowledge Updates",
}


# ── Dataset Download ──────────────────────────────────────────────────

def download_dataset(variant: str = "s") -> Path:
    """Download LongMemEval dataset from HuggingFace if not cached."""
    filename_map = {
        "oracle": "longmemeval_oracle.json",
        "s": "longmemeval_s_cleaned.json",
        "m": "longmemeval_m_cleaned.json",
    }
    filename = filename_map[variant]
    local_path = DATASET_DIR / filename

    if local_path.exists():
        print(f"Dataset cached: {local_path}")
        return local_path

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{DATASET_URL}/{filename}"
    print(f"Downloading {url} ...")

    import urllib.request
    urllib.request.urlretrieve(url, local_path)
    print(f"Saved to {local_path} ({local_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return local_path


def load_dataset(path: Path) -> list[dict]:
    """Load and validate LongMemEval JSON."""
    with open(path) as f:
        data = json.load(f)

    # Normalize answer field to string (32 questions have int answers)
    for q in data:
        q["answer"] = str(q["answer"])

    print(f"Loaded {len(data)} questions")

    # Print distribution
    type_counts = {}
    abs_count = 0
    for q in data:
        qtype = q["question_type"]
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
        if q["question_id"].endswith("_abs"):
            abs_count += 1

    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype}: {count}")
    print(f"  abstention: {abs_count}")

    return data


# ── Zikkaron Engine Factory ──────────────────────────────────────────

def make_benchmark_settings(**overrides) -> Settings:
    """Create Zikkaron settings optimized for LongMemEval retrieval."""
    defaults = {
        # Write gate: disable for benchmark (we want all memories stored)
        "WRITE_GATE_THRESHOLD": 0.0,
        # Retrieval signals
        "CROSS_ENCODER_ENABLED": True,
        "CROSS_ENCODER_TOP_K": 75,
        "CROSS_ENCODER_WEIGHT": 1.0,
        "WRRF_VECTOR_WEIGHT": 1.0,
        "WRRF_FTS_WEIGHT": 0.5,
        # Graph signals: disabled (no real graph in benchmark data)
        "WRRF_PPR_WEIGHT": 0.0,
        "WRRF_SPREADING_WEIGHT": 0.0,
        "WRRF_SR_WEIGHT": 0.0,
        "WRRF_HOPFIELD_WEIGHT": 0.2,
        "WRRF_HDC_WEIGHT": 0.0,
        "WRRF_FRACTAL_WEIGHT": 0.0,
        # Reranking
        "GTE_RERANKER_ENABLED": True,
        "NLI_RERANKING_ENABLED": True,
        "NLI_ONLY_FOR_OPEN_DOMAIN": False,
        "MULTI_PASSAGE_RERANKING_ENABLED": True,
        # Index enrichment
        "INDEX_ENRICHMENT_ENABLED": True,
        "CONCEPTNET_ENRICHMENT_ENABLED": True,
        "LOGIC_ENRICHMENT_ENABLED": True,
        "COMET_ENRICHMENT_ENABLED": False,
        "DOC2QUERY_ENRICHMENT_ENABLED": False,
        # Profiles & beliefs
        "PROFILE_EXTRACTION_ENABLED": True,
        "DERIVED_BELIEFS_ENABLED": True,
        # Comparison routing
        "COMPARISON_DUAL_SEARCH_ENABLED": True,
        # Query expansion
        "QUERY_EXPANSION_ENABLED": True,
        # Temporal retrieval
        "TEMPORAL_RETRIEVAL_ENABLED": True,
        # Disable zero-gap features for benchmark (they're for live use)
        "REINJECTION_ENABLED": False,
        "MICRO_CHECKPOINT_ENABLED": False,
        "ACTION_STREAM_ENABLED": False,
        "DECISION_AUTO_PROTECT": False,
    }
    defaults.update(overrides)

    # Build Settings with env prefix disabled
    os.environ.update({f"ZIKKARON_{k}": str(v) for k, v in defaults.items()})
    return Settings()


def create_engines(db_path: str, settings: Settings):
    """Create a minimal Zikkaron engine set for benchmarking."""
    storage = StorageEngine(db_path)
    embeddings = EmbeddingEngine(settings.EMBEDDING_MODEL)
    kg = KnowledgeGraph(storage, settings)
    thermo = MemoryThermodynamics(storage, embeddings, settings)
    retriever = HippoRetriever(storage, embeddings, kg, settings)
    curator = MemoryCurator(storage, embeddings, thermo, settings)

    return storage, embeddings, retriever, curator, thermo


# ── Ingestion ─────────────────────────────────────────────────────────

def ingest_question_haystack(
    question: dict,
    storage: StorageEngine,
    embeddings: EmbeddingEngine,
    curator: MemoryCurator,
    thermo: MemoryThermodynamics,
    settings: Settings,
) -> dict[str, list[int]]:
    """Ingest a question's haystack sessions into Zikkaron.

    Uses round-level decomposition: each user-assistant turn pair becomes
    a separate memory. This is the optimal granularity per the paper
    (+11.3% recall vs session-level).

    Embeds temporal metadata in content for temporal reasoning queries.

    Returns: mapping of session_id -> list of memory_ids (for retrieval eval)
    """
    session_map: dict[str, list[int]] = {}
    sessions = question["haystack_sessions"]
    session_ids = question["haystack_session_ids"]
    session_dates = question["haystack_dates"]

    for idx, (session, session_id, session_date) in enumerate(
        zip(sessions, session_ids, session_dates)
    ):
        memory_ids = []

        # Decompose session into rounds (user-assistant turn pairs)
        rounds = []
        i = 0
        while i < len(session):
            user_msg = None
            asst_msg = None

            if session[i]["role"] == "user":
                user_msg = session[i]["content"]
                if i + 1 < len(session) and session[i + 1]["role"] == "assistant":
                    asst_msg = session[i + 1]["content"]
                    i += 2
                else:
                    i += 1
            elif session[i]["role"] == "assistant":
                asst_msg = session[i]["content"]
                i += 1
            else:
                i += 1
                continue

            rounds.append((user_msg, asst_msg))

        # Store each round as a memory with temporal metadata
        for round_idx, (user_text, asst_text) in enumerate(rounds):
            # Build content with temporal context embedded
            parts = [f"[Date: {session_date}]"]
            if user_text:
                parts.append(f"User: {user_text}")
            if asst_text:
                parts.append(f"Assistant: {asst_text}")
            content = "\n".join(parts)

            # Tags for retrieval evaluation mapping
            tags = [
                f"session:{session_id}",
                f"date:{session_date}",
                f"round:{round_idx}",
            ]

            # Embed and store
            embedding = embeddings.encode(content)
            if embedding is None:
                continue

            memory_id = storage.insert_memory({
                "content": content,
                "embedding": embedding,
                "tags": tags,
                "directory_context": "/benchmark/longmemeval",
                "heat": 1.0,
                "is_stale": False,
                "file_hash": None,
                "embedding_model": embeddings.get_model_name(),
            })

            # Set importance and surprise scores
            importance = thermo.compute_importance(content, tags)
            storage.update_memory_scores(
                memory_id,
                surprise_score=0.5,
                importance=importance,
                emotional_valence=0.0,
            )

            memory_ids.append(memory_id)

        session_map[session_id] = memory_ids

    return session_map


# ── Retrieval Evaluation ──────────────────────────────────────────────

def compute_ndcg(retrieved_session_ids: list[str], gold_session_ids: set[str], k: int) -> float:
    """Compute NDCG@k with binary relevance."""
    dcg = 0.0
    for i, sid in enumerate(retrieved_session_ids[:k]):
        if sid in gold_session_ids:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG: all relevant items at top
    ideal_count = min(len(gold_session_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_recall(retrieved_session_ids: list[str], gold_session_ids: set[str], k: int) -> float:
    """Compute Recall@k: fraction of gold sessions found in top-k."""
    if not gold_session_ids:
        return 0.0
    found = sum(1 for sid in retrieved_session_ids[:k] if sid in gold_session_ids)
    return found / len(gold_session_ids)


def evaluate_retrieval(
    question: dict,
    retriever: HippoRetriever,
    session_map: dict[str, list[int]],
    max_results: int = 50,
) -> dict:
    """Run retrieval and compute session-level metrics.

    Returns dict with recall@k and ndcg@k for k in {5, 10, 50}.
    """
    query = question["question"]
    gold_session_ids = set(question["answer_session_ids"])
    is_abstention = question["question_id"].endswith("_abs")

    # Skip retrieval metrics for abstention questions (no ground truth location)
    if is_abstention:
        return {"skipped": True, "reason": "abstention"}

    # Run retrieval (catch FTS5 syntax errors from apostrophes etc.)
    try:
        results = retriever.recall(query, max_results=max_results, min_heat=0.0)
    except Exception as e:
        logger.warning("Retrieval failed for question %s: %s", question["question_id"], e)
        results = []

    # Build reverse map: memory_id -> session_id
    mid_to_session: dict[int, str] = {}
    for session_id, memory_ids in session_map.items():
        for mid in memory_ids:
            mid_to_session[mid] = session_id

    # Map retrieved memories to session IDs (deduplicated, preserving order)
    retrieved_sessions = []
    seen = set()
    for mem in results:
        sid = mid_to_session.get(mem["id"])
        if sid and sid not in seen:
            retrieved_sessions.append(sid)
            seen.add(sid)

    # Compute metrics at multiple k values
    metrics = {}
    for k in [5, 10, 50]:
        metrics[f"recall@{k}"] = compute_recall(retrieved_sessions, gold_session_ids, k)
        metrics[f"ndcg@{k}"] = compute_ndcg(retrieved_sessions, gold_session_ids, k)

    # MRR: reciprocal rank of first relevant session
    mrr = 0.0
    for i, sid in enumerate(retrieved_sessions):
        if sid in gold_session_ids:
            mrr = 1.0 / (i + 1)
            break
    metrics["mrr"] = mrr

    # Hit rank
    hit_rank = None
    for i, sid in enumerate(retrieved_sessions):
        if sid in gold_session_ids:
            hit_rank = i + 1
            break
    metrics["hit_rank"] = hit_rank

    metrics["retrieved_sessions"] = len(retrieved_sessions)
    metrics["gold_sessions"] = len(gold_session_ids)

    return metrics


# ── Answer Generation (claude -p) ────────────────────────────────────

READER_SYSTEM_PROMPT = """You are answering questions about a user's conversation history.
You will be given relevant excerpts from past conversations and a question.
Answer the question based ONLY on the provided context.
If the context does not contain enough information to answer, say "I don't have enough information to answer this question."
Be concise and specific. Give the most direct answer possible."""

READER_PROMPT_TEMPLATE = """## Relevant conversation history:
{context}

## Question (asked on {question_date}):
{question}

## Answer:"""

JUDGE_SYSTEM_PROMPT = """You are evaluating whether a system's answer to a question is correct.
You will be given the question, the gold (correct) answer, and the system's answer.
Determine if the system's answer is correct.

Rules:
- The system answer does NOT need to match the gold answer word-for-word
- It IS correct if it conveys the same core information
- For temporal questions, accept minor date format differences
- For knowledge-update questions, the answer must reflect the LATEST known information
- For preference questions, the answer must capture the user's preference accurately
- If the question is unanswerable and the system correctly identifies this, mark as correct

Respond with ONLY a JSON object: {"correct": true} or {"correct": false}"""

JUDGE_PROMPT_TEMPLATE = """Question: {question}
Gold answer: {gold_answer}
System answer: {hypothesis}"""


def call_claude_pipe(prompt: str, system_prompt: str = "", timeout: int = 120) -> str:
    """Call Claude via `claude -p` pipe mode.

    Returns the generated text. Falls back to empty string on error.
    """
    cmd = ["claude", "-p", "--output-format", "json"]
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            logger.warning("claude -p failed: %s", result.stderr[:200])
            return ""

        response = json.loads(result.stdout)
        return response.get("result", "")
    except subprocess.TimeoutExpired:
        logger.warning("claude -p timed out after %ds", timeout)
        return ""
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("claude -p error: %s", e)
        return ""


def generate_answer(
    question: dict,
    retrieved_memories: list[dict],
    top_k_context: int = 10,
) -> str:
    """Generate an answer using Claude as the reader LLM."""
    # Format retrieved context
    context_parts = []
    for i, mem in enumerate(retrieved_memories[:top_k_context]):
        content = mem.get("content", "")
        context_parts.append(f"[{i+1}] {content}")
    context = "\n\n".join(context_parts)

    prompt = READER_PROMPT_TEMPLATE.format(
        context=context,
        question=question["question"],
        question_date=question["question_date"],
    )

    return call_claude_pipe(prompt, READER_SYSTEM_PROMPT)


def judge_answer(question: dict, hypothesis: str) -> dict:
    """Judge whether the generated answer is correct using Claude."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question["question"],
        gold_answer=question["answer"],
        hypothesis=hypothesis,
    )

    response = call_claude_pipe(prompt, JUDGE_SYSTEM_PROMPT)

    # Parse judge response
    try:
        # Try to extract JSON from response
        if "{" in response:
            json_str = response[response.index("{"):response.rindex("}") + 1]
            result = json.loads(json_str)
            return {"correct": bool(result.get("correct", False)), "raw": response}
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: look for yes/true/correct in response
    lower = response.lower()
    correct = any(w in lower for w in ["correct", '"correct": true', "yes"])
    return {"correct": correct, "raw": response}


# ── Main Benchmark Pipeline ──────────────────────────────────────────

def run_benchmark(
    dataset_path: Path,
    retrieval_only: bool = False,
    max_questions: int = 0,
    question_types: list[str] | None = None,
    max_results: int = 50,
    top_k_context: int = 10,
    settings_overrides: dict | None = None,
    output_path: str | None = None,
) -> dict:
    """Run the full LongMemEval benchmark.

    Phase 1: Retrieval evaluation (always runs)
    Phase 2: Answer generation + judging (unless retrieval_only=True)
    """
    data = load_dataset(dataset_path)

    # Filter by question types if specified
    if question_types:
        data = [q for q in data if q["question_type"] in question_types]
        print(f"Filtered to {len(data)} questions of types: {question_types}")

    # Limit for quick testing
    if max_questions > 0:
        data = data[:max_questions]
        print(f"Limited to {max_questions} questions")

    settings = make_benchmark_settings(**(settings_overrides or {}))

    # Initialize embedding engine once (shared across all questions)
    print("Loading embedding model...")
    embeddings = EmbeddingEngine(settings.EMBEDDING_MODEL)

    results = {
        "benchmark": "LongMemEval",
        "variant": dataset_path.stem,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(data),
        "retrieval_only": retrieval_only,
        "max_results": max_results,
        "top_k_context": top_k_context,
        "settings_overrides": settings_overrides or {},
        "per_query": [],
        "aggregated": {},
    }

    start_time = time.monotonic()

    for qi, question in enumerate(data):
        qid = question["question_id"]
        qtype = question["question_type"]
        is_abs = qid.endswith("_abs")

        print(f"\r[{qi+1}/{len(data)}] {qtype}: {question['question'][:60]}...", end="", flush=True)

        # Create fresh DB for this question (no cross-contamination)
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "bench.db")

            storage = StorageEngine(db_path)
            kg = KnowledgeGraph(storage, settings)
            thermo = MemoryThermodynamics(storage, embeddings, settings)
            retriever = HippoRetriever(storage, embeddings, kg, settings)
            curator = MemoryCurator(storage, embeddings, thermo, settings)

            # Phase 1a: Ingest haystack
            t_ingest = time.monotonic()
            session_map = ingest_question_haystack(
                question, storage, embeddings, curator, thermo, settings
            )
            ingest_time = time.monotonic() - t_ingest

            total_memories = sum(len(mids) for mids in session_map.values())

            # Phase 1b: Retrieval evaluation
            t_retrieve = time.monotonic()
            retrieval_metrics = evaluate_retrieval(
                question, retriever, session_map, max_results=max_results
            )
            retrieve_time = time.monotonic() - t_retrieve

            # Phase 2: Answer generation + judging
            hypothesis = ""
            judge_result = {}
            gen_time = 0.0
            judge_time = 0.0

            if not retrieval_only:
                # Get retrieved memories for answer generation
                try:
                    retrieved = retriever.recall(
                        question["question"], max_results=max_results, min_heat=0.0
                    )
                except Exception:
                    retrieved = []

                t_gen = time.monotonic()
                hypothesis = generate_answer(question, retrieved, top_k_context)
                gen_time = time.monotonic() - t_gen

                if hypothesis:
                    t_judge = time.monotonic()
                    judge_result = judge_answer(question, hypothesis)
                    judge_time = time.monotonic() - t_judge

            # Record per-query result
            query_result = {
                "question_id": qid,
                "question_type": qtype,
                "is_abstention": is_abs,
                "question": question["question"],
                "gold_answer": question["answer"],
                "sessions_in_haystack": len(question["haystack_session_ids"]),
                "memories_ingested": total_memories,
                "ingest_seconds": round(ingest_time, 2),
                "retrieve_seconds": round(retrieve_time, 2),
                **{k: v for k, v in retrieval_metrics.items() if k != "skipped"},
            }

            if not retrieval_only:
                query_result["hypothesis"] = hypothesis
                query_result["correct"] = judge_result.get("correct", False)
                query_result["gen_seconds"] = round(gen_time, 2)
                query_result["judge_seconds"] = round(judge_time, 2)

            results["per_query"].append(query_result)
            storage.close()

    print()  # newline after progress

    # ── Aggregate metrics ─────────────────────────────────────────────

    elapsed = time.monotonic() - start_time
    results["elapsed_seconds"] = round(elapsed, 1)

    # Group by question type
    by_type: dict[str, list[dict]] = {}
    for qr in results["per_query"]:
        qtype = qr["question_type"]
        if qtype not in by_type:
            by_type[qtype] = []
        by_type[qtype].append(qr)

    agg = {}
    for qtype, queries in sorted(by_type.items()):
        # Retrieval metrics (skip abstention)
        retrieval_queries = [q for q in queries if not q.get("is_abstention")]
        type_agg = {"count": len(queries)}

        if retrieval_queries:
            for metric in ["recall@5", "recall@10", "recall@50", "ndcg@5", "ndcg@10", "ndcg@50", "mrr"]:
                vals = [q.get(metric, 0) for q in retrieval_queries if metric in q]
                if vals:
                    type_agg[metric] = round(sum(vals) / len(vals), 4)

        # QA accuracy (if available)
        if not retrieval_only:
            correct_count = sum(1 for q in queries if q.get("correct", False))
            type_agg["qa_accuracy"] = round(correct_count / len(queries), 4) if queries else 0
            type_agg["qa_correct"] = correct_count
            type_agg["qa_total"] = len(queries)

        agg[qtype] = type_agg

    # Overall
    all_retrieval = [q for q in results["per_query"] if not q.get("is_abstention") and "mrr" in q]
    overall = {"count": len(results["per_query"])}
    if all_retrieval:
        for metric in ["recall@5", "recall@10", "recall@50", "ndcg@5", "ndcg@10", "ndcg@50", "mrr"]:
            vals = [q.get(metric, 0) for q in all_retrieval]
            if vals:
                overall[metric] = round(sum(vals) / len(vals), 4)

    if not retrieval_only:
        all_correct = sum(1 for q in results["per_query"] if q.get("correct", False))
        overall["qa_accuracy"] = round(all_correct / len(results["per_query"]), 4)
        overall["qa_correct"] = all_correct
        overall["qa_total"] = len(results["per_query"])

    agg["overall"] = overall

    # Abstention accuracy (separate)
    abs_queries = [q for q in results["per_query"] if q.get("is_abstention")]
    if abs_queries and not retrieval_only:
        abs_correct = sum(1 for q in abs_queries if q.get("correct", False))
        agg["abstention"] = {
            "count": len(abs_queries),
            "qa_accuracy": round(abs_correct / len(abs_queries), 4),
            "qa_correct": abs_correct,
        }

    results["aggregated"] = agg

    # ── Output ────────────────────────────────────────────────────────

    # Print summary table
    print("\n" + "=" * 80)
    print("LongMemEval Results")
    print("=" * 80)

    header = f"{'Type':<30} {'Count':>5} {'MRR':>7} {'R@5':>7} {'R@10':>7} {'NDCG@10':>7}"
    if not retrieval_only:
        header += f" {'QA Acc':>7}"
    print(header)
    print("-" * len(header))

    for qtype in QUESTION_TYPES + ["overall"]:
        if qtype not in agg:
            continue
        a = agg[qtype]
        line = f"{qtype:<30} {a['count']:>5} {a.get('mrr', 0):>7.3f} {a.get('recall@5', 0):>7.3f} {a.get('recall@10', 0):>7.3f} {a.get('ndcg@10', 0):>7.3f}"
        if not retrieval_only:
            line += f" {a.get('qa_accuracy', 0):>7.1%}"
        print(line)

    if "abstention" in agg and not retrieval_only:
        a = agg["abstention"]
        print(f"{'abstention':<30} {a['count']:>5} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>7} {a.get('qa_accuracy', 0):>7.1%}")

    print(f"\nElapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save JSON
    if output_path is None:
        variant = dataset_path.stem.replace("longmemeval_", "").replace("_cleaned", "")
        mode = "retrieval" if retrieval_only else "full"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(DATASET_DIR.parent / f"longmemeval_{variant}_{mode}_{ts}.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {output_path}")

    # Also save JSONL hypothesis file (for external eval scripts)
    if not retrieval_only:
        hyp_path = output_path.replace(".json", "_hypotheses.jsonl")
        with open(hyp_path, "w") as f:
            for qr in results["per_query"]:
                f.write(json.dumps({
                    "question_id": qr["question_id"],
                    "hypothesis": qr.get("hypothesis", ""),
                }) + "\n")
        print(f"Hypotheses JSONL: {hyp_path}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run LongMemEval benchmark against Zikkaron"
    )
    parser.add_argument(
        "--variant", choices=["oracle", "s", "m"], default="s",
        help="Dataset variant: oracle (evidence only), s (~40 sessions), m (~500 sessions)",
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Only compute retrieval metrics (no LLM calls, fast)",
    )
    parser.add_argument(
        "--max-questions", type=int, default=0,
        help="Limit number of questions (0 = all)",
    )
    parser.add_argument(
        "--types", type=str, default="",
        help="Comma-separated question types to evaluate",
    )
    parser.add_argument(
        "--max-results", type=int, default=50,
        help="Max memories to retrieve per question",
    )
    parser.add_argument(
        "--top-k-context", type=int, default=10,
        help="Top-k retrieved memories to include in reader prompt",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path",
    )

    args = parser.parse_args()

    dataset_path = download_dataset(args.variant)

    question_types = None
    if args.types:
        question_types = [t.strip() for t in args.types.split(",")]

    run_benchmark(
        dataset_path=dataset_path,
        retrieval_only=args.retrieval_only,
        max_questions=args.max_questions,
        question_types=question_types,
        max_results=args.max_results,
        top_k_context=args.top_k_context,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
