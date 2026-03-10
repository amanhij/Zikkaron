#!/usr/bin/env python3
"""GPU-accelerated LoCoMo benchmark runner with extreme step-by-step logging.

Usage:
  CUDA_VISIBLE_DEVICES=0 python benchmarks/run_benchmark_gpu.py
  CUDA_VISIBLE_DEVICES=0 python benchmarks/run_benchmark_gpu.py \
    --label baseline --overrides-json '{"INDEX_ENRICHMENT_ENABLED": false}'
"""
import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

# Force GPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.retrieval import HippoRetriever
from zikkaron.storage import StorageEngine

# Import benchmark helpers
from benchmarks.test_e_locomo import (
    CATEGORY_MAP,
    CATEGORY_NAMES,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    LOCOMO_JSON_PATH,
    _ingest_conversation,
    _make_settings,
    _normalize_answer,
    _token_f1,
)

# ─── Logging ───────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)

def log_gpu():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e6
        resrv = torch.cuda.memory_reserved() / 1e6
        log(f"  GPU mem: {alloc:.0f}MB allocated, {resrv:.0f}MB reserved")

def log_ram():
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB on Linux
    log(f"  RAM RSS: {rss:.0f}MB")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="default", help="Human-readable label for this run.")
    parser.add_argument(
        "--overrides-json",
        default="",
        help="Inline JSON object or path to a JSON file containing Settings overrides.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path for structured benchmark results.",
    )
    parser.add_argument(
        "--conversation-indexes",
        default="",
        help="Comma-separated zero-based conversation indexes to run. Default: all conversations.",
    )
    return parser.parse_args()


def _load_overrides(raw_value: str) -> dict:
    if not raw_value:
        return {}
    if os.path.exists(raw_value):
        with open(raw_value) as f:
            return json.load(f)
    return json.loads(raw_value)


def _parse_conversation_indexes(raw_value: str) -> list[int] | None:
    if not raw_value:
        return None
    indexes = []
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        indexes.append(int(part))
    return indexes


def _aggregate_metrics(all_results: dict[str, list[dict]]) -> dict[str, dict]:
    aggregated = {}
    for cat in CATEGORY_NAMES + ["overall"]:
        if cat not in all_results:
            continue
        items = all_results[cat]
        total_n = sum(r.get("count", 0) for r in items)
        if total_n > 0:
            avg_mrr = sum(r["mrr"] * r["count"] for r in items) / total_n
            avg_recall = sum(r.get("recall@10", 0) * r.get("count", 0) for r in items) / total_n
            avg_f1 = sum(r.get("f1", 0) * r.get("count", 0) for r in items) / total_n
        else:
            avg_mrr = avg_recall = avg_f1 = 0.0
        aggregated[cat] = {
            "mrr": avg_mrr,
            "recall@10": avg_recall,
            "f1": avg_f1,
            "count": total_n,
        }
    return aggregated


def run_benchmark(
    *,
    label: str = "default",
    overrides: dict | None = None,
    conversation_indexes: list[int] | None = None,
) -> dict:
    overrides = overrides or {}

    # ─── Setup ─────────────────────────────────────────────────────────────
    log("=" * 70)
    log(f"LoCoMo FULL BENCHMARK — GPU Mode — Extreme Logging — {label}")
    log("=" * 70)

    # Verify GPU
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"GPU: {dev} ({vram:.1f}GB VRAM)")
    else:
        log("WARNING: No GPU detected! Running on CPU.")

    # Check ONNX providers
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        log(f"ONNX providers: {providers}")
    except ImportError:
        log("WARNING: onnxruntime not available")

    log(f"Embedding model: {EMBEDDING_MODEL} ({EMBEDDING_DIM}d)")
    log(f"Data path: {LOCOMO_JSON_PATH}")
    if overrides:
        log(f"Overrides: {json.dumps(overrides, sort_keys=True)}")

    # ─── Load data ─────────────────────────────────────────────────────────
    log("Loading LoCoMo dataset...")
    t0 = time.time()
    with open(LOCOMO_JSON_PATH) as f:
        locomo_data = json.load(f)
    log(f"Loaded {len(locomo_data)} conversations in {time.time()-t0:.1f}s")

    # Count total QA
    total_qa = sum(len(c.get("qa", c.get("qa_pairs", []))) for c in locomo_data)
    log(f"Total QA pairs across all conversations: {total_qa}")

    if conversation_indexes is not None:
        selected = [(idx, locomo_data[idx]) for idx in conversation_indexes]
        log(f"Selected conversations: {conversation_indexes}")
    else:
        selected = list(enumerate(locomo_data))

    # ─── Per-conversation benchmark ───────────────────────────────────────
    all_results = defaultdict(list)
    query_results = []
    conversation_results = []
    grand_t0 = time.time()

    # Share a single embedding engine across conversations
    embeddings = EmbeddingEngine(EMBEDDING_MODEL)
    log("Warming up embedding model...")
    _ = embeddings.encode_query("test warmup query")
    log_gpu()
    log_ram()

    for run_idx, (conv_idx, conv) in enumerate(selected):
        conv_t0 = time.time()
        qa_items = conv.get("qa", conv.get("qa_pairs", []))
        n_qa = len(qa_items)

        log("")
        log(f"{'='*60}")
        log(f"CONVERSATION {run_idx+1}/{len(selected)} (source={conv_idx}) — {n_qa} QA pairs")
        log(f"{'='*60}")

        if not qa_items:
            log("  SKIP: No QA items")
            continue

        cat_counts = defaultdict(int)
        for qa in qa_items:
            cat = CATEGORY_MAP.get(qa.get("category", "?"), str(qa.get("category", "?")))
            cat_counts[cat] += 1
        log(f"  Categories: {dict(cat_counts)}")

        tmp_dir = f"/tmp/locomo_bench_gpu_{label}_{conv_idx}"
        os.makedirs(tmp_dir, exist_ok=True)
        db_path = os.path.join(tmp_dir, "memory.db")
        project_dir = os.path.join(tmp_dir, "project")
        os.makedirs(project_dir, exist_ok=True)

        if os.path.exists(db_path):
            os.remove(db_path)

        settings = _make_settings(DB_PATH=db_path, **overrides)
        storage = StorageEngine(db_path, embedding_dim=EMBEDDING_DIM)
        kg = KnowledgeGraph(storage, settings)

        log(
            "  [INGEST] Starting ingestion "
            f"(obs_mode=True, enrichment={settings.INDEX_ENRICHMENT_ENABLED})..."
        )
        ingest_t0 = time.time()
        dia_map = _ingest_conversation(
            conv,
            storage,
            embeddings,
            project_dir,
            obs_mode=True,
            settings=settings,
        )
        ingest_elapsed = time.time() - ingest_t0
        log(f"  [INGEST] Done — {len(dia_map)} dialogue IDs mapped in {ingest_elapsed:.1f}s")
        log_ram()

        retriever = HippoRetriever(storage, embeddings, kg, settings)

        per_category = defaultdict(list)
        eval_t0 = time.time()

        for qi, qa in enumerate(qa_items):
            question = qa.get("question", qa.get("query", ""))
            answer = str(qa.get("answer", ""))
            raw_category = qa.get("category", "unknown")
            category = CATEGORY_MAP.get(raw_category, str(raw_category))
            evidence_ids = qa.get("evidence", qa.get("evidence_ids", []))
            if isinstance(evidence_ids, str):
                evidence_ids = [evidence_ids]

            if not question:
                continue

            q_t0 = time.time()
            results = retriever.recall(query=question, max_results=10)
            q_elapsed = time.time() - q_t0

            retrieved_mem_ids = [r.get("id", r.get("memory_id", -1)) for r in results]
            relevant_mem_ids = {dia_map.get(str(eid), -999) for eid in evidence_ids}

            rr = 0.0
            hit_rank = 0
            for rank, mem_id in enumerate(retrieved_mem_ids, 1):
                if mem_id in relevant_mem_ids:
                    rr = 1.0 / rank
                    hit_rank = rank
                    break

            recall_at_k = 1.0 if any(mid in relevant_mem_ids for mid in retrieved_mem_ids) else 0.0
            retrieved_text = " ".join(r.get("content", "") for r in results[:3])
            f1 = _token_f1(retrieved_text, answer)

            sample = {
                "conversation_index": conv_idx,
                "query_index": qi,
                "category": category,
                "question": question,
                "answer": answer,
                "mrr": rr,
                "recall@10": recall_at_k,
                "f1": f1,
                "hit_rank": hit_rank,
            }
            per_category[category].append(sample)
            query_results.append(sample)

            if (qi + 1) % 25 == 0 or qi == n_qa - 1:
                elapsed_so_far = time.time() - eval_t0
                rate = (qi + 1) / elapsed_so_far if elapsed_so_far > 0 else 0
                remaining = (n_qa - qi - 1) / rate if rate > 0 else 0
                log(
                    f"  [EVAL] Query {qi+1}/{n_qa} ({category}) — {q_elapsed:.2f}s — "
                    f"MRR={rr:.3f} — Rate: {rate:.1f}q/s — ETA: {remaining:.0f}s"
                )
            elif category == "open_domain" and rr == 0:
                log(
                    f"  [EVAL] Query {qi+1}/{n_qa} MISS ({category}) — "
                    f"Q: {question[:80]}... — MRR=0.000 — {q_elapsed:.2f}s"
                )

        eval_elapsed = time.time() - eval_t0

        conv_results = {}
        all_mrrs = []
        all_recalls = []
        all_f1s = []
        for cat, items in per_category.items():
            mrrs = [i["mrr"] for i in items]
            f1s = [i["f1"] for i in items]
            recalls = [i["recall@10"] for i in items]
            all_mrrs.extend(mrrs)
            all_recalls.extend(recalls)
            all_f1s.extend(f1s)
            conv_results[cat] = {
                "mrr": sum(mrrs) / len(mrrs) if mrrs else 0.0,
                "f1": sum(f1s) / len(f1s) if f1s else 0.0,
                "recall@10": sum(recalls) / len(recalls) if recalls else 0.0,
                "count": len(items),
            }
        conv_results["overall"] = {
            "mrr": sum(all_mrrs) / len(all_mrrs) if all_mrrs else 0.0,
            "f1": sum(all_f1s) / len(all_f1s) if all_f1s else 0.0,
            "recall@10": sum(all_recalls) / len(all_recalls) if all_recalls else 0.0,
            "count": len(all_mrrs),
        }

        for cat, metrics in conv_results.items():
            all_results[cat].append(metrics)

        conv_elapsed = time.time() - conv_t0
        conversation_results.append(
            {
                "conversation_index": conv_idx,
                "elapsed_seconds": conv_elapsed,
                "categories": conv_results,
            }
        )
        log(f"  [RESULTS] Conv {conv_idx+1} — {eval_elapsed:.1f}s eval, {conv_elapsed:.1f}s total")
        for cat in CATEGORY_NAMES + ["overall"]:
            if cat in conv_results:
                r = conv_results[cat]
                marker = " <<<" if cat == "open_domain" else ""
                log(
                    f"    {cat:15s}: MRR={r['mrr']:.3f}  "
                    f"R@10={r.get('recall@10',0):.3f}  n={r['count']}{marker}"
                )

        log_ram()
        log_gpu()

        del retriever, storage, kg, per_category
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    grand_elapsed = time.time() - grand_t0
    aggregated = _aggregate_metrics(all_results)

    log("")
    log("=" * 70)
    log(f"FINAL RESULTS — {len(selected)} conversations — {grand_elapsed:.1f}s total")
    log("=" * 70)

    for cat in CATEGORY_NAMES + ["overall"]:
        if cat in aggregated:
            metrics = aggregated[cat]
            marker = " <<<" if cat in ("open_domain", "overall") else ""
            log(
                f"  {cat:15s}: MRR={metrics['mrr']:.3f}  "
                f"R@10={metrics['recall@10']:.3f}  n={metrics['count']}{marker}"
            )

    log("")
    log(f"Total time: {grand_elapsed:.1f}s ({grand_elapsed/60:.1f}min)")

    return {
        "label": label,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "data_path": LOCOMO_JSON_PATH,
        "overrides": overrides,
        "conversation_indexes": conversation_indexes if conversation_indexes is not None else [],
        "elapsed_seconds": grand_elapsed,
        "aggregated": aggregated,
        "per_conversation": conversation_results,
        "per_query": query_results,
    }


def main() -> int:
    args = _parse_args()
    overrides = _load_overrides(args.overrides_json)
    conversation_indexes = _parse_conversation_indexes(args.conversation_indexes)
    results = run_benchmark(
        label=args.label,
        overrides=overrides,
        conversation_indexes=conversation_indexes,
    )
    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        log(f"Wrote JSON results to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
