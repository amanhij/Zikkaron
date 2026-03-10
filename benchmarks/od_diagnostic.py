"""Open_domain diagnostic: show what CE ranks above correct evidence."""
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.retrieval import HippoRetriever
from zikkaron.storage import StorageEngine

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768
CATEGORY_MAP = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}

LOCOMO_JSON_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--Percena--locomo-mc10/"
    "snapshots/7d59a0463d83f97b042684310c0b3d17553004cd/raw/locomo10.json"
)

# Import benchmark helpers
from test_e_locomo import (
    _ingest_conversation, _make_settings, _reformulate_to_observation
)


def run_diagnostic(n_convs=3, target_category="open_domain"):
    with open(LOCOMO_JSON_PATH) as f:
        data = json.load(f)

    embeddings = EmbeddingEngine(EMBEDDING_MODEL)

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_10 = 0
    total = 0
    miss_details = []

    for conv_idx in range(min(n_convs, len(data))):
        conv = data[conv_idx]
        tmp_dir = f"/tmp/locomo_diag_{conv_idx}"
        os.makedirs(tmp_dir, exist_ok=True)
        db_path = os.path.join(tmp_dir, "memory.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        project_dir = os.path.join(tmp_dir, "project")
        os.makedirs(project_dir, exist_ok=True)

        settings = _make_settings(DB_PATH=db_path)
        storage = StorageEngine(db_path, embedding_dim=EMBEDDING_DIM)
        kg = KnowledgeGraph(storage, settings)
        dia_map = _ingest_conversation(conv, storage, embeddings, project_dir, obs_mode=True)
        retriever = HippoRetriever(storage, embeddings, kg, settings)

        qa_items = conv.get("qa", [])
        for qa in qa_items:
            raw_cat = qa.get("category", "")
            category = CATEGORY_MAP.get(raw_cat, str(raw_cat))
            if category != target_category:
                continue

            question = qa.get("question", "")
            answer = qa.get("answer", "")
            evidence_ids = qa.get("evidence", [])
            if isinstance(evidence_ids, str):
                evidence_ids = [evidence_ids]

            relevant_mem_ids = {dia_map.get(str(eid), -999) for eid in evidence_ids}

            results = retriever.recall(query=question, max_results=10)
            retrieved_ids = [r.get("id", -1) for r in results]

            # Find rank of first relevant result
            rank = None
            for i, mid in enumerate(retrieved_ids, 1):
                if mid in relevant_mem_ids:
                    rank = i
                    break

            total += 1
            if rank == 1:
                hit_at_1 += 1
            if rank and rank <= 3:
                hit_at_3 += 1
            if rank and rank <= 10:
                hit_at_10 += 1

            rr = 1.0 / rank if rank else 0.0

            if rank is None or rank > 1:
                # Show what was ranked above the correct evidence
                print(f"\n--- Conv {conv_idx}, Q: {question[:80]}")
                print(f"    A: {answer[:80]}")
                print(f"    Evidence: {evidence_ids}")
                print(f"    Rank: {rank or 'NOT IN TOP 10'} (RR={rr:.3f})")
                for i, r in enumerate(results[:5], 1):
                    mid = r.get("id", -1)
                    content = r.get("content", "")[:120]
                    ce_score = r.get("_cross_encoder_score", "?")
                    is_relevant = "***" if mid in relevant_mem_ids else "   "
                    print(f"    {is_relevant} #{i} (CE={ce_score}) id={mid}: {content}")

    print(f"\n=== {target_category} Diagnostic ===")
    print(f"Total: {total}")
    print(f"Hit@1: {hit_at_1}/{total} = {hit_at_1/total:.3f}" if total else "No questions")
    print(f"Hit@3: {hit_at_3}/{total} = {hit_at_3/total:.3f}" if total else "")
    print(f"Hit@10: {hit_at_10}/{total} = {hit_at_10/total:.3f}" if total else "")
    print(f"MRR: {sum(1/r if r else 0 for r in []) / total:.3f}" if total else "")  # placeholder


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    cat = sys.argv[2] if len(sys.argv) > 2 else "open_domain"
    run_diagnostic(n, cat)
