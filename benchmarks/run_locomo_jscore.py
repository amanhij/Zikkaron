#!/usr/bin/env python3
"""LoCoMo Standardized Evaluation Harness — Publishable Comparison.

Two evaluation protocols, both industry-standard:

  1. Official LoCoMo F1 (Maharana et al. ACL 2024):
     Stemmed token F1, category-specific handling, adversarial = binary.
     Matches: snap-research/locomo/task_eval/evaluation.py exactly.

  2. Community J-Score (Hindsight, MemMachine, Backboard):
     LLM-as-Judge, binary CORRECT/WRONG, generous grading.
     For apples-to-apples comparison with published systems.

Pipeline:
  1. Ingest conversation into Zikkaron memory
  2. Retrieve top-K memories per question
  3. LLM generates answer from retrieved context
  4. Score with BOTH F1 (automatic) and J-Score (LLM judge)

Usage:
  # FREE — Local model (Qwen2.5-3B on your GPU, no API key):
  python benchmarks/run_locomo_jscore.py --provider local

  # Single conversation quick test:
  python benchmarks/run_locomo_jscore.py --conversation-indexes 0

  # With OpenAI (matches published papers exactly):
  OPENAI_API_KEY=sk-... python benchmarks/run_locomo_jscore.py --provider openai

  # F1-only mode (no LLM judge needed, just generation):
  python benchmarks/run_locomo_jscore.py --f1-only
"""
import argparse
import gc
import json
import os
import re
import string
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.retrieval import HippoRetriever
from zikkaron.storage import StorageEngine

from benchmarks.test_e_locomo import (
    CATEGORY_MAP,
    CATEGORY_NAMES,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    LOCOMO_JSON_PATH,
    _ingest_conversation,
    _make_settings,
)

# ─── Logging ───────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Official LoCoMo F1 Scoring (exact match to snap-research/locomo)
# ═══════════════════════════════════════════════════════════════════════════

try:
    from nltk.stem import PorterStemmer
    _STEMMER = PorterStemmer()
except ImportError:
    _STEMMER = None


def _normalize_answer_official(s: str) -> str:
    """Official LoCoMo normalization: lowercase, remove articles+and, remove punct, whitespace fix."""
    s = s.replace(",", "")
    s = re.sub(r"\b(a|an|the|and)\b", " ", s.lower())
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def _f1_score_official(prediction: str, ground_truth: str) -> float:
    """Official LoCoMo F1 with Porter stemming."""
    pred_norm = _normalize_answer_official(prediction)
    gt_norm = _normalize_answer_official(ground_truth)

    if _STEMMER:
        pred_tokens = [_STEMMER.stem(w) for w in pred_norm.split()]
        gt_tokens = [_STEMMER.stem(w) for w in gt_norm.split()]
    else:
        pred_tokens = pred_norm.split()
        gt_tokens = gt_norm.split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = num_same / len(gt_tokens) if gt_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _f1_multi_answer(prediction: str, ground_truth: str) -> float:
    """Official multi-hop F1: split by comma, max F1 per ground truth sub-answer."""
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    if not ground_truths:
        return 0.0
    scores = []
    for gt in ground_truths:
        scores.append(max(_f1_score_official(pred, gt) for pred in predictions))
    return sum(scores) / len(scores)


def compute_official_f1(prediction: str, gold_answer: str, category_num: int) -> float:
    """Compute F1 exactly matching official LoCoMo eval per category.

    Category mapping: 1=multi_hop, 2=temporal, 3=open_domain, 4=single_hop, 5=adversarial
    """
    # Cat 3 (open_domain): take first answer before semicolon
    if category_num == 3:
        gold_answer = gold_answer.split(";")[0].strip()

    # Cat 2, 3, 4 (temporal, open_domain, single_hop): standard F1
    if category_num in (2, 3, 4):
        return _f1_score_official(prediction, gold_answer)

    # Cat 1 (multi_hop): multi-answer F1
    if category_num == 1:
        return _f1_multi_answer(prediction, gold_answer)

    # Cat 5 (adversarial): binary — check for "no information" / "not mentioned"
    if category_num == 5:
        lower_pred = prediction.lower()
        if "no information available" in lower_pred or "not mentioned" in lower_pred:
            return 1.0
        return 0.0

    return _f1_score_official(prediction, gold_answer)


# ═══════════════════════════════════════════════════════════════════════════
# LLM Providers
# ═══════════════════════════════════════════════════════════════════════════

def _call_openai(model: str, system: str, user: str, temperature: float = 0.1) -> str:
    import openai
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()


def _call_anthropic(model: str, system: str, user: str, temperature: float = 0.1) -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=256,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()


# ─── Google Gemini Provider ───────────────────────────────────────────────
_GEMINI_CLIENT = None


def _ensure_gemini_client():
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is not None:
        return
    from google import genai
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY env var")
    _GEMINI_CLIENT = genai.Client(api_key=api_key)
    log("Gemini client initialized")


def _call_gemini(model: str, system: str, user: str, temperature: float = 0.1) -> str:
    from google.genai import types
    _ensure_gemini_client()

    config_kwargs = dict(
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=1024,
    )
    # Disable thinking for 2.5+ models to save tokens
    if "2.5" in model or "3" in model:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    for attempt in range(5):
        try:
            resp = _GEMINI_CLIENT.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            text = resp.text
            if text is None:
                return ""
            return text.strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = min(15 * (attempt + 1), 60)
                log(f"  [RATE] Gemini 429, waiting {wait}s (attempt {attempt+1}/5)")
                time.sleep(wait)
            else:
                raise
    return ""


# ─── Local Model Provider ─────────────────────────────────────────────────
_LOCAL_MODEL = None
_LOCAL_TOKENIZER = None


_USE_4BIT = False


def _ensure_local_model(model_name: str):
    """Load local model once, reuse across calls."""
    global _LOCAL_MODEL, _LOCAL_TOKENIZER, _USE_4BIT
    if _LOCAL_MODEL is not None:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"Loading local model: {model_name}")
    _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    load_kwargs = dict(device_map="auto", trust_remote_code=True)

    if _USE_4BIT:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        log("Using 4-bit NF4 quantization")
    else:
        load_kwargs["torch_dtype"] = torch.float16

    _LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    log(f"Local model loaded on {_LOCAL_MODEL.device}")


def _call_local(model: str, system: str, user: str, temperature: float = 0.1) -> str:
    import torch
    _ensure_local_model(model)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = _LOCAL_TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = _LOCAL_TOKENIZER(text, return_tensors="pt").to(_LOCAL_MODEL.device)

    with torch.no_grad():
        outputs = _LOCAL_MODEL.generate(
            **inputs,
            max_new_tokens=256,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=_LOCAL_TOKENIZER.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return _LOCAL_TOKENIZER.decode(generated, skip_special_tokens=True).strip()


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Templates
# ═══════════════════════════════════════════════════════════════════════════

GENERATION_SYSTEM = """You answer questions about past conversations between people. Use ONLY the provided context.

Critical rules:
- Give the SHORTEST possible answer. Prefer 1-5 words. Never explain your reasoning.
- If asked "when", use the DATE shown in the memory header (e.g., "7 May 2023"), NOT relative terms like "yesterday" or "last year".
- If asked "what", give just the thing itself, not a full sentence.
- If the context doesn't contain enough information, say "No information available."
- For yes/no questions, answer "Yes" or "No" followed by a brief reason.

Examples of good answers:
- Q: "What did the race raise awareness for?" → "mental health"
- Q: "When did she paint?" → "May 2022"
- Q: "What fields would she pursue?" → "Psychology, counseling"
- Q: "Would she prefer A or B?" → "A, because she enjoys outdoor activities"
"""

GENERATION_USER = """Context (retrieved memories):
{context}

Question: {question}

Answer (be brief):"""

JUDGE_SYSTEM = """You are an expert judge evaluating whether an AI assistant's answer is correct compared to the ground truth answer.

Grading Rules (generous — standard LoCoMo evaluation protocol):
- CORRECT: The generated answer captures the essential meaning of the gold answer, even if worded differently.
- CORRECT: Partial answers that touch on the same topic/facts as the gold answer.
- CORRECT: Different date/time formats that refer to the same event (e.g., "May 7th" vs "7 May").
- CORRECT: Relative time references that align with the gold answer's period.
- CORRECT: Answers that correctly identify unanswerable/adversarial questions (when gold answer indicates the question has no valid answer or is based on false premises).
- WRONG: The answer is factually contradictory to the gold answer.
- WRONG: The answer discusses completely different topics/entities than the gold answer.
- WRONG: The answer says "I don't know" when there IS a valid gold answer.

When in doubt, lean towards CORRECT if the answer demonstrates relevant knowledge.

Respond with ONLY a JSON object:
{"reasoning": "brief explanation", "label": "CORRECT"}
or
{"reasoning": "brief explanation", "label": "WRONG"}"""

JUDGE_USER = """Question: {question}
Gold Answer: {gold_answer}
Generated Answer: {generated_answer}

Evaluate:"""


# ═══════════════════════════════════════════════════════════════════════════
# Core Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def generate_answer(
    question: str,
    retrieved_memories: list[dict],
    *,
    provider: str,
    model: str,
    top_k_context: int = 5,
) -> str:
    """Generate answer using retrieved memories as context."""
    context_parts = []
    for i, mem in enumerate(retrieved_memories[:top_k_context], 1):
        content = mem.get("content", "")
        created_at = mem.get("created_at", "")
        date_str = ""
        if created_at:
            try:
                from datetime import datetime as _dt
                dt = _dt.fromisoformat(created_at.replace("Z", "+00:00"))
                date_str = f" (Date: {dt.strftime('%-d %B %Y')})"
            except (ValueError, TypeError):
                date_str = f" (Date: {created_at})"
        context_parts.append(f"[Memory {i}]{date_str}\n{content}")

    context = "\n\n".join(context_parts) if context_parts else "(No relevant memories found)"
    user_msg = GENERATION_USER.format(context=context, question=question)

    call_fn = {"openai": _call_openai, "anthropic": _call_anthropic, "local": _call_local, "gemini": _call_gemini}[provider]
    return call_fn(model, GENERATION_SYSTEM, user_msg, temperature=0.1)


def judge_answer(
    question: str,
    gold_answer: str,
    generated_answer: str,
    *,
    provider: str,
    model: str,
) -> dict:
    """LLM judge: CORRECT/WRONG."""
    user_msg = JUDGE_USER.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )

    call_fn = {"openai": _call_openai, "anthropic": _call_anthropic, "local": _call_local, "gemini": _call_gemini}[provider]
    raw = call_fn(model, JUDGE_SYSTEM, user_msg, temperature=0.1)

    try:
        cleaned = raw
        if "```" in cleaned:
            cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
        result = json.loads(cleaned)
        if "label" not in result:
            result["label"] = "WRONG"
        return result
    except (json.JSONDecodeError, IndexError):
        if "CORRECT" in raw.upper():
            return {"label": "CORRECT", "reasoning": raw}
        return {"label": "WRONG", "reasoning": raw}


# ─── Reverse category map: name → number ──────────────────────────────────
CATEGORY_NUM = {v: k for k, v in CATEGORY_MAP.items()}


def run_eval(
    *,
    provider: str = "local",
    gen_model: str = "Qwen/Qwen2.5-3B-Instruct",
    judge_model: str | None = None,
    judge_provider: str | None = None,
    conversation_indexes: list[int] | None = None,
    top_k_retrieve: int = 10,
    top_k_context: int = 5,
    overrides: dict | None = None,
    output_path: str | None = None,
    f1_only: bool = False,
) -> dict:
    """Run full LoCoMo evaluation with both F1 and J-Score metrics."""
    overrides = overrides or {}
    judge_provider = judge_provider or provider
    judge_model = judge_model or gen_model

    log("=" * 70)
    log("LoCoMo Standardized Evaluation — Publishable Comparison")
    log("=" * 70)
    log(f"Provider: {provider} | Gen: {gen_model}")
    if not f1_only:
        log(f"Judge: {judge_provider}/{judge_model}")
    log(f"Metrics: Official F1 (automatic){'' if f1_only else ' + J-Score (LLM judge)'}")

    # Load data
    log("Loading LoCoMo dataset...")
    with open(LOCOMO_JSON_PATH) as f:
        locomo_data = json.load(f)
    log(f"Loaded {len(locomo_data)} conversations")

    if conversation_indexes is not None:
        selected = [(idx, locomo_data[idx]) for idx in conversation_indexes]
    else:
        selected = list(enumerate(locomo_data))

    embeddings = EmbeddingEngine(EMBEDDING_MODEL)
    log("Warming up embeddings...")
    _ = embeddings.encode_query("test warmup")

    all_results = []
    grand_t0 = time.time()
    llm_calls = 0

    for run_idx, (conv_idx, conv) in enumerate(selected):
        conv_t0 = time.time()
        qa_items = conv.get("qa", conv.get("qa_pairs", []))
        n_qa = len(qa_items)

        log(f"\n{'='*60}")
        log(f"CONVERSATION {run_idx+1}/{len(selected)} (source={conv_idx}) — {n_qa} QA pairs")
        log(f"{'='*60}")

        if not qa_items:
            continue

        # Setup Zikkaron
        tmp_dir = f"/tmp/locomo_eval_{conv_idx}"
        os.makedirs(tmp_dir, exist_ok=True)
        db_path = os.path.join(tmp_dir, "memory.db")
        project_dir = os.path.join(tmp_dir, "project")
        os.makedirs(project_dir, exist_ok=True)
        if os.path.exists(db_path):
            os.remove(db_path)

        settings = _make_settings(DB_PATH=db_path, **overrides)
        storage = StorageEngine(db_path, embedding_dim=EMBEDDING_DIM)
        kg = KnowledgeGraph(storage, settings)

        log("  [INGEST] Starting...")
        ingest_t0 = time.time()
        _ingest_conversation(conv, storage, embeddings, project_dir, obs_mode=True, settings=settings)
        log(f"  [INGEST] Done in {time.time()-ingest_t0:.1f}s")

        retriever = HippoRetriever(storage, embeddings, kg, settings)

        for qi, qa in enumerate(qa_items):
            question = qa.get("question", qa.get("query", ""))
            gold_answer = str(qa.get("answer", ""))
            raw_category = qa.get("category", "unknown")
            category = CATEGORY_MAP.get(raw_category, str(raw_category))
            category_num = raw_category if isinstance(raw_category, int) else CATEGORY_NUM.get(category, 0)

            if not question:
                continue

            # Step 1: Retrieve
            results = retriever.recall(query=question, max_results=top_k_retrieve)

            # Step 2: Generate
            try:
                generated = generate_answer(
                    question, results,
                    provider=provider, model=gen_model,
                    top_k_context=top_k_context,
                )
                llm_calls += 1
            except Exception as e:
                log(f"  [ERROR] Gen Q{qi}: {e}")
                generated = "No information available."

            # Step 3: Official F1 (automatic, free)
            official_f1 = compute_official_f1(generated, gold_answer, category_num)

            # Step 4: J-Score (LLM judge, optional)
            j_correct = None
            j_reasoning = ""
            if not f1_only:
                try:
                    judgment = judge_answer(
                        question, gold_answer, generated,
                        provider=judge_provider, model=judge_model,
                    )
                    llm_calls += 1
                    j_correct = judgment["label"].upper() == "CORRECT"
                    j_reasoning = judgment.get("reasoning", "")
                except Exception as e:
                    log(f"  [ERROR] Judge Q{qi}: {e}")
                    j_correct = False
                    j_reasoning = f"Error: {e}"

            sample = {
                "conversation_index": conv_idx,
                "query_index": qi,
                "category": category,
                "category_num": category_num,
                "question": question,
                "gold_answer": gold_answer,
                "generated_answer": generated,
                "official_f1": official_f1,
                "j_correct": j_correct,
                "j_reasoning": j_reasoning,
            }
            all_results.append(sample)

            if (qi + 1) % 10 == 0 or qi == n_qa - 1:
                avg_f1 = sum(r["official_f1"] for r in all_results) / len(all_results)
                status = f"F1={avg_f1:.3f}"
                if not f1_only:
                    j_total = sum(1 for r in all_results if r["j_correct"] is not None)
                    j_ok = sum(1 for r in all_results if r["j_correct"])
                    status += f" J={j_ok/j_total*100:.1f}%" if j_total else ""
                log(f"  [EVAL] Q{qi+1}/{n_qa} ({category}) — {status} — LLM calls: {llm_calls}")

        log(f"  [DONE] Conv {conv_idx} in {time.time()-conv_t0:.1f}s")
        del retriever, storage, kg
        gc.collect()

    # ─── Aggregate ─────────────────────────────────────────────────────────
    grand_elapsed = time.time() - grand_t0

    per_category = defaultdict(list)
    for r in all_results:
        per_category[r["category"]].append(r)
        per_category["overall"].append(r)

    metrics_table = {}
    for cat in CATEGORY_NAMES + ["overall"]:
        items = per_category.get(cat, [])
        if not items:
            continue
        n = len(items)
        avg_f1 = sum(r["official_f1"] for r in items) / n
        entry = {"f1": avg_f1, "n": n}
        if not f1_only:
            j_items = [r for r in items if r["j_correct"] is not None]
            if j_items:
                j_ok = sum(1 for r in j_items if r["j_correct"])
                entry["j_score"] = j_ok / len(j_items) * 100
                entry["j_correct"] = j_ok
                entry["j_total"] = len(j_items)
        metrics_table[cat] = entry

    # ─── Print results ─────────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log(f"FINAL RESULTS — {len(selected)} conversations — {grand_elapsed:.1f}s")
    log(f"{'='*70}")
    log(f"Provider: {provider} | Gen: {gen_model} | LLM calls: {llm_calls}")
    log("")

    # Official F1 table
    log("Official LoCoMo F1 (Maharana et al. ACL 2024):")
    for cat in CATEGORY_NAMES + ["overall"]:
        if cat in metrics_table:
            m = metrics_table[cat]
            marker = " <<<" if cat in ("open_domain", "overall") else ""
            log(f"  {cat:15s}: F1={m['f1']:.3f}  n={m['n']}{marker}")

    # J-Score table
    if not f1_only:
        log("")
        log("Community J-Score (LLM-as-Judge, binary CORRECT/WRONG):")
        for cat in CATEGORY_NAMES + ["overall"]:
            if cat in metrics_table and "j_score" in metrics_table[cat]:
                m = metrics_table[cat]
                marker = " <<<" if cat in ("open_domain", "overall") else ""
                log(f"  {cat:15s}: J-Score={m['j_score']:5.1f}%  ({m['j_correct']}/{m['j_total']}){marker}")

    # Comparison
    log("")
    log("┌─────────────────────────────────────────────────────────────┐")
    log("│  Comparison with Published Systems (LoCoMo-mc10)           │")
    log("├───────────────────────┬──────────┬─────────────┬───────────┤")
    log("│ System                │ Overall  │ Open_Domain │ Metric    │")
    log("├───────────────────────┼──────────┼─────────────┼───────────┤")
    if "overall" in metrics_table:
        f1_str = f"{metrics_table['overall']['f1']:.3f}"
        od_f1 = f"{metrics_table.get('open_domain', {}).get('f1', 0):.3f}"
        log(f"│ Zikkaron (ours)       │  {f1_str:>6s}  │    {od_f1:>6s}   │ F1        │")
        if not f1_only and "j_score" in metrics_table.get("overall", {}):
            j_str = f"{metrics_table['overall']['j_score']:.1f}%"
            od_j = f"{metrics_table.get('open_domain', {}).get('j_score', 0):.1f}%"
            log(f"│ Zikkaron (ours)       │  {j_str:>6s}  │    {od_j:>6s}   │ J-Score   │")
    log("├───────────────────────┼──────────┼─────────────┼───────────┤")
    log("│ MemMachine v0.2       │  91.7%   │    75.0%    │ J-Score   │")
    log("│ Hindsight             │  89.6%   │      —      │ J-Score   │")
    log("│ Full-Context baseline │  89.0%   │    71.9%    │ J-Score   │")
    log("│ Memobase v0.37        │  75.8%   │    77.2%    │ J-Score   │")
    log("│ Mem0-Graph            │  68.4%   │    75.7%    │ J-Score   │")
    log("└───────────────────────┴──────────┴─────────────┴───────────┘")
    log("")
    log("Note: F1 and J-Score are different metrics. F1 is automatic token")
    log("overlap; J-Score is LLM-judged answer correctness. Both are standard")
    log("for LoCoMo evaluation. Report both for completeness.")

    output = {
        "eval_protocol": "locomo_standardized_v1",
        "provider": provider,
        "gen_model": gen_model,
        "judge_model": judge_model if not f1_only else None,
        "f1_only": f1_only,
        "top_k_retrieve": top_k_retrieve,
        "top_k_context": top_k_context,
        "overrides": overrides,
        "conversation_indexes": conversation_indexes or list(range(len(locomo_data))),
        "elapsed_seconds": grand_elapsed,
        "llm_calls": llm_calls,
        "metrics_table": metrics_table,
        "per_query": all_results,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        log(f"Results written to {output_path}")

    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="local", choices=["local", "openai", "anthropic", "gemini"],
                        help="LLM provider. 'local' runs free on GPU (default).")
    parser.add_argument("--gen-model", default=None,
                        help="Model for answer generation.")
    parser.add_argument("--judge-model", default=None,
                        help="Model for J-Score judging. Default: same as gen-model.")
    parser.add_argument("--judge-provider", default=None,
                        choices=["local", "openai", "anthropic", "gemini"],
                        help="Provider for J-Score judging. Default: same as --provider.")
    parser.add_argument("--conversation-indexes", default="",
                        help="Comma-separated conversation indexes. Default: all.")
    parser.add_argument("--top-k-retrieve", type=int, default=10,
                        help="Number of memories to retrieve.")
    parser.add_argument("--top-k-context", type=int, default=5,
                        help="Number of memories to include in LLM context.")
    parser.add_argument("--output-json", default="benchmarks/eval_results.json",
                        help="Output JSON path.")
    parser.add_argument("--overrides-json", default="",
                        help="JSON overrides for Zikkaron settings.")
    parser.add_argument("--f1-only", action="store_true",
                        help="Only compute F1 (skip LLM judge). Halves LLM calls.")
    parser.add_argument("--4bit", dest="use_4bit", action="store_true",
                        help="Load model in 4-bit NF4 quantization (fits 7B in 8GB VRAM).")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    global _USE_4BIT
    _USE_4BIT = args.use_4bit

    DEFAULT_MODELS = {
        "local": "Qwen/Qwen2.5-7B-Instruct" if args.use_4bit else "Qwen/Qwen2.5-3B-Instruct",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-sonnet-4-20250514",
        "gemini": "gemini-2.5-flash",
    }
    if args.gen_model is None:
        args.gen_model = DEFAULT_MODELS[args.provider]

    judge_prov = args.judge_provider or args.provider
    if args.judge_model is None:
        args.judge_model = DEFAULT_MODELS.get(judge_prov, args.gen_model)

    conv_indexes = None
    if args.conversation_indexes:
        conv_indexes = [int(x.strip()) for x in args.conversation_indexes.split(",") if x.strip()]

    overrides = {}
    if args.overrides_json:
        if os.path.exists(args.overrides_json):
            with open(args.overrides_json) as f:
                overrides = json.load(f)
        else:
            overrides = json.loads(args.overrides_json)

    run_eval(
        provider=args.provider,
        gen_model=args.gen_model,
        judge_model=args.judge_model,
        judge_provider=judge_prov,
        conversation_indexes=conv_indexes,
        top_k_retrieve=args.top_k_retrieve,
        top_k_context=args.top_k_context,
        overrides=overrides,
        output_path=args.output_json,
        f1_only=args.f1_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
