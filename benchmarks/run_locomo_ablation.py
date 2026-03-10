#!/usr/bin/env python3
"""Run the requested LoCoMo ablation study and write a markdown report."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmarks" / "run_benchmark_gpu.py"
RESULTS_DIR = ROOT / "benchmarks" / "ablation_runs"
REPORT_PATH = ROOT / "benchmarks" / "ABLATION_RESULTS.md"
CATEGORY_NAMES = ["single_hop", "multi_hop", "temporal", "open_domain", "adversarial", "overall"]
BENCHMARK_CATEGORIES = CATEGORY_NAMES[:-1]
ALL_CONVERSATIONS = list(range(10))
TARGET_OPEN_DOMAIN_MRR = 0.50
TARGET_OVERALL_MRR = 0.72
TARGET_MAX_DROP = 0.02
BOOTSTRAP_SEED = 1337
BOOTSTRAP_SAMPLES = 5000

BASELINE_OVERRIDES = {
    "INDEX_ENRICHMENT_ENABLED": False,
    "PROFILE_EXTRACTION_ENABLED": False,
    "DERIVED_BELIEFS_ENABLED": False,
    "COMPARISON_DUAL_SEARCH_ENABLED": False,
    "FUSION_METHOD": "wrrf",
    "GTE_RERANKER_ENABLED": False,
    "NLI_RERANKING_ENABLED": False,
    "MULTI_PASSAGE_RERANKING_ENABLED": False,
    "CONCEPTNET_ENRICHMENT_ENABLED": True,
    "LOGIC_ENRICHMENT_ENABLED": True,
    "COMET_ENRICHMENT_ENABLED": False,
    "DOC2QUERY_ENRICHMENT_ENABLED": False,
    "NLI_ONLY_FOR_OPEN_DOMAIN": True,
}


@dataclass(frozen=True)
class ConfigSpec:
    label: str
    phase: str
    description: str
    overrides: dict


PHASE1_CONFIGS = [
    ConfigSpec("baseline_v16", "baseline", "All v17+ features disabled", BASELINE_OVERRIDES),
    ConfigSpec(
        "enrichment_only",
        "ablation",
        "Index enrichment only (ConceptNet + Logic; COMET/Doc2Query off)",
        {
            **BASELINE_OVERRIDES,
            "INDEX_ENRICHMENT_ENABLED": True,
            "CONCEPTNET_ENRICHMENT_ENABLED": True,
            "LOGIC_ENRICHMENT_ENABLED": True,
            "COMET_ENRICHMENT_ENABLED": False,
            "DOC2QUERY_ENRICHMENT_ENABLED": False,
        },
    ),
    ConfigSpec(
        "profiles_beliefs_only",
        "ablation",
        "Profile extraction + derived beliefs only",
        {
            **BASELINE_OVERRIDES,
            "PROFILE_EXTRACTION_ENABLED": True,
            "DERIVED_BELIEFS_ENABLED": True,
        },
    ),
    ConfigSpec(
        "fusion_convex_only",
        "ablation",
        "Convex fusion only (vs WRRF)",
        {
            **BASELINE_OVERRIDES,
            "FUSION_METHOD": "convex",
        },
    ),
    ConfigSpec(
        "gte_reranker_only",
        "ablation",
        "GTE reranker only (vs FlashRank)",
        {
            **BASELINE_OVERRIDES,
            "GTE_RERANKER_ENABLED": True,
            "GTE_RERANKER_FALLBACK_TO_FLASHRANK": True,
        },
    ),
    ConfigSpec(
        "nli_only",
        "ablation",
        "NLI reranking only (open-domain only)",
        {
            **BASELINE_OVERRIDES,
            "NLI_RERANKING_ENABLED": True,
            "NLI_ONLY_FOR_OPEN_DOMAIN": True,
        },
    ),
    ConfigSpec(
        "multi_passage_only",
        "ablation",
        "Multi-passage reranking only",
        {
            **BASELINE_OVERRIDES,
            "MULTI_PASSAGE_RERANKING_ENABLED": True,
        },
    ),
    ConfigSpec(
        "comparison_dual_search_only",
        "ablation",
        "Comparison dual-search only",
        {
            **BASELINE_OVERRIDES,
            "COMPARISON_DUAL_SEARCH_ENABLED": True,
        },
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="Parallel benchmark shards per config.")
    parser.add_argument(
        "--phase",
        default="all",
        choices=["all", "phase1", "phase3", "phase4"],
        help="Run just one phase or the full study.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run configs even if merged JSON outputs already exist.",
    )
    return parser.parse_args()


def _partition_round_robin(items: list[int], parts: int) -> list[list[int]]:
    buckets = [[] for _ in range(parts)]
    for idx, item in enumerate(items):
        buckets[idx % parts].append(item)
    return [bucket for bucket in buckets if bucket]


def _query_key(sample: dict) -> tuple[int, int]:
    return sample["conversation_index"], sample["query_index"]


def _aggregate_from_queries(per_query: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in per_query:
        grouped[sample["category"]].append(sample)
        grouped["overall"].append(sample)

    aggregated = {}
    for category, items in grouped.items():
        count = len(items)
        aggregated[category] = {
            "mrr": sum(item["mrr"] for item in items) / count if count else 0.0,
            "recall@10": sum(item["recall@10"] for item in items) / count if count else 0.0,
            "f1": sum(item["f1"] for item in items) / count if count else 0.0,
            "count": count,
        }
    return aggregated


def _merge_run_parts(spec: ConfigSpec, part_files: list[Path], wall_elapsed: float) -> dict:
    merged_queries = []
    merged_conversations = []
    embedding_model = None
    embedding_dim = None
    data_path = None
    for part_file in part_files:
        with open(part_file) as f:
            part = json.load(f)
        embedding_model = part["embedding_model"]
        embedding_dim = part["embedding_dim"]
        data_path = part["data_path"]
        merged_queries.extend(part["per_query"])
        merged_conversations.extend(part["per_conversation"])

    merged_queries.sort(key=_query_key)
    merged_conversations.sort(key=lambda x: x["conversation_index"])
    aggregated = _aggregate_from_queries(merged_queries)
    return {
        "label": spec.label,
        "phase": spec.phase,
        "description": spec.description,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "data_path": data_path,
        "overrides": spec.overrides,
        "conversation_indexes": ALL_CONVERSATIONS,
        "elapsed_seconds": wall_elapsed,
        "aggregated": aggregated,
        "per_conversation": merged_conversations,
        "per_query": merged_queries,
        "part_files": [str(path) for path in part_files],
    }


def _run_sharded_config(spec: ConfigSpec, *, workers: int, force: bool) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    merged_path = RESULTS_DIR / f"{spec.label}.json"
    if merged_path.exists() and not force:
        return merged_path

    shards = _partition_round_robin(ALL_CONVERSATIONS, max(1, workers))
    procs = []
    part_paths = []
    part_logs = []
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    env.setdefault("NUMEXPR_NUM_THREADS", "2")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    started = time.time()
    for shard_idx, shard in enumerate(shards):
        part_path = RESULTS_DIR / f"{spec.label}.part{shard_idx}.json"
        part_log = RESULTS_DIR / f"{spec.label}.part{shard_idx}.log"
        part_paths.append(part_path)
        part_logs.append(part_log)
        command = [
            sys.executable,
            "-u",
            str(RUNNER),
            "--label",
            f"{spec.label}_part{shard_idx}",
            "--conversation-indexes",
            ",".join(str(i) for i in shard),
            "--overrides-json",
            json.dumps(spec.overrides),
            "--output-json",
            str(part_path),
        ]
        with open(part_log, "w") as log_f:
            proc = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
        procs.append(proc)

    failed = []
    for proc, part_log in zip(procs, part_logs):
        rc = proc.wait()
        if rc != 0:
            failed.append((rc, part_log))
    if failed:
        details = ", ".join(f"{log} (rc={rc})" for rc, log in failed)
        raise RuntimeError(f"{spec.label} failed: {details}")

    merged = _merge_run_parts(spec, part_paths, time.time() - started)
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)
    return merged_path


def _load_run(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _extract_series(run: dict, category: str) -> list[float]:
    return [sample["mrr"] for sample in run["per_query"] if category == "overall" or sample["category"] == category]


def _paired_delta_stats(baseline: dict, candidate: dict, category: str) -> dict:
    base_map = {
        _query_key(sample): sample["mrr"]
        for sample in baseline["per_query"]
        if category == "overall" or sample["category"] == category
    }
    cand_map = {
        _query_key(sample): sample["mrr"]
        for sample in candidate["per_query"]
        if category == "overall" or sample["category"] == category
    }
    keys = sorted(base_map.keys())
    deltas = [cand_map[key] - base_map[key] for key in keys]
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0

    rng = random.Random(BOOTSTRAP_SEED)
    boot_means = []
    n = len(deltas)
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(0.025 * len(boot_means))]
    hi = boot_means[int(0.975 * len(boot_means))]

    wins = sum(1 for delta in deltas if delta > 0)
    losses = sum(1 for delta in deltas if delta < 0)
    ties = n - wins - losses
    p_value = _two_sided_sign_test_pvalue(wins, losses)
    return {
        "mean_delta": mean_delta,
        "ci95": (lo, hi),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n": n,
        "p_value": p_value,
    }


def _two_sided_sign_test_pvalue(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    k = min(wins, losses)
    cumulative = sum(math.comb(trials, i) for i in range(k + 1)) / (2 ** trials)
    return min(1.0, 2 * cumulative)


def _max_category_drop(baseline_run: dict, candidate_run: dict) -> float:
    drops = []
    for category in BENCHMARK_CATEGORIES:
        base = baseline_run["aggregated"][category]["mrr"]
        cand = candidate_run["aggregated"].get(category, {}).get("mrr", 0.0)
        drops.append(base - cand)
    return max(drops) if drops else 0.0


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _format_signed(value: float) -> str:
    return f"{value:+.3f}"


def _build_phase1_rows(runs: dict[str, dict], baseline_run: dict) -> list[dict]:
    rows = []
    for spec in PHASE1_CONFIGS:
        run = runs[spec.label]
        row = {
            "label": spec.label,
            "phase": spec.phase,
            "description": spec.description,
            "run": run,
            "overall_stats": _paired_delta_stats(baseline_run, run, "overall"),
            "open_stats": _paired_delta_stats(baseline_run, run, "open_domain"),
            "max_drop": _max_category_drop(baseline_run, run),
        }
        rows.append(row)
    return rows


def _pick_combined_spec(phase1_rows: list[dict]) -> ConfigSpec:
    selected = {}
    descriptions = []
    for row in phase1_rows:
        if row["label"] == "baseline_v16":
            continue
        if row["overall_stats"]["mean_delta"] >= 0:
            selected.update(row["run"]["overrides"])
            descriptions.append(row["description"])

    if not selected:
        selected = dict(BASELINE_OVERRIDES)
        descriptions = ["No feature had non-negative overall MRR; combined config falls back to baseline."]
    else:
        selected = {**BASELINE_OVERRIDES, **selected}

    return ConfigSpec(
        "combined_non_regressing",
        "combined",
        "; ".join(descriptions),
        selected,
    )


def _maybe_build_grid_specs(combined_spec: ConfigSpec) -> list[ConfigSpec]:
    specs = []
    base = dict(combined_spec.overrides)
    # Keep the sweep narrow enough to finish in practice on one dev conversation.
    if base.get("NLI_RERANKING_ENABLED"):
        for weight in [0.05, 0.10, 0.15, 0.20, 0.30]:
            specs.append(
                ConfigSpec(
                    f"grid_dev_nli_{str(weight).replace('.', '_')}",
                    "grid-dev",
                    f"Dev sweep: NLI_WEIGHT={weight}",
                    {**base, "NLI_WEIGHT": weight},
                )
            )
    if base.get("PROFILE_EXTRACTION_ENABLED"):
        for weight in [0.5, 0.8, 1.0]:
            specs.append(
                ConfigSpec(
                    f"grid_dev_profile_{str(weight).replace('.', '_')}",
                    "grid-dev",
                    f"Dev sweep: PROFILE_SEARCH_WEIGHT={weight}",
                    {**base, "PROFILE_SEARCH_WEIGHT": weight},
                )
            )
    if base.get("INDEX_ENRICHMENT_ENABLED"):
        for threshold in [0.15, 0.25, 0.35]:
            specs.append(
                ConfigSpec(
                    f"grid_dev_fpa_{str(threshold).replace('.', '_')}",
                    "grid-dev",
                    f"Dev sweep: FPA_SIMILARITY_THRESHOLD={threshold}",
                    {**base, "FPA_SIMILARITY_THRESHOLD": threshold},
                )
            )
    return specs


def _run_dev_config(spec: ConfigSpec, *, force: bool) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{spec.label}.json"
    if path.exists() and not force:
        return path

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    env.setdefault("NUMEXPR_NUM_THREADS", "2")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    command = [
        sys.executable,
        "-u",
        str(RUNNER),
        "--label",
        spec.label,
        "--conversation-indexes",
        "0",
        "--overrides-json",
        json.dumps(spec.overrides),
        "--output-json",
        str(path),
    ]
    log_path = RESULTS_DIR / f"{spec.label}.log"
    with open(log_path, "w") as log_f:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT, check=True)
    return path


def _select_best_dev_run(base_dev: dict, dev_runs: list[dict]) -> dict | None:
    best = None
    for run in dev_runs:
        overall_delta = run["aggregated"]["overall"]["mrr"] - base_dev["aggregated"]["overall"]["mrr"]
        open_delta = run["aggregated"]["open_domain"]["mrr"] - base_dev["aggregated"]["open_domain"]["mrr"]
        key = (open_delta, overall_delta)
        if best is None or key > best[0]:
            best = (key, run)
    return best[1] if best else None


def _markdown_metric_table(rows: list[dict], baseline_run: dict) -> str:
    header = (
        "| Run | Phase | overall MRR | open_domain MRR | single_hop | multi_hop | temporal | adversarial | "
        "Δoverall | Δopen | Max drop | Overall 95% CI | Open 95% CI | Win/Loss/Tie (overall) | Target |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|\n"
    )
    lines = [header]
    for row in rows:
        run = row["run"]
        agg = run["aggregated"]
        overall_stats = row["overall_stats"]
        open_stats = row["open_stats"]
        target_ok = (
            agg["open_domain"]["mrr"] >= TARGET_OPEN_DOMAIN_MRR
            and agg["overall"]["mrr"] >= TARGET_OVERALL_MRR
            and row["max_drop"] <= TARGET_MAX_DROP
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    row["phase"],
                    _format_float(agg["overall"]["mrr"]),
                    _format_float(agg["open_domain"]["mrr"]),
                    _format_float(agg["single_hop"]["mrr"]),
                    _format_float(agg["multi_hop"]["mrr"]),
                    _format_float(agg["temporal"]["mrr"]),
                    _format_float(agg["adversarial"]["mrr"]),
                    _format_signed(overall_stats["mean_delta"]),
                    _format_signed(open_stats["mean_delta"]),
                    _format_float(row["max_drop"]),
                    f"{_format_signed(overall_stats['ci95'][0])} to {_format_signed(overall_stats['ci95'][1])}",
                    f"{_format_signed(open_stats['ci95'][0])} to {_format_signed(open_stats['ci95'][1])}",
                    f"{overall_stats['wins']}/{overall_stats['losses']}/{overall_stats['ties']}",
                    "pass" if target_ok else "fail",
                ]
            )
            + " |\n"
        )
    return "".join(lines)


def _markdown_dev_table(base_run: dict, runs: list[dict]) -> str:
    header = (
        "| Run | overall MRR | open_domain MRR | Δoverall vs combined | Δopen vs combined |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    lines = [header]
    for run in runs:
        lines.append(
            "| "
            + " | ".join(
                [
                    run["label"],
                    _format_float(run["aggregated"]["overall"]["mrr"]),
                    _format_float(run["aggregated"]["open_domain"]["mrr"]),
                    _format_signed(run["aggregated"]["overall"]["mrr"] - base_run["aggregated"]["overall"]["mrr"]),
                    _format_signed(run["aggregated"]["open_domain"]["mrr"] - base_run["aggregated"]["open_domain"]["mrr"]),
                ]
            )
            + " |\n"
        )
    return "".join(lines)


def _write_report(
    *,
    phase1_rows: list[dict],
    combined_row: dict,
    dev_rows: list[dict],
    validated_row: dict | None,
) -> None:
    lines = []
    lines.append("# LoCoMo Ablation Results\n\n")
    lines.append("Benchmark command: `python3 benchmarks/run_benchmark_gpu.py`\n\n")
    lines.append(
        "Method: full LoCoMo micro-averaged MRR/Recall@10 from the benchmark harness, "
        "paired by query against the baseline. Confidence intervals are paired bootstrap "
        f"95% intervals over reciprocal-rank deltas ({BOOTSTRAP_SAMPLES} resamples, seed={BOOTSTRAP_SEED}).\n\n"
    )
    lines.append("## Phase 1-3 Results\n\n")
    lines.append(_markdown_metric_table(phase1_rows + [combined_row], phase1_rows[0]["run"]))
    lines.append("\n")

    combined_agg = combined_row["run"]["aggregated"]
    combined_target_ok = (
        combined_agg["open_domain"]["mrr"] >= TARGET_OPEN_DOMAIN_MRR
        and combined_agg["overall"]["mrr"] >= TARGET_OVERALL_MRR
        and combined_row["max_drop"] <= TARGET_MAX_DROP
    )
    lines.append(
        "Combined decision: "
        f"`{combined_row['label']}` uses every phase-1 feature with non-negative overall MRR delta. "
        f"Targets {'met' if combined_target_ok else 'not met'}.\n\n"
    )

    if dev_rows:
        lines.append("## Phase 4 Dev Sweep\n\n")
        lines.append(_markdown_dev_table(combined_row["run"], [row["run"] for row in dev_rows]))
        lines.append("\n")
        if validated_row is not None:
            lines.append("## Phase 4 Validation\n\n")
            lines.append(_markdown_metric_table([validated_row], phase1_rows[0]["run"]))
            lines.append("\n")

    REPORT_PATH.write_text("".join(lines))


def main() -> int:
    args = parse_args()

    phase1_paths = {}
    if args.phase in {"all", "phase1", "phase3", "phase4"}:
        for spec in PHASE1_CONFIGS:
            phase1_paths[spec.label] = _run_sharded_config(spec, workers=args.workers, force=args.force)

    phase1_runs = {label: _load_run(path) for label, path in phase1_paths.items()}
    baseline_run = phase1_runs["baseline_v16"]
    phase1_rows = _build_phase1_rows(phase1_runs, baseline_run)

    combined_spec = _pick_combined_spec(phase1_rows)
    combined_path = _run_sharded_config(combined_spec, workers=args.workers, force=args.force)
    combined_run = _load_run(combined_path)
    combined_row = {
        "label": combined_spec.label,
        "phase": combined_spec.phase,
        "description": combined_spec.description,
        "run": combined_run,
        "overall_stats": _paired_delta_stats(baseline_run, combined_run, "overall"),
        "open_stats": _paired_delta_stats(baseline_run, combined_run, "open_domain"),
        "max_drop": _max_category_drop(baseline_run, combined_run),
    }

    dev_rows = []
    validated_row = None
    combined_target_ok = (
        combined_run["aggregated"]["open_domain"]["mrr"] >= TARGET_OPEN_DOMAIN_MRR
        and combined_run["aggregated"]["overall"]["mrr"] >= TARGET_OVERALL_MRR
        and combined_row["max_drop"] <= TARGET_MAX_DROP
    )

    if not combined_target_ok and args.phase in {"all", "phase4"}:
        dev_specs = _maybe_build_grid_specs(combined_spec)
        dev_runs = []
        if dev_specs:
            combined_dev_spec = ConfigSpec(
                "combined_non_regressing_dev",
                "grid-dev",
                "Combined config on conversation 0",
                combined_spec.overrides,
            )
            combined_dev_run = _load_run(_run_dev_config(combined_dev_spec, force=args.force))
            for spec in dev_specs:
                run = _load_run(_run_dev_config(spec, force=args.force))
                dev_runs.append(run)
                dev_rows.append(
                    {
                        "label": spec.label,
                        "phase": spec.phase,
                        "description": spec.description,
                        "run": run,
                        "overall_stats": {
                            "mean_delta": run["aggregated"]["overall"]["mrr"] - combined_dev_run["aggregated"]["overall"]["mrr"],
                            "ci95": (0.0, 0.0),
                            "wins": 0,
                            "losses": 0,
                            "ties": 0,
                        },
                        "open_stats": {
                            "mean_delta": run["aggregated"]["open_domain"]["mrr"] - combined_dev_run["aggregated"]["open_domain"]["mrr"],
                            "ci95": (0.0, 0.0),
                            "wins": 0,
                            "losses": 0,
                            "ties": 0,
                        },
                        "max_drop": 0.0,
                    }
                )

            best_dev = _select_best_dev_run(combined_dev_run, dev_runs)
            if best_dev is not None:
                validated_spec = ConfigSpec(
                    "validated_best_dev",
                    "grid-validated",
                    f"Best dev sweep validated on full benchmark ({best_dev['label']})",
                    best_dev["overrides"],
                )
                validated_run = _load_run(_run_sharded_config(validated_spec, workers=args.workers, force=args.force))
                validated_row = {
                    "label": validated_spec.label,
                    "phase": validated_spec.phase,
                    "description": validated_spec.description,
                    "run": validated_run,
                    "overall_stats": _paired_delta_stats(baseline_run, validated_run, "overall"),
                    "open_stats": _paired_delta_stats(baseline_run, validated_run, "open_domain"),
                    "max_drop": _max_category_drop(baseline_run, validated_run),
                }

    _write_report(
        phase1_rows=phase1_rows,
        combined_row=combined_row,
        dev_rows=dev_rows,
        validated_row=validated_row,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
