# LoCoMo Ablation Results

Benchmark command: `python3 benchmarks/run_benchmark_gpu.py`

Method: full LoCoMo micro-averaged MRR/Recall@10 from the benchmark harness, paired by query against the baseline. Confidence intervals are paired bootstrap 95% intervals over reciprocal-rank deltas (5000 resamples, seed=1337).

## Phase 1-3 Results

| Run | Phase | overall MRR | open_domain MRR | single_hop | multi_hop | temporal | adversarial | Δoverall | Δopen | Max drop | Overall 95% CI | Open 95% CI | Win/Loss/Tie (overall) | Target |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| baseline_v16 | baseline | 0.708 | 0.381 | 0.740 | 0.642 | 0.737 | 0.738 | +0.000 | +0.000 | 0.000 | +0.000 to +0.000 | +0.000 to +0.000 | 0/0/1986 | fail |
| enrichment_only | ablation | 0.709 | 0.382 | 0.742 | 0.642 | 0.738 | 0.738 | +0.001 | +0.000 | -0.000 | -0.000 to +0.002 | +0.000 to +0.001 | 15/11/1960 | fail |
| profiles_beliefs_only | ablation | 0.708 | 0.381 | 0.740 | 0.642 | 0.737 | 0.738 | +0.000 | +0.000 | 0.000 | +0.000 to +0.000 | +0.000 to +0.000 | 0/0/1986 | fail |
| fusion_convex_only | ablation | 0.708 | 0.381 | 0.740 | 0.642 | 0.737 | 0.738 | +0.000 | +0.000 | 0.000 | +0.000 to +0.000 | +0.000 to +0.000 | 0/0/1986 | fail |
| gte_reranker_only | ablation | 0.679 | 0.360 | 0.741 | 0.638 | 0.710 | 0.637 | -0.029 | -0.022 | 0.101 | -0.043 to -0.014 | -0.099 to +0.053 | 337/397/1252 | fail |
| nli_only | ablation | 0.706 | 0.368 | 0.740 | 0.633 | 0.734 | 0.739 | -0.002 | -0.014 | 0.014 | -0.004 to -0.000 | -0.040 to +0.003 | 8/12/1966 | fail |
| multi_passage_only | ablation | 0.708 | 0.381 | 0.740 | 0.642 | 0.737 | 0.738 | -0.000 | +0.000 | 0.000 | -0.000 to +0.000 | +0.000 to +0.000 | 0/2/1984 | fail |
| comparison_dual_search_only | ablation | 0.708 | 0.381 | 0.740 | 0.642 | 0.737 | 0.738 | +0.000 | -0.000 | 0.000 | -0.000 to +0.000 | -0.000 to +0.000 | 2/1/1983 | fail |
| combined_non_regressing | combined | 0.708 | 0.381 | 0.740 | 0.642 | 0.737 | 0.738 | +0.000 | -0.000 | 0.000 | -0.000 to +0.000 | -0.000 to +0.000 | 2/1/1983 | fail |

Combined decision: `combined_non_regressing` uses every phase-1 feature with non-negative overall MRR delta. Targets not met.

