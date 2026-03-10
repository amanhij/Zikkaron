# Codex Implementation Brief — LoCoMo Retrieval Optimization
## Status: All techniques implemented but REGRESSED. Need statistical tuning.

---

## CURRENT RESULTS (2026-03-09)

### Baseline (before v17 changes):
| Category | MRR | R@10 | n |
|---|---|---|---|
| single_hop | 0.757 | 0.906 | 841 |
| multi_hop | 0.646 | 0.862 | 282 |
| temporal | 0.712 | 0.850 | 321 |
| open_domain | 0.381 | 0.625 | 96 |
| adversarial | 0.665 | 0.854 | 446 |
| **overall** | **0.708** | **0.868** | **1986** |

### After v17 (all new features ON, convex fusion + GTE-Reranker):
| Category | MRR | R@10 | n | Delta |
|---|---|---|---|---|
| single_hop | 0.739 | 0.894 | 841 | **-0.018** |
| multi_hop | 0.634 | 0.862 | 282 | **-0.012** |
| temporal | 0.708 | 0.850 | 321 | -0.004 |
| open_domain | 0.351 | 0.625 | 96 | **-0.030** |
| adversarial | 0.637 | 0.827 | 446 | **-0.028** |
| **overall** | **0.678** | **0.854** | **1986** | **-0.030** |

### TARGETS:
- open_domain MRR: 0.50+ (ideally 0.55+)
- overall MRR: 0.72+
- No category may drop more than 0.02

---

## WHAT WAS IMPLEMENTED (lockstep-verified, all tests pass)

### Spec 01: Index-Time Enrichment (`zikkaron/enrichment.py`)
- **ConceptNetExpander**: Hardcoded fallback (HTTP API too slow). Maps activities to implied categories (camping→national_park, outdoor_activity)
- **LogicExpander**: Hypernym lifting (Yellowstone→national_park), verb nominalization
- **CometInferencer**: COMET-BART xAttr/xIntent/xWant (DISABLED in benchmark — too slow, downloads 1.5GB)
- **Doc2QueryExpander**: Synthetic questions (DISABLED — too slow)
- **FPAFilter**: Cosine similarity noise filter (threshold 0.25)
- **EnrichmentPipeline**: Orchestrates all, stores in `enriched_content` column, updates FTS

### Spec 02: Profiles + Fusion (`zikkaron/profiles.py`)
- **ProfileExtractor**: Rule-based NER → (entity, attribute_type, key, value) stored in `user_profiles` table
- **BeliefDeriver**: Aggregate profiles into derived beliefs (activity category grouping)
- **Convex Combination fusion**: Min-max normalize + weighted sum (alternative to WRRF)
- **Comparison dual-search**: Split "A or B?" queries into separate searches

### Spec 03: Advanced Reranking (`zikkaron/retrieval.py`)
- **GTE-Reranker-ModernBERT-base** (149M): Primary CE with FlashRank fallback
- **NLI entailment scoring**: cross-encoder/nli-deberta-v3-base, question→statement conversion, P(entailment) blended with CE score
- **Multi-passage clustering**: Jaccard overlap → cluster → re-score concatenated evidence
- **Dual-vector architecture**: Schema prep only (DUAL_VECTORS_ENABLED=False)

---

## WHY IT REGRESSED — ROOT CAUSE ANALYSIS

1. **Convex fusion vs WRRF**: Convex combination uses raw score magnitudes which are unstable across signal types. WRRF uses ranks which are more robust. The convex fusion needs careful per-signal normalization calibration.

2. **GTE-Reranker vs FlashRank**: GTE-Reranker (149M) may be overfitting to its training distribution. FlashRank's ms-marco-MiniLM was already well-calibrated for our use case. GTE needs fine-tuning or score calibration.

3. **NLI score blending**: Blending NLI entailment (0.3 weight) with CE scores may be diluting the CE signal for non-open-domain queries. The NLI model's 3-class output (contradiction/neutral/entailment) needs careful probability calibration.

4. **Multi-passage clustering**: Jaccard overlap at 0.3 threshold may be over-clustering, and the boost formula may be too aggressive.

5. **Enrichment terms in FTS**: Adding enrichment terms like "outdoor_activity | national_park" to FTS content may be DILUTING exact-match signals for single_hop queries.

---

## WHAT NEEDS TO HAPPEN (STATISTICAL APPROACH)

### Phase 1: Ablation Study (CRITICAL)
Run benchmark with each feature toggled independently. Need a 2^k factorial design or at minimum one-at-a-time:
- Baseline (v16, no new features)
- +enrichment only
- +profiles only
- +convex fusion only (vs WRRF)
- +GTE-Reranker only (vs FlashRank)
- +NLI only
- +multi-passage only
- +comparison routing only

This tells us which features HELP and which HURT.

### Phase 2: Weight/Threshold Optimization
For features that help, grid search over:
- `NLI_WEIGHT`: [0.05, 0.1, 0.15, 0.2, 0.3]
- `CROSS_ENCODER_WEIGHT`: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
- `FPA_SIMILARITY_THRESHOLD`: [0.15, 0.2, 0.25, 0.3, 0.4]
- `MULTI_PASSAGE_CLUSTER_OVERLAP_THRESHOLD`: [0.2, 0.3, 0.4, 0.5]
- `PROFILE_SEARCH_WEIGHT`: [0.3, 0.5, 0.8, 1.0]
- `BELIEF_HIGH_CONFIDENCE_BOOST`: [1.0, 1.2, 1.5, 2.0]

Use Conv 1 (199 QA pairs, 13 open_domain) as dev set. Validate on full 10-conv.

### Phase 3: Per-Category Feature Routing
Different features should activate for different query categories:
- **single_hop**: Baseline WRRF + FlashRank (don't touch what works)
- **open_domain**: Enrichment + NLI + Profile/Belief search + comparison routing
- **multi_hop**: Multi-passage clustering + enrichment
- **temporal**: Keep existing temporal boost
- **adversarial**: Adversarial detection (already implemented)

### Phase 4: Enable Heavy Models
Once lightweight features are tuned:
- Enable COMET-BART (run enrichment offline, one-time cost)
- Enable Doc2Query (same)
- These add the richest implied facts but need GPU + time

---

## KEY RESEARCH INSIGHTS (from 42 papers, see RESEARCH_FINDINGS.md)

1. **CE is "semantic BM25"**: ms-marco-MiniLM computes soft TF+IDF (0.84 Pearson with BM25). Cannot do inference. ALL inference must happen BEFORE the CE stage.

2. **Cue-trigger semantic disconnect**: Query has zero lexical/semantic overlap with stored evidence despite implying the answer. This is THE fundamental problem for open_domain.

3. **IMPLIRET benchmark**: Best retrievers get 14.91% nDCG@10 on implicit facts. This is an UNSOLVED problem — our 0.381 open_domain MRR is actually competitive.

4. **Convex > RRF proven only with calibrated scores**: Bruch et al. (ACM TOIS 2023) showed convex beats RRF when scores are properly normalized. Our score distributions may not satisfy this.

5. **NLI entailment for retrieval**: Standard NLI models score 50% (random) on implied entailments vs 90%+ on explicit. The NLI model helps for EXPLICIT entailment but NOT for implicit inference.

6. **DualCSE is highest potential**: Two embeddings per sentence (explicit + implicit), +0.10-0.15 MRR. But requires model training — not a config change.

---

## FILE MAP

- `benchmarks/RESEARCH_FINDINGS.md` — Full 565-line research compilation (42 papers)
- `benchmarks/run_benchmark_gpu.py` — GPU benchmark runner with logging
- `benchmarks/test_e_locomo.py` — LoCoMo benchmark harness
- `zikkaron/enrichment.py` — Index-time enrichment pipeline (NEW)
- `zikkaron/profiles.py` — Structured profiles + derived beliefs (NEW)
- `zikkaron/retrieval.py` — Core retrieval (MODIFIED: convex fusion, GTE, NLI, multi-passage, dual-vector, comparison routing, profile/belief search)
- `zikkaron/storage.py` — Storage engine (MODIFIED: enrichment columns, profile/belief tables, FTS triggers)
- `zikkaron/config.py` — Settings v17-v25 (MODIFIED)
- `zikkaron/embeddings.py` — Embedding engine (MODIFIED: encode_document_enriched)
- `lockstep-specs/v3/*.yml` — Lockstep specs used to implement everything

## BENCHMARK COMMAND

```bash
# Run with current settings:
python3 benchmarks/run_benchmark_gpu.py

# Settings are in benchmarks/test_e_locomo.py::_make_settings()
# Toggle features there and re-run
```

## CRITICAL NOTE
The implementations are structurally correct (all lockstep validators passed, 158+ tests pass). The problem is TUNING — wrong weights, wrong feature combinations, score distribution mismatches. This needs systematic ablation, not code changes.
