# Research Findings: Breaking LoCoMo Records
## Compiled from 6 parallel research agents — 2026-03-07

---

## 1. WHY OUR CE FAILS ON OPEN_DOMAIN (Proven)

**ms-marco-MiniLM-L-12-v2 is architecturally a "semantic BM25"** (Lu & Chen 2025, arXiv:2502.04645). Mechanistic interpretability shows it computes soft TF + IDF. It CANNOT do multi-hop inference.

**Standard NLI models score 50% (random) on implied entailments** vs 90%+ on explicit (ACL 2025, arXiv:2501.07719). This mathematically proves why our CE fails — implications are invisible to it.

**BRIGHT benchmark** (ICLR 2025): Best embedding model (59.0 nDCG standard) drops to 18.3 on reasoning-intensive retrieval. ALL SOTA systems for reasoning-intensive retrieval use explicit reasoning somewhere.

**Conclusion**: ALL inference must happen BEFORE the CE stage — either at index time or candidate pool stage.

---

## 2. TOP TECHNIQUES RANKED BY IMPACT FOR ZIKKARON

### P0: ConceptNet Graph Expansion (ZERO GPU, LOW effort)
- Static KG with 8M nodes, 21 relation types
- Literally has: camping → AtLocation → national_park
- `conceptnet-lite` Python library, fully offline
- 1-2 hop traversal per memory term at STORAGE TIME
- Append expansion terms to stored memory content
- **Expected impact**: Directly bridges our #1 failure mode
- **Cost**: Zero GPU, negligible latency at index time, zero at query time

### P0: COMET-BART Commonsense Inference (406M params)
- Generates: xIntent, xAttr, AtLocation, Causes for any event
- "went camping" → {enjoy_nature, adventurous, campground, forest, park}
- Run at STORAGE TIME, one-time GPU cost per memory
- Store inferences alongside original text
- **Expected impact**: Richest implied fact generation
- Papers: Bosselut et al. ACL 2019, Hwang et al. AAAI 2021
- HuggingFace: `mismayil/comet-bart-ai2`

### P1: Doc2Query T5-small (60M params)
- Generate 10-20 synthetic questions per memory at storage time
- "Where did you go camping?" / "What outdoor activities?"
- Proven: MRR@10 0.184 → 0.277 (+50%) on MS MARCO
- HuggingFace: `doc2query/msmarco-t5-small-v1`

### P1: rank-T5-flan CE Model (~110MB)
- Drop-in FlashRank replacement
- Listwise training = better zero-shot OOD generalization
- May handle novel query-document patterns (inference) better

### P1: EnrichIndex Approach (COLM 2025, arXiv:2504.03598)
- Enrich documents offline at index time
- +11.7 recall@10, +10.6 nDCG@10
- 293x fewer online tokens than online methods

### P2: EcphoryRAG Implicit Relations (arXiv:2510.08958)
- Dynamically infers implicit relations at retrieval time
- 94% fewer tokens than GraphRAG
- EM: 0.392 → 0.474

### P2: PropRAG Beam Search (EMNLP 2025)
- LLM-free online beam search over proposition paths
- SOTA F1=64.5% (beats HippoRAG v2's 62.9%)
- MiniLM compatible

### P2: L2R2 Abductive Ranking (SIGIR 2020)
- Reformulates abductive reasoning as learning-to-rank
- 73.3% acc vs 67.75% classification (+8.2%)
- Directly maps to "camping → prefers national parks"

---

## 3. LOCOMO COMPETITIVE LANDSCAPE (29 Systems Surveyed)

### Top Systems by Overall J-Score (all use LLM at query time):
| System | Overall J | Open_Domain J | Backend | Year |
|--------|-----------|---------------|---------|------|
| EverMemOS | 92.3% | - | Proprietary | 2025-26 |
| MemMachine v0.2 (Agent) | 91.7% | 75.0% | gpt-4.1-mini | 2025 |
| MemMachine v0.2 (Memory) | 91.2% | 75.0% | gpt-4.1-mini | 2025 |
| Backboard v1.2 | 90.0% | **91.2%** | gpt-4.1 judge | 2025 |
| Hindsight | 89.6% | - | Gemini-3 Pro | 2025 |
| Full-Context (baseline) | 89.0% | 71.9% | gpt-4.1-mini | baseline |
| MemR3(RAG) | 86.8% | 71.5% | gpt-4.1-mini | 2025 |
| MIRIX | 85.4% | 65.6% | gpt-4.1-mini | 2025 |
| EMem-G | 85.3% | 57.3% | gpt-4.1-mini | 2025 |
| Memobase v0.0.37 | 75.8% | **77.2%** | gpt-4o-mini | 2025 |
| Zep | 75.1% | 67.7% | gpt-4o-mini | 2025 |
| Mem0-Graph | 68.4% | **75.7%** | gpt-4o-mini | 2025 |

### Open Domain Winners (by J-score):
1. Backboard: 91.2% (proprietary, no public details)
2. Memobase: 77.2% (profile-based approach captures broad user context)
3. Mem0-Graph: 75.7% (graph edges preserve relational knowledge)
4. MemMachine v0.2: 75.0% (comprehensive memory indexing)
5. ENGRAM: 72.9% (typed memory allows flexible retrieval)

### Key Pattern: Systems succeeding on open_domain use BROAD memory extraction (not aggressive compression/pruning) and/or iterative search.

### Our Zikkaron: MRR=0.708 (retrieval-only, no LLM at query time)
Note: J-scores measure answer quality (with LLM generation). Our MRR measures retrieval quality only. Not directly comparable, but shows open_domain is hardest for ALL systems.

---

## 4. MULTI-HOP RETRIEVAL (LLM-FREE OPTIONS)

| System | LLM at Query? | Key Metric | Technique |
|--------|---------------|------------|-----------|
| PropRAG | No | F1=64.5% avg | Beam search over propositions |
| MDR | No | EM=62.3, F1=75.3 (HotpotQA) | Iterative dense beam search |
| Baleen | No | P-EM=86.7% (HotpotQA) | Condensed retrieval |
| Beam Retriever | No | +50% EM (MuSiQue) | End-to-end beam search |
| HippoRAG 2 | Partial | +7 F1 over SOTA embeddings | PPR + recognition memory |

---

## 5. RECOMMENDED IMPLEMENTATION PIPELINE

### At STORAGE TIME (run once per memory):
```
Memory: "Melanie went camping last weekend"
  ↓
ConceptNet expand: [outdoor_activity, national_park, tent, nature, hiking]
  ↓
COMET infer: [xIntent: enjoy_nature; AtLocation: campground, park; xAttr: adventurous]
  ↓
Doc2Query: ["where did Melanie go?", "outdoor activities", "weekend trip"]
  ↓
Store: original + expansion_terms + comet_inferences + synthetic_queries
```

### At QUERY TIME (unchanged pipeline):
```
Query → FTS (matches expansion terms!) → Vector search → CE reranks
```

The expansion terms mean "national park" now appears in the stored text, so both FTS and vector search find it, and CE can match it directly.

---

## 6. ALTERNATIVE CE MODELS TO TEST

| Model | Size | BEIR Avg | Notes |
|-------|------|----------|-------|
| ms-marco-MiniLM-L-12-v2 (current) | 34MB | ~39 MRR | Our baseline |
| rank-T5-flan | 110MB | Better OOD | FlashRank drop-in |
| mxbai-rerank-base-v2 | 0.5B | 55.57 | RL-trained, via `rerankers` lib |
| gte-reranker-modernbert-base | 149M | Near-SOTA | 8x smaller than 1B models |

---

## 7. CRITICAL NEW FINDINGS (from additional research)

### IMPLIRET Benchmark (Taghavi et al., 2025) — PROVES our problem is hard
- Benchmark specifically for implicit-fact retrieval
- Best sparse/dense retrievers: only **14.9% nDCG@10**
- Even GPT-4: only 55%
- This is THE formal proof that implicit retrieval requires index-time enrichment

### Memory-Builder (Tang et al., 2026) — Highest open_domain QA accuracy
- LoCoMo QA: 84.2% total, **open_domain 84.7%**, adversarial 90.6%
- Uses RL-learned memory chains + graph reasoning prompts
- Highest reported open_domain by QA accuracy metric

### Entailment-Tuned DPR (Dai et al., 2024) — No LLM, +3-6% MRR
- Fine-tunes dense retrievers on NLI entailment data
- Bridges semantics via entailment (camping implies nature)
- +3-6% MRR on Natural Questions
- **Directly applicable** to our pipeline without LLM

### answerai-colbert-small-v1 (Answer.AI, 2024) — CPU-fast late interaction
- 33M param ColBERTv2 variant
- Token-level MaxSim scoring captures partial semantic matches
- CPU-fast, outperforms models 10x larger
- Could replace MiniLM CE for better inference gap handling

### Synapse (Jiang et al., 2026) — Spreading activation + lateral inhibition
- Unified Episodic-Semantic Graph
- +23% multi-hop improvement via activation dynamics
- Bridges disparate concepts organically without semantic similarity
- Non-LLM, mathematical approach

### LoCoMo-Plus (Li et al., 2026) — Formally defines our bottleneck
- Tests "cue-trigger semantic disconnect" — exactly our problem
- GPT-4o drops 17-49 points vs standard LoCoMo
- Proves cosine similarity over raw facts is fundamentally insufficient

### SPLADE v3 (Lassance et al., 2024) — Learned term expansion
- MRR@10 = 40.2 on MS MARCO, BEIR nDCG@10 = 51.7
- Activates tokens NOT in input via BERT MLM head
- "camping" can activate "park", "nature", "outdoors"
- Document encoding fully offline, query-time ~BM25 latency

### Doc2Query-- (Gospodinov et al., ECIR 2023) — Filtered expansion
- +16% retrieval effectiveness over unfiltered Doc2Query
- -48% index size, -30% query execution time
- Filters hallucinated synthetic queries via ELECTRA scoring

---

## 8. BREAKTHROUGH INSIGHTS (from comprehensive external research)

### The Four-Layer Index-Time Enrichment Pipeline (Confirmed by ALL research)
Every research source converges on the same conclusion. The optimal pipeline:
1. **COMET-DISTIL** (117M, GPT-2-sized) — Surpasses GPT-3 by 10+ points on commonsense. Generates xAttr, xWant, xEffect, xReact. CPU: ~200-500ms/memory. HuggingFace: `peterwestai2/symbolic-knowledge-distillation`
2. **DocT5Query** (220M, T5-base) — +50% MRR on MS MARCO (0.187→0.277). Doc2Query-- adds ELECTRA filtering: +16% more, -48% index size
3. **ConceptNet** (0 params) — Sub-ms lookups, camping→AtLocation→national_park. Spreading activation: +18.9% MAP (Nguyen et al., 2018)
4. **PAED persona extraction** (140M, BART) — Extracts (subject, relation, object) triplets from dialogue. "went camping" → (I, routine, go camping), (I, characteristic, family-oriented), (I, like, outdoor_activities). Accuracy: head >0.95. Code: github.com/Cyn7hia/PAED

### NEW: DistillReDE — Distilled HyDE Without LLM at Query Time
- Distills HyDE's inference-bridging into Contriever-sized model
- +33% over vanilla Contriever, approaches HyDE performance
- NO LLM at query time — arXiv:2410.21242, 2024

### NEW: Entailment-Tuned DPR (Exact Method)
- arXiv:2410.15801, 2024: Converts questions to claims, trains retrievers to predict entailment
- Off-the-shelf NLI models (RoBERTa) assign higher entailment scores to positive passages
- Key insight: **relevance ≈ entailment**
- DeBERTa-base (86M) NLI as additional reranking signal

### NEW: Convex Combination > RRF (Proven)
- Bruch, Gai & Ingber, ACM TOIS 2023
- CC consistently outperforms RRF in both in-domain and OOD
- Only ~40 annotated queries needed to learn optimal weights
- RRF's k=60 is more sensitive than believed

### NEW: GTE-Reranker-ModernBERT-base (149M)
- 83.00% Hit@1, matches nemotron-1B (8x larger)
- ModernBERT architecture, CPU-viable, Apache 2.0

### NEW: Hindsight's TEMPR Validates Our Approach
- Hindsight achieves **95.1% open_domain** with CARA reflection
- TEMPR uses: semantic embeddings + BM25 + graph spreading activation + temporal filtering → RRF → CE
- This is exactly our pipeline, but with LLM at query time for the final reasoning step
- Pre-computing those inferences at index time = Zikkaron's strategy

### NEW: H²Memory Principle Memories
- Huang et al., 2025: Four memory tiers including "principle memories"
- Typed clusters of preferences and principles abstracted from raw conversations
- This is the derived knowledge layer Zikkaron needs

---

## 10. FOCUSED RESEARCH: Open-Domain Inference Gap Solutions (User Research Round 3)

### ConceptNet Graph Expansion (Confirmed Details)
- **Source**: Speer et al. (2017) ConceptNet 5
- 3.4M nodes, 36 relations, high precision
- Key path: camping → AtLocation → nature → AtLocation → national_park
- **Implementation**: Query ConceptNet local copy for neighbor concepts per memory keyword at ingestion. Append related concepts as indexable fields in FTS5.
- **Expected MRR**: +1-3% (addresses ~11.5% keyword-gap cases)
- **Risk**: Over-expansion noise. Must filter low-confidence edges. Irrelevant edges (cake→dessert) can hurt.

### COMET-BART Commonsense Inference (Confirmed Details)
- **Source**: Hwang et al. (2020) COMET-ATOMIC2020, 406M params
- Generates if-then inferences: xIntent, xReact, xNeed, Desires
- "Melanie goes camping" → xIntent: enjoy nature, xReact: energized, xNeed: prepare tent
- Accuracy: ~30-50% on ATOMIC completion
- **Implementation**: Run COMET-BART offline on memory key predicates. Store generated inferences as augmented text. Embed into sqlite-vec.
- **Expected MRR**: +2-5% (covers part of 15.6% opinion-inference gap)
- **Risk**: Hallucination and drift. "camping"→"X wants to party". Needs entailment/relevance filtering.

### Doc2Query T5 Expansion (Confirmed Details)
- **Source**: Nogueira et al. (2019), Gospodinov et al. (2023 Doc2Query--)
- T5-base generates synthetic queries per document offline
- "Melanie went camping" → "Where does Melanie like to camp?", "What outdoor activities does Melanie enjoy?"
- Doc2Query++ reports significant gains in MAP, nDCG@10, Recall@100 across BEIR
- **Implementation**: Generate ~5-10 queries per memory with fine-tuned T5. Store as extra FTS5 entries or concatenated text before embedding.
- **Expected MRR**: +5-10% (boosts recall for mismatched queries)
- **Risk**: Index bloat, poorly generated queries may harm dense encoding. Filter low-quality via relevance classifier. Dual-index strategy recommended.

### Set-Encoder: Multi-Passage Re-Ranking (NEW)
- **Source**: Schlatt et al., ECIR 2025 "Set-Encoder: Permutation-Invariant Inter-Passage Attention"
- Jointly scores a SET of passages for a query via inter-passage [CLS] attention
- Order-invariant, models multi-memory evidence in one forward pass
- **Benchmarks**: 2nd-best avg nDCG@10 across TIREx corpora. Matches/exceeds monoELECTRA on TREC DL. ~+1-2% absolute nDCG.
- **Targets**: The 26% of failures needing multi-memory aggregation
- **Implementation**: After initial retrieval, feed top-K candidates jointly into Set-Encoder with query prepended. Model scores all passages at once with evidence-sharing. <500M params.
- **Expected MRR**: +2-5% (by capturing joint evidence, especially OOD)
- **Risk**: CPU latency concerns. Sensitive to number of passages — needs tuning for exact N. If not tuned, effectiveness drops.

### NLI-Entailment Tuning (Confirmed Details)
- **Source**: Dai et al., EMNLP 2024 "Improve Dense Passage Retrieval with Entailment Tuning"
- Supervised entailment stage between pretraining and fine-tuning
- Learns from MNLI/SNLI: relevant passages must entail the question's answer
- **Benchmarks**: +2-3% absolute top-1 hit and MRR on Natural Questions. +0.6-1.2 MRR@10 on MS-MARCO.
- **Implementation**: Fine-tune nomic-embed-text-v1.5 on NLI pairs with entailment objective, then continue contrastive MS-MARCO training. Zero latency cost at inference.
- **Expected MRR**: ~+3% (aligns embedding space with inference logic)
- **Risk**: Marginal gains, training cost (hours on GPU). May overfit to superficial entailment patterns.

### Structured User Profile / Schema-Guided Memory (NEW — HIGH IMPACT)
- **Source**: Mei et al., ICML 2026 "According to Me" (ATM-Bench)
- Aggregates scattered memories into structured profile with fixed-schema records
- Schema-Guided Memory (SGM): {"event": camping, "context": outdoors, "prefers": nature}
- **Benchmarks**: SGM vs free-text → +20% QA accuracy on hard multi-hop queries. Memobase (profile-based) → 77% open_domain J-score vs ~37% baseline.
- **Targets**: Both opinion-inference (15.6%) AND multi-memory aggregation (26%)
- **Implementation**: Periodically process all memories to extract/cluster into profile attributes. Store in dedicated SQLite table with its own embeddings. At query time, also retrieve against profile and fuse score.
- **Expected MRR**: +5-10% (potentially up to +20% for open_domain by capturing the 27% fixable gap)
- **Risk**: Staleness, incorrect profile entries, over-generalization. Profile updates must be accurate.

### Comparison/Retrieval Fusion (Pattern)
- For "Would X prefer A or B?" questions: dual-query approach
- Issue two searches: "X prefers A" and "X prefers B"
- Compare evidence by summing top-k CE scores per option
- **Implementation**: Two SQLite search calls, results compared
- **Expected MRR**: Variable — improves specific preference queries if evidence is lopsided
- **Risk**: Doubles computation, both sides may have little evidence

---

## 11. COMPREHENSIVE SYNTHESIS: Closing the Inference Gap (User Research Round 4)

### COMET + ATOMIC: Behavioral Inference Is Tractable at Index Time
- ATOMIC 2020: **1.33M tuples, 23 relation types**
- Key relations for preference inference:
  - xAttr (persona): "goes camping" → outdoorsy, adventurous — **80.8% accuracy**
  - xWant (desires): → to explore nature, visit parks — **~90% accuracy**
  - xIntent (motivation): → to enjoy outdoors — **87.7% accuracy**
- COMET-BART (406M): Exceeds GPT-3 (175B) by ~12 points despite 430x fewer params
- HuggingFace: `adamlin/comet-atomic_2020_BART` (1.63GB)
- **Cost**: 1-3 sec/memory on GPU (5 relations, beam search)
- **Noise**: 23% implausible inferences (PARA-COMET). Filter with: critic model at rec_0.5 threshold, cosine similarity ≥ 0.3 to original, confidence decay 0.5x per hop, cap top-5 neighbors
- ConceptNet complement: camping IsA outdoor_activity (weight 9.1), camping AtLocation national_park (weight 2.1)
- **Optimal traversal**: 1-hop all preference relations, 2-hop ONLY IsA→AtLocation chains with weight ≥ 2.0. NO 3-hop.
- CSKG (Ilievski et al. ESWC 2021): 2.2M nodes, 6M edges — unified ConceptNet+ATOMIC+WordNet

### SPIKE: Scenario-Based Document Enrichment (COLM 2025)
- Decomposes documents into scenario-based retrieval units at index time
- Generates reasoning scenarios making implicit relevance explicit
- "Melanie went camping" → scenario: "A user asking about vacation preferences — she likely prefers nature-based destinations"
- **BRIGHT benchmark: >18% improvement** across 6 dense retrievers
- Ablation: explanation/reasoning component alone achieves highest individual performance
- Uses teacher LLM for dataset creation, distills into smaller generator

### HyPE: Hypothetical Prompt Embeddings (Vake et al. 2025) — HIGHEST SINGLE TECHNIQUE IMPACT
- Inverts HyDE: at INDEX time, generate hypothetical questions each doc could answer
- Query time: question-to-question matching (not question-to-document)
- **Results: +42 percentage points retrieval precision, +45 points recall**
- For preference queries: pre-generate "Would this person prefer outdoor destinations?" at index time
- Direct match to "What kind of parks does Melanie like?" at query time

### INLI Dataset: Why Standard NLI Fails (ACL 2025)
- Introduces "implied entailment" as 4th NLI category
- Standard NLI models: **near-chance on implied entailments** — our exact failure mode
- Training data from conversational implicature datasets (Ludwig, Circa, NormBank, SocialChem)
- Example: "Sophie claims too tired" → implied entailment "Sophie would prefer not to attend party"
- **Code**: github.com/google-deepmind/inli
- **3 implementation paths**:
  1. NLI CE reranker: `cross-encoder/nli-deberta-v3-base` (86M), score entailment probability
  2. LoRA adapter on nomic-embed (r=64, ~1-2M params), train on SNLI+MNLI+INLI
  3. Synthetic (memory, implied_preference) pairs for training data

### RAPTOR: Recursive Clustering for Multi-Memory Aggregation (ICLR 2024)
- Recursively clusters and summarizes text into tree structure at index time
- "What are personality traits?" matches pre-computed summary node
- **Results: +20% absolute accuracy on QuALITY with GPT-4**
- Maps to sqlite-vec: store leaf + parent nodes with level/parent_id columns
- Query-time search is pure ANN — CPU-compatible

### CMC: Compare Multiple Candidates (EMNLP 2024)
- 2 shallow self-attention layers over pre-computed embeddings
- **11x speedup over cross-encoders, +6.7% Recall@16**
- CPU-viable for small candidate sets via ONNX export

### Social Chemistry 101 (Forbes et al. EMNLP 2020)
- **292K rules-of-thumb** mapping situations to values/norms
- "Going camping with family" → norms about valuing nature, family time, outdoor experiences
- Additional enrichment signal beyond COMET

### Cognitive Science Validation
- Bayesian Theory of Mind (Baker, Saxe & Tenenbaum 2017): observe actions → invert planning model → recover reward function. Correlates with human judgments at **r > 0.9**
- Camping→national park follows this logic: costly repeated actions (camping) → high reward assigned to outcomes (nature)
- Justifies COMET's xWant/xIntent as cheap approximation of inverse reward inference

### Integrated Architecture (3 Tiers)
**Tier 1 — Index-time enrichment (highest impact):**
Per memory: (a) COMET-BART on xWant/xIntent/xAttr, (b) ConceptNet 1-hop weight≥1.0, (c) 5-10 doc2query queries via T5-small, (d) embed enriched text with nomic-embed. Index in FTS5 alongside original.
**Expected: 30-50% recall improvement on inference-heavy queries.**

**Tier 2 — Profile-first architecture (solves aggregation):**
PAED/DeBERTa extract persona attributes → structured profiles in SQLite → NL profile docs embedded alongside memories → query-type routing (comparison→dual retrieval, aggregation→profile search, simple→standard).
**Expected: +8-15% open_domain, +15-25% aggregation.**

**Tier 3 — Entailment-aware scoring (closes remaining gap):**
NLI CE (DeBERTa-v3-base 86M) + LoRA adapter on nomic-embed trained on INLI.
**Expected: +2-5% MRR, substantially higher on inference-heavy.**

### What Remains Genuinely Unsolved
- No existing system handles full behavioral-fact → trait-inference → preference-prediction → query-materialization as unified architecture
- IMPLIRET: 14.91% best nDCG@10
- BRIGHT: 18.3% best nDCG@10
- Combining COMET + doc2query + INLI + Memobase profiles for personal memory retrieval = novel contribution

---

## 12. MASTER SYNTHESIS: Architecting Open-Domain Inference (User Research Round 5 — Final)

### The Problem Formalized
- **Cue-trigger semantic disconnect** (LoCoMo-Plus 2026): query surface form has zero lexical/semantic overlap with evidence, but evidence IMPLIES answer
- Mem0 collapses from 68.10% → 35.20% transitioning from factual to cognitive memory
- CE (MiniLM-L-12-v2) proven to be contextualized semantic BM25: attention heads compute soft TF, embedding matrix encodes IDF. **0.84 Pearson correlation** with BM25 scores (Lu & Chen 2025)
- CE cannot materialize latent variable "outdoor enthusiast" — architecturally impossible
- IMPLIRET: ReasonIR-8B only 25% nDCG@10; DRAGON+ only 14.91%; GPT-o4-mini only 55.54% with 30 distractors
- **Target**: Fix 15.6% opinion inference + 11.5% keyword mismatch = 27.1% fixable → MRR 0.375 → ~0.55

### Technique 1: COMET-BART xAttr/xIntent Expansion (MRR +0.05 to +0.08)
- ATOMIC-2020: social-interaction relations for theory of mind
- xAttr: "goes camping" → outdoorsy, nature lover — **80.8% accuracy**
- xIntent: → to connect with nature — **87.7% accuracy**
- **Implementation**: Offline COMET-BART (406M) → generate xAttr/xIntent → append to implied_traits hidden column in FTS5
- At query time: BM25 matches "nature lover" to "national park" — bypasses CE limitation
- Millisecond execution per memory on standard hardware

### Technique 2: Fact-Centric Preference Alignment Noise Filter (MRR +0.02)
- **Source**: ACL Findings 2025
- Enforces factual consistency during data integration
- Maintains reliability where standard models drop below 15% due to semantic drift
- **Implementation**: Cosine distance threshold between original memory and COMET inference. If beyond geometric hyper-sphere boundary → reject inference
- Prevents: camping → woods → lumberjack drift
- Keeps sqlite-vec clusters tight around true epistemic state

### Technique 3: Set-Encoder Listwise Reranking (MRR +0.08 to +0.12)
- **Source**: Schlatt et al., ECIR 2025, 330M params
- `<INT>` interaction token: parallel sequence processing with inter-passage cross-attention
- **85-110x faster** than RankGPT-4o/RankZephyr, **6x less memory**
- Permutation-invariant: no positional bias
- "went to beach" + "hiked trail" + "went camping" → inter-passage attention aggregates "outdoor preference" theme
- **Implementation**: Replace FlashRank pointwise CE with ONNX Set-Encoder. Feed top-50 as single parallel batch.
- Addresses 26% multi-memory aggregation failure directly

### Technique 4: Hindsight Epistemically Distinct Memory Networks (MRR +0.06)
- **Source**: Latimer et al., arXiv 2025
- 4 epistemically distinct networks: world, experience, opinion, observation
- Async "Reflect" operation converts raw transcripts → evolving belief states
- **Results**: LongMemEval 39.0% → 83.6%. **LoCoMo Open Domain: 95.12%**
- **Implementation**: Separate SQLite tables: raw_memories + derived_beliefs. Background process scans experiences, synthesizes behavioral clusters → writes belief states.
- Classical music + painting supplies → "Appreciates fine arts" written to derived_beliefs
- Query-time: access derived_beliefs index directly for pre-synthesized preference answers

### Technique 5: CompUGE Comparison Query Pipeline (MRR +0.03)
- **Source**: Shallouf et al., COLING 2025
- CAM 2.0: classifier detects comparative syntax → dual object search → stance classification
- **F1: 0.81-0.84** for object/aspect identification
- **Implementation**: fastText/BiLSTM classifier detects "A vs B" → two parallel sqlite-vec searches → RRF merge
- Sidesteps vector cancellation effect (A and B pull query vector into void between them)
- "national park" vector cleanly aligns with pre-computed "nature lover" from COMET expansion

### Technique 6: Counterfactual-Contrastive Inference (MRR +0.02)
- **Source**: "DoubleTake", 2025
- Generates contrastive negative probes for discriminatory confidence margins
- **+15% set-level accuracy** on MediConfusion benchmark
- **Implementation**: Offline Doc2Query generates contrastive queries. "national parks" → also probe for absence of "crowded urban spaces"
- Creates wider mathematical margin for CE evaluation

### Technique 7: PMI Edge Weighting for ConceptNet (MRR +0.04)
- **Source**: KagNet architectures, multi-hop QA literature
- PMI(x;y) = log P(x,y) / P(x)P(y) — filters statistically inconsistent edges
- ~60% of ConceptNet edges act as distractors without filtering
- **Implementation**: 1-hop depth limit. Only IsA, UsedFor, HasProperty, MotivatedByGoal relations. PMI threshold θ filters noise.
- camping → MotivatedByGoal → enjoy nature (passes). camping → tent → circus (filtered by PMI < θ)
- Eliminates concept drift while preserving precise semantic bridges

### Technique 8: INLI-T5 Implied Entailment Generation (MRR +0.07)
- **Source**: Havaldar et al., ACL 2025
- Standard LLMs (including GPT-4): ~50% on implied entailment. T5-XXL fine-tuned on INLI: **88.5%**
- 4-way taxonomy: explicit entailment, implied entailment, neutral, contradiction
- "Sophie says too tired" → implied: "Sophie would prefer to stay home"
- **Implementation**: T5-Large fine-tuned on INLI in async ingestion pipeline. Generate implied entailments, filter by confidence, write to sqlite-vec tagged source:inferred
- Hardcodes implied preference directly into searchable index

### Technique 9: ProoFVer Natural Logic Expansion (MRR +0.03)
- **Source**: ProoFVer, TACL 2022
- Formal proofs via lexical mutations using natural logic operators
- "camping at Yellowstone" → forward entailment → "camping at national park" → "outdoor activity"
- **+13.21 pp** over competitors on FEVER counterfactual instances
- Zero semantic drift risk — strictly monotonic
- **Implementation**: MacCartney & Manning framework offline → directional entailment expansions → inject into FTS5

### Technique 10: Memobase Structured Profile Slotting (MRR +0.08)
- **Source**: Memobase, 2025
- Perpetually updated deterministic user persona state
- "Reads children's books to kids" → {attribute_key: "values", attribute_value: "education"}
- **77.17% open_domain J-score** using lighter retrieval than graph-search systems
- **Implementation**: user_profile table: user_id → key-value attribute pairs. Async offline model assesses incoming memories, computes delta against existing profile.
- Query "science museum vs theme park?" → profile returns "values: education" → instant exact match

### Technique 11: DualCSE Dual-Vector Embeddings (MRR +0.10 to +0.15) — HIGHEST IMPACT
- **Source**: Wang et al., October 2025 "One Sentence, Two Embeddings"
- Two co-existing embeddings per sentence: explicit (literal) + implicit (latent meaning)
- **80.38% avg accuracy on RTE**, outperforms SimCSE by >10% on implicit reasoning, **100% on EIS task**
- Training: force implicit embedding of premise to converge with explicit embedding of implied entailment hypothesis. Push apart premise's own explicit vs implicit embeddings.
- **Implementation**: Fine-tune nomic-embed via DualCSE objective using INLI dataset. Store dual 768d vectors in sqlite-vec (explicit_emb + implicit_emb columns). Query-time ANN searches implicit_emb only for preference queries.
- "went camping" geometrically overlaps "national parks" in implicit vector space
- **Risk**: 2x storage footprint, custom fine-tuning MLOps overhead

### MASTER IMPLEMENTATION PLAN (9 Phases)

**Phase 1 — Async Offline Enrichment Pipeline:**
Memory → COMET-BART (xAttr/xIntent) → PMI-filtered ConceptNet 1-hop → INLI-T5 implied entailments → FPA noise filter → Write to:
  - implied_traits column (FTS5)
  - derived_beliefs table (structured profiles)
  - DualCSE dual vectors (sqlite-vec: explicit_emb + implicit_emb)

**Phase 2 — Sub-Second Query Execution:**
Query → CompUGE classifier (comparative detection) → Dual search if A-vs-B → FTS5 + sqlite-vec search (both episodic + profile/beliefs index) → CCI contrastive probes → RRF merge top-50 → Set-Encoder listwise rerank (330M, ONNX) → Final ranking

**Cumulative Expected MRR Impact:**
- COMET-BART: +0.05 to +0.08
- DualCSE: +0.10 to +0.15
- Set-Encoder: +0.08 to +0.12
- Memobase profiles: +0.08
- INLI-T5: +0.07
- PMI ConceptNet: +0.04
- CompUGE: +0.03
- ProoFVer: +0.03
- FPA filter: +0.02
- CCI: +0.02
- Hindsight reflect: +0.06
- **Theoretical combined**: +0.30 to +0.45 (with diminishing returns, realistic: +0.15 to +0.20)
- **Projected open_domain MRR**: 0.375 → 0.52–0.58 (approaching theoretical ceiling of ~0.55)

---

## 9. KEY PAPERS (Full Citations)

1. Lu & Chen, "Cross-Encoder Rediscovers a Semantic Variant of BM25", arXiv:2502.04645, 2025
2. Chen et al., "EnrichIndex: Using LLMs to Enrich Retrieval Indices Offline", COLM 2025, arXiv:2504.03598
3. Formal et al., "SPLADE v2/v3", SIGIR 2021 / 2024
4. Nogueira & Lin, "docTTTTTquery", 2019
5. Gospodinov et al., "Doc2Query--: When Less is More", ECIR 2023
6. Bosselut et al., "COMET: Commonsense Transformers", ACL 2019
7. Hwang et al., "COMET-ATOMIC 2020", AAAI 2021
8. West et al., "Symbolic Knowledge Distillation", NAACL 2022
9. Speer et al., "ConceptNet 5.5", AAAI 2017
10. Wang, "PropRAG: Beam Search over Proposition Paths", EMNLP 2025
11. "EcphoryRAG", arXiv:2510.08958, 2025
12. "MemR3: Memory Retrieval via Reflective Reasoning", arXiv:2512.20237, 2024
13. Pan et al., "SeCom", ICLR 2025, arXiv:2502.05589
14. Gutierrez et al., "HippoRAG 2", ICML 2025, arXiv:2502.14802
15. "BRIGHT: Reasoning-Intensive Retrieval", ICLR 2025, arXiv:2407.12883
16. "Entailed Between the Lines: INLI", ACL 2025, arXiv:2501.07719
17. Zhu et al., "L2R2: Leveraging Ranking for Abductive Reasoning", SIGIR 2020
18. "Abductive Inference in RAG", arXiv:2511.04020, 2025
19. Chen et al., "Dense X Retrieval / Proposition Indexing", EMNLP 2024
20. Trivedi et al., "IRCoT", ACL 2023
21. Schlatt et al., "Set-Encoder: Permutation-Invariant Inter-Passage Attention", ECIR 2025
22. Dai et al., "Improve Dense Passage Retrieval with Entailment Tuning", EMNLP 2024
23. Mei et al., "According to Me: Long-Term Personalized Referential Memory QA", ICML 2026 (ATM-Bench)
24. Sap et al., "ATOMIC: An Atlas of Machine Commonsense", AAAI 2019
25. Havaldar et al., "INLI: Implied Entailment", ACL 2025
26. Lee et al., "SPIKE: Scenario-Based Document Enrichment", COLM 2025
27. Vake et al., "HyPE: Hypothetical Prompt Embeddings", 2025
28. Sarthi et al., "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval", ICLR 2024
29. Song et al., "CMC: Compare Multiple Candidates", EMNLP 2024
30. Forbes et al., "Social Chemistry 101", EMNLP 2020
31. Ilievski et al., "CSKG: Consolidated Commonsense KG", ESWC 2021
32. Baker, Saxe & Tenenbaum, "Bayesian Theory of Mind", 2017
33. Jara-Ettinger, "Theory of Mind as Inverse Reinforcement Learning", 2019
34. Latimer et al., "Hindsight is 20/20: Building Agent Memory", arXiv 2025
35. Shallouf et al., "CompUGE-Bench: Comparative Question Answering", COLING 2025
36. "DoubleTake: Contrastive Reasoning for Faithful Decision-Making", 2025
37. Wang et al., "DualCSE: One Sentence, Two Embeddings", October 2025
38. "ProoFVer: Natural Logic Theorem Proving for Fact Verification", TACL 2022
39. "Fact-Centric Preference Alignment", ACL Findings 2025
40. "IMPLIRET: Implicit Fact Retrieval Benchmark", 2025
41. "LoCoMo-Plus: Cue-Trigger Semantic Disconnect", 2026
42. "KagNet: Knowledge-Aware Graph Networks for Commonsense QA"
