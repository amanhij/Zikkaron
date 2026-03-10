# Lockstep Prompt: Index-Time Enrichment Pipeline

## Goal
Implement an offline index-time enrichment pipeline in Zikkaron's storage layer that automatically generates implied facts, commonsense inferences, and synthetic queries for every memory at ingestion time. This is the single highest-impact change for improving open_domain retrieval MRR from 0.381 to 0.50+.

## Why This Matters
Our cross-encoder (ms-marco-MiniLM-L-12-v2) is architecturally a "semantic BM25" (Lu & Chen 2025). It CANNOT do multi-hop inference. When a user stores "Melanie went camping" and later queries "Would Melanie prefer a national park?", the CE scores this at ~0.01 relevance because "camping" and "national park" have zero lexical overlap. The solution: materialize the implied facts ("nature lover", "outdoor enthusiast", "enjoys hiking") at storage time so they appear in the FTS5 index and vector space, allowing BM25 and cosine similarity to find them.

## Current Architecture (DO NOT BREAK)
- **Storage**: `zikkaron/storage.py` — `StorageEngine` class with SQLite backend
- **Tables**: `memories` (content, embedding, tags, heat, etc.), `memories_fts` (FTS5 on content), `memory_vectors` (sqlite-vec)
- **Insert flow**: `insert_memory(memory: dict) -> int` stores content, embeds via sentence-transformers, auto-syncs FTS5 and sqlite-vec via triggers
- **FTS enrichment**: `_enrich_content_for_fts()` currently only splits CamelCase/snake_case identifiers
- **Config**: `zikkaron/config.py` — `Settings` dataclass with all feature flags
- **Embeddings**: `zikkaron/embeddings.py` — `EmbeddingEngine` wrapping sentence-transformers (nomic-embed-text-v1.5, 768d)
- **Retrieval**: `zikkaron/retrieval.py` — `HippoRetriever` with WRRF fusion + FlashRank CE reranking

## Technique 1: ConceptNet 1-Hop Expansion (ZERO GPU)

### What
Query ConceptNet for related concepts for key terms in each memory. Append related terms to the memory's FTS5-indexed content.

### Implementation
- Install `conceptnet-lite` Python package (local SQLite dump, no API needed, ~2GB download)
- Alternative if conceptnet-lite unavailable: use `requests` to query `http://api.conceptnet.io/query?node=/c/en/{term}&rel=/r/{relation}&limit=10` (but prefer local)
- For each memory at insert time:
  1. Extract content-bearing nouns/verbs (skip stop words, pronouns, common verbs)
  2. For each extracted term, query ConceptNet with ONLY these relations: `IsA`, `UsedFor`, `HasProperty`, `AtLocation`, `MotivatedByGoal`, `CausesDesire`, `CapableOf`
  3. DO NOT use `RelatedTo` (too noisy) or traverse beyond 1 hop
  4. Filter edges by weight >= 1.0 (ConceptNet edge weights range 0-10+)
  5. Collect the target node labels (English only, /c/en/ prefix)
  6. Deduplicate and limit to top 10 expansion terms per memory
- Store expansion terms in a new column `enrichment_concepts` (TEXT) on the memories table
- Include expansion terms in the FTS5 content so BM25 can match them
- Example: "went camping" → ConceptNet yields: outdoor_activity (IsA, w=9.1), national_park (AtLocation, w=2.1), tent (UsedFor), nature (MotivatedByGoal)

### Risk Mitigation
- Cap at 10 expansion terms per memory to prevent index bloat
- Only use high-confidence edges (weight >= 1.0)
- Skip expansion for very short memories (<20 chars) — these are filler like "Sounds great!"

## Technique 2: COMET-BART Commonsense Inference (406M params)

### What
Run COMET-BART (trained on ATOMIC-2020) on each memory to generate implied personality traits, intentions, and desires. These are the psychological inferences that bridge "went camping" → "nature lover".

### Implementation
- Model: `mismayil/comet-bart-ai2` from HuggingFace (1.63GB, BART-large architecture)
- Alternative smaller: explore if a distilled version exists, otherwise use the full 406M
- For each memory at insert time:
  1. Extract the main behavioral predicate (the action/event described). Use simple heuristics: find sentences with named subjects + verbs
  2. For each predicate, generate inferences for these ATOMIC relations:
     - `xAttr` (personality traits of the subject): "went camping" → "outdoorsy", "adventurous" — 80.8% accuracy
     - `xIntent` (motivation): → "to enjoy nature", "to relax" — 87.7% accuracy
     - `xWant` (what subject wants next): → "to explore more", "to visit parks" — 90.0% accuracy
  3. Use beam search with num_beams=5, generate top 3 per relation
  4. Filter: discard inferences with confidence < 0.3 (softmax probability from beam search)
  5. Apply FPA noise filter (Technique 5 below): cosine similarity between original memory embedding and inference embedding must be >= 0.25. This prevents drift like camping → party
- Store inferences in a new column `enrichment_comet` (TEXT, JSON array) on the memories table
- Include key inference terms in FTS5 content
- Also embed the enriched content (original + inferences) for vector search

### GPU/CPU Handling
- At import time, check `torch.cuda.is_available()`. If GPU: use it for COMET inference. If CPU-only: still works but slower (~1-3 sec/memory vs ~0.1 sec/memory on GPU)
- Make enrichment OPTIONAL via config flag `INDEX_ENRICHMENT_ENABLED` (default True)
- Make COMET specifically toggleable: `COMET_ENRICHMENT_ENABLED` (default True)
- COMET model should be lazy-loaded (only load when first memory is inserted with enrichment enabled)

### Risk Mitigation
- PARA-COMET research shows 23% of COMET inferences are implausible. The FPA cosine filter (Technique 5) catches most of these.
- Cap at 9 inferences per memory (3 relations × 3 beams)
- Log rejected inferences at DEBUG level for tuning

## Technique 3: Doc2Query T5-small Expansion (60M params)

### What
Generate synthetic questions that each memory could answer. This bridges the query-document mismatch: stored documents are statements, queries are questions.

### Implementation
- Model: `doc2query/msmarco-t5-small-v1` from HuggingFace (~240MB)
- For each memory at insert time:
  1. Feed the memory content as input to T5
  2. Generate 5 synthetic queries using beam search (num_beams=5, max_length=64)
  3. Filter: remove duplicates and queries that are >80% token overlap with the original content (these add nothing)
- Store generated queries in a new column `enrichment_queries` (TEXT, JSON array) on memories table
- Include generated queries in FTS5 content
- Example: "Melanie went on camping trips last weekend" → generates: "Where did Melanie go last weekend?", "What outdoor activities does Melanie enjoy?", "Did Melanie go camping?", "What did Melanie do over the weekend?", "Does Melanie like camping?"

### GPU/CPU Handling
- Same pattern as COMET: check CUDA, use GPU if available
- Config flag: `DOC2QUERY_ENRICHMENT_ENABLED` (default True)
- Lazy-load the T5 model

## Technique 4: ProoFVer Natural Logic Expansion

### What
Apply simple natural logic entailment rules to generate safe, faithful expansions. These are strictly monotonic — zero risk of semantic drift.

### Implementation
- NO external model needed. Pure rule-based.
- Rules to implement:
  1. **Hypernym lifting**: If content mentions a specific named entity at a location, generate the category. "camping at Yellowstone" → "camping at a national park" → "outdoor activity"
  2. **Verb nominalization**: "went camping" → "camping trip", "enjoys reading" → "reading hobby"
  3. **Adjectival extraction**: "beautiful painting" → "painting, art, visual art"
- Use a small hardcoded hypernym map for common LoCoMo-relevant categories:
  ```
  Yellowstone/Yosemite/Grand Canyon → national park → outdoor destination
  Bach/Vivaldi/Beethoven → classical composer → classical music
  Dr. Seuss/Roald Dahl → children's author → children's literature
  camping/hiking/fishing → outdoor activity → nature
  painting/sculpture → visual art → creative hobby
  piano/violin/guitar → musical instrument → music
  ```
- Store in `enrichment_logic` column (TEXT)
- Include in FTS5 content

## Technique 5: FPA Noise Filter (Applied to ALL enrichments)

### What
Fact-Centric Preference Alignment: a cosine similarity threshold that prevents enrichment terms from drifting too far from the original memory's meaning.

### Implementation
- After generating ANY enrichment (ConceptNet, COMET, Doc2Query, ProoFVer), compute cosine similarity between:
  - The original memory's embedding (already computed)
  - An embedding of the enrichment text
- If similarity < 0.25, REJECT the enrichment term/inference
- This is cheap: one embedding call per enrichment batch (not per term)
- The threshold 0.25 is deliberately low — we WANT bridging terms that are somewhat distant, but not completely unrelated
- Log rejection rate at INFO level so we can tune the threshold

## Storage Schema Changes

### New columns on `memories` table:
```sql
ALTER TABLE memories ADD COLUMN enrichment_concepts TEXT DEFAULT NULL;  -- ConceptNet terms (JSON array)
ALTER TABLE memories ADD COLUMN enrichment_comet TEXT DEFAULT NULL;     -- COMET inferences (JSON array)
ALTER TABLE memories ADD COLUMN enrichment_queries TEXT DEFAULT NULL;   -- Doc2Query questions (JSON array)
ALTER TABLE memories ADD COLUMN enrichment_logic TEXT DEFAULT NULL;     -- ProoFVer expansions (JSON array)
ALTER TABLE memories ADD COLUMN enriched_content TEXT DEFAULT NULL;     -- Combined enriched text for FTS/embedding
ALTER TABLE memories ADD COLUMN enrichment_model_versions TEXT DEFAULT NULL; -- Track which models were used
```

### FTS5 update:
The `memories_fts` table currently indexes only `content`. We need it to also index the enriched content. Options:
1. **Preferred**: Change the FTS insert trigger to use `COALESCE(enriched_content, content)` instead of just `content`. This way enriched memories get the full enriched text in FTS, and non-enriched memories still work.
2. Update the FTS delete/insert triggers to include enriched_content.

### Embedding update:
When enrichment is enabled, the embedding should be computed on the enriched_content (original + enrichments combined) rather than just the original content. This pulls the vector representation toward the implied facts, improving vector search.

## Config Changes (zikkaron/config.py)

Add these settings to the Settings dataclass:
```python
# v17 Index-Time Enrichment Settings
INDEX_ENRICHMENT_ENABLED: bool = True
CONCEPTNET_ENRICHMENT_ENABLED: bool = True
CONCEPTNET_MIN_EDGE_WEIGHT: float = 1.0
CONCEPTNET_MAX_TERMS: int = 10
CONCEPTNET_RELATIONS: str = "IsA,UsedFor,HasProperty,AtLocation,MotivatedByGoal,CausesDesire,CapableOf"
COMET_ENRICHMENT_ENABLED: bool = True
COMET_MODEL: str = "mismayil/comet-bart-ai2"
COMET_NUM_BEAMS: int = 5
COMET_TOP_K_PER_RELATION: int = 3
COMET_MIN_CONFIDENCE: float = 0.3
COMET_RELATIONS: str = "xAttr,xIntent,xWant"
DOC2QUERY_ENRICHMENT_ENABLED: bool = True
DOC2QUERY_MODEL: str = "doc2query/msmarco-t5-small-v1"
DOC2QUERY_NUM_QUERIES: int = 5
LOGIC_ENRICHMENT_ENABLED: bool = True
FPA_SIMILARITY_THRESHOLD: float = 0.25
ENRICHMENT_MIN_CONTENT_LENGTH: int = 20
```

## New Module: zikkaron/enrichment.py

Create a new module `zikkaron/enrichment.py` containing:
1. `EnrichmentPipeline` class that orchestrates all enrichment techniques
2. `ConceptNetExpander` class for ConceptNet expansion
3. `CometInferencer` class for COMET-BART inference
4. `Doc2QueryExpander` class for synthetic query generation
5. `LogicExpander` class for natural logic rules
6. `FPAFilter` class for noise filtering
7. Main method: `enrich(content: str, embedding: bytes, settings: Settings) -> EnrichmentResult`
8. `EnrichmentResult` dataclass with fields for each technique's output plus combined `enriched_content` string

## Integration with storage.py

Modify `StorageEngine.insert_memory()`:
1. After computing the embedding but before inserting into SQLite
2. If `INDEX_ENRICHMENT_ENABLED` is True and content length >= `ENRICHMENT_MIN_CONTENT_LENGTH`:
   a. Call `EnrichmentPipeline.enrich(content, embedding, settings)`
   b. Store individual enrichment results in their respective columns
   c. Build `enriched_content` = original content + "\n" + all enrichment terms/queries joined
   d. Use `enriched_content` for the FTS5 insert (via trigger update)
   e. Re-compute embedding on `enriched_content` for better vector representation
3. The enrichment pipeline should be lazy-initialized (models loaded on first use)

## Integration with retrieval.py

Minimal changes needed in retrieval.py:
- The enrichment is transparent — FTS5 and vector search automatically benefit because the indexed content is richer
- Remove or reduce the hardcoded `_OPEN_DOMAIN_TOPIC_EXPANSIONS` dict since ConceptNet handles this generically
- The `_derive_implied_fact_passages()` function in CE reranking can be simplified or removed since implied facts are now in the stored content

## Validation Criteria

1. `enrichment.py` module exists and all 4 enrichment classes are implemented
2. `config.py` has all new settings with correct defaults
3. `storage.py` calls enrichment pipeline during insert_memory()
4. Schema migration adds new columns without breaking existing DBs
5. Unit test: insert a memory "Melanie went camping", verify enrichment_concepts contains "national_park" or "outdoor"
6. Unit test: insert a memory, verify enrichment_comet contains personality traits
7. Unit test: insert a memory, verify enrichment_queries contains synthetic questions
8. Unit test: FPA filter rejects unrelated enrichment (cosine < 0.25)
9. Integration test: store "went camping", search for "national park" — should find the memory via FTS
10. All existing tests pass (no regressions)
11. Enrichment can be disabled via config flags without errors

## Files to Create/Modify
- CREATE: `zikkaron/enrichment.py` (new module, ~400-600 lines)
- MODIFY: `zikkaron/storage.py` (schema migration + insert_memory integration)
- MODIFY: `zikkaron/config.py` (new settings)
- MODIFY: `zikkaron/retrieval.py` (remove hardcoded expansions, simplify CE implied facts)
- CREATE: `zikkaron/tests/test_enrichment.py` (unit tests)

## Dependencies to Install
- `conceptnet-lite` or fallback to API
- `transformers` (already installed for sentence-transformers)
- `torch` (already installed)
- Models will auto-download from HuggingFace on first use

## Performance Constraints
- Enrichment runs at STORAGE TIME only — zero cost at query time
- With GPU: ~0.5 sec/memory total (all techniques)
- With CPU: ~5-10 sec/memory total (acceptable for offline processing)
- Memory footprint: COMET-BART ~1.6GB, Doc2Query ~240MB, ConceptNet ~2GB on disk
- All models lazy-loaded — if user never inserts memories, nothing loads
