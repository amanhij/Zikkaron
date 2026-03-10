# Lockstep Prompt: Structured Profiles + Query Routing Improvements

## Goal
Implement Memobase-style structured user profiles, Hindsight-style epistemically distinct memory layers, proper comparison query routing, and replace RRF with Convex Combination scoring. These techniques address the 26% multi-memory aggregation failure and the comparison question failure mode.

## Current Architecture Context
- **Storage**: `zikkaron/storage.py` — SQLite with `memories`, `memories_fts`, `memory_vectors` tables
- **Retrieval**: `zikkaron/retrieval.py` — `HippoRetriever` with WRRF fusion + FlashRank CE
- **Config**: `zikkaron/config.py` — Settings dataclass
- **Benchmark**: LoCoMo-mc10, 1986 QA pairs, current open_domain MRR=0.381, overall MRR=0.708
- **Problem**: 26% of failures need multi-memory aggregation. "What are Melanie's personality traits?" requires synthesizing dozens of scattered memories. No single memory scores high enough.

## Technique 1: Memobase Structured User Profiles (Expected: +0.08 MRR)

### What
Maintain a structured profile table that aggregates scattered behavioral memories into deterministic attribute-value pairs. When "Melanie went camping", "Melanie hiked a trail", "Melanie visited Yellowstone" are stored across different sessions, the profile should contain `{interest: "outdoor activities", trait: "adventurous", preference: "nature"}`.

### Implementation

#### New SQLite table: `user_profiles`
```sql
CREATE TABLE IF NOT EXISTS user_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_name TEXT NOT NULL,           -- "Melanie", "John", etc.
    attribute_type TEXT NOT NULL,         -- "interest", "trait", "preference", "value", "habit", "goal", "opinion"
    attribute_key TEXT NOT NULL,          -- "outdoor_activities", "adventurous"
    attribute_value TEXT NOT NULL,        -- Full natural language description
    evidence_memory_ids TEXT DEFAULT '[]', -- JSON array of memory IDs that support this attribute
    confidence REAL DEFAULT 0.5,         -- How confident we are (0-1)
    created_at TEXT,
    updated_at TEXT,
    directory_context TEXT,
    UNIQUE(entity_name, attribute_type, attribute_key, directory_context)
);
```

Also create an FTS5 index on profiles:
```sql
CREATE VIRTUAL TABLE IF NOT EXISTS profiles_fts USING fts5(
    entity_name, attribute_type, attribute_key, attribute_value,
    content=user_profiles, content_rowid=id
);
```

And a vector index for profile embeddings:
```sql
-- Each profile attribute gets its own embedding for vector search
-- Store in memory_vectors alongside memory embeddings, distinguished by a type column
-- OR create a separate profiles_vectors vec0 table
```

#### Profile Extraction Logic
Create `zikkaron/profiles.py` with a `ProfileExtractor` class:

1. **Rule-based extraction** (no LLM needed, runs at storage time):
   - Pattern: `{Name} {likes/loves/enjoys/prefers} {X}` → `{interest: X}`
   - Pattern: `{Name} is {adjective}` → `{trait: adjective}`
   - Pattern: `{Name} {went to/visited/traveled to} {place}` → `{interest: travel, preference: place_type}`
   - Pattern: `{Name} works as/is a {job}` → `{career: job}`
   - Pattern: `{Name} believes/thinks {X}` → `{opinion: X}`
   - Pattern: `{Name} wants to/hopes to {X}` → `{goal: X}`
   - Pattern: `{Name} always/usually/often {X}` → `{habit: X}`

2. **COMET-assisted extraction** (if COMET enrichment is enabled from Spec 01):
   - Use xAttr inferences as trait candidates
   - Use xWant inferences as goal/preference candidates
   - Higher confidence (0.7) for direct pattern matches, lower (0.4) for COMET-derived

3. **Profile update logic**:
   - On each memory insert, extract profile attributes
   - If attribute already exists for entity: update confidence (increment by 0.1, cap at 1.0), append memory_id to evidence list
   - If new attribute: insert with base confidence
   - Conflicting attributes: keep both, let confidence scores differentiate

4. **Profile-as-document generation**:
   - After updating profiles, generate a natural language summary per entity:
     ```
     "Melanie is adventurous and enjoys outdoor activities including camping, hiking, and visiting national parks.
      She values nature and has visited Yellowstone. She works as a teacher and enjoys reading children's books."
     ```
   - Embed this summary and store in a dedicated `profile_summaries` table or as a special memory with tag "profile_summary"
   - This summary becomes searchable via both FTS and vector search

### Integration with Retrieval
In `HippoRetriever.recall()`:
1. After standard FTS + vector search, also search `profiles_fts` and profile vector embeddings
2. If profile results are found, include them in the candidate pool before CE reranking
3. Profile results should be weighted at 0.8x normal results (they're synthesized, not verbatim memories)

## Technique 2: Hindsight Epistemically Distinct Memory (Expected: +0.06 MRR)

### What
Separate raw episodic memories from derived beliefs/traits. Create a `derived_beliefs` table that stores synthesized knowledge distinct from raw observations.

### Implementation

#### New SQLite table: `derived_beliefs`
```sql
CREATE TABLE IF NOT EXISTS derived_beliefs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    belief_type TEXT NOT NULL,           -- "trait", "preference", "relationship", "pattern", "summary"
    subject TEXT NOT NULL,               -- Entity the belief is about
    content TEXT NOT NULL,               -- Natural language belief statement
    evidence_memory_ids TEXT DEFAULT '[]', -- Supporting memory IDs
    confidence REAL DEFAULT 0.5,
    embedding BLOB,
    embedding_model TEXT,
    created_at TEXT,
    updated_at TEXT,
    directory_context TEXT
);
```

With FTS5 and vec0 indexes, same pattern as memories.

#### Belief Derivation
- Runs as a background "reflect" operation after memory insertion
- For each new memory, check if it adds evidence to existing beliefs or creates new ones
- Belief types:
  - **trait**: "Melanie is adventurous" (from multiple outdoor activity memories)
  - **preference**: "Melanie prefers nature over urban settings" (from camping, hiking, park visits)
  - **relationship**: "Melanie and Caroline are close friends" (from frequent interactions)
  - **pattern**: "Melanie typically exercises on weekends" (from temporal patterns)
  - **summary**: "Melanie's hobbies include camping, reading, and painting" (aggregation)

#### Integration with Retrieval
- Search derived_beliefs alongside memories
- Beliefs with high confidence (>0.7) get a 1.2x score boost (they're pre-synthesized answers)
- For open_domain queries specifically, search beliefs FIRST (they directly answer inference questions)

## Technique 3: CompUGE Comparison Query Routing (Expected: +0.03 MRR)

### What
Detect "A or B?" comparison questions and route them through a dual-search pipeline instead of a single search.

### Implementation
In `zikkaron/retrieval.py`:

1. **Detection** (enhance existing `analyze_query()`):
   - Already has `_extract_comparison_options()` — this works
   - Add detection for implicit comparisons: "prefer X over Y", "rather X than Y", "X instead of Y"

2. **Dual-search execution**:
   When comparison detected with options [A, B]:
   ```python
   # Search for evidence of each option separately
   results_a = self._search_for_option(query, option_a, subject)  # "Melanie national park"
   results_b = self._search_for_option(query, option_b, subject)  # "Melanie theme park"

   # Score each option by summing top-3 CE scores
   score_a = sum(r["_cross_encoder_score"] for r in results_a[:3])
   score_b = sum(r["_cross_encoder_score"] for r in results_b[:3])

   # Merge results, prioritizing the winning option
   merged = interleave(results_a, results_b, bias_toward=max(score_a, score_b))
   ```

3. **Subject extraction**: Parse the query to find the subject entity ("Would Melanie prefer...") and include it in both sub-searches

## Technique 4: Convex Combination Replacing RRF (Expected: +0.01-0.02 MRR)

### What
Replace Weighted Reciprocal Rank Fusion with Convex Combination scoring (proven superior by Bruch et al., ACM TOIS 2023).

### Implementation
In `zikkaron/retrieval.py`, add a new fusion method alongside existing `_wrrf_fuse`:

```python
def _convex_fuse(self, signal_scores: dict[str, dict[int, float]],
                  weights: dict[str, float]) -> list[tuple[int, float]]:
    """Convex combination of normalized signal scores.

    Unlike RRF which uses ranks, this uses actual scores with min-max normalization.
    Proven to outperform RRF (Bruch et al., ACM TOIS 2023).

    signal_scores: {signal_name: {memory_id: raw_score}}
    weights: {signal_name: weight} (must sum to 1.0, will be normalized)
    """
    # Normalize weights to sum to 1.0
    total_w = sum(weights.values())
    norm_weights = {k: v/total_w for k, v in weights.items()}

    # Min-max normalize each signal independently
    normalized: dict[str, dict[int, float]] = {}
    for signal, scores in signal_scores.items():
        if not scores:
            continue
        vals = list(scores.values())
        min_s, max_s = min(vals), max(vals)
        range_s = max_s - min_s
        if range_s > 0:
            normalized[signal] = {mid: (s - min_s) / range_s for mid, s in scores.items()}
        else:
            normalized[signal] = {mid: 0.5 for mid in scores}

    # Convex combination
    all_mids = set()
    for scores in normalized.values():
        all_mids.update(scores.keys())

    combined = {}
    for mid in all_mids:
        combined[mid] = sum(
            norm_weights.get(sig, 0) * normalized.get(sig, {}).get(mid, 0)
            for sig in normalized
        )

    return sorted(combined.items(), key=lambda x: x[1], reverse=True)
```

### Config
```python
FUSION_METHOD: str = "convex"  # "wrrf" or "convex"
```

### Integration
In `HippoRetriever.recall()`, check `self._settings.FUSION_METHOD`:
- If "convex": collect raw scores from each signal, call `_convex_fuse()`
- If "wrrf": use existing `_wrrf_fuse()` (backward compatible)
- Default to "convex" for new installations

## Config Changes (zikkaron/config.py)

```python
# v18 Structured Profiles
PROFILE_EXTRACTION_ENABLED: bool = True
PROFILE_CONFIDENCE_DIRECT: float = 0.7      # Confidence for direct pattern matches
PROFILE_CONFIDENCE_INFERRED: float = 0.4    # Confidence for COMET-derived attributes
PROFILE_SEARCH_WEIGHT: float = 0.8          # Weight of profile results vs memory results
PROFILE_SUMMARY_ENABLED: bool = True        # Generate NL profile summaries

# v19 Derived Beliefs (Hindsight)
DERIVED_BELIEFS_ENABLED: bool = True
BELIEF_MIN_CONFIDENCE: float = 0.3
BELIEF_HIGH_CONFIDENCE_BOOST: float = 1.2   # Score multiplier for high-confidence beliefs
BELIEF_SEARCH_PRIORITY_FOR_OPEN_DOMAIN: bool = True  # Search beliefs first for inference queries

# v20 Comparison Query Routing
COMPARISON_DUAL_SEARCH_ENABLED: bool = True
COMPARISON_TOP_K_PER_OPTION: int = 10

# v21 Fusion Method
FUSION_METHOD: str = "convex"  # "wrrf" or "convex"
```

## Validation Criteria

1. `user_profiles` table created with correct schema
2. `derived_beliefs` table created with correct schema
3. Profile extraction works: insert "Melanie loves camping", verify profile entry exists with interest="camping"
4. Profile accumulation: insert 3 camping-related memories, verify confidence increases
5. Profile summary generation: entity with 3+ attributes gets NL summary
6. Profile search integration: query "outdoor activities" finds Melanie's profile
7. Derived beliefs: insert multiple outdoor memories, verify a belief like "enjoys outdoor activities" is derived
8. Comparison routing: query "national park or theme park" triggers dual search (verify via log or return metadata)
9. Convex combination: verify it produces different rankings than WRRF on the same input
10. All existing tests pass
11. Each feature can be independently disabled via config

## Files to Create/Modify
- CREATE: `zikkaron/profiles.py` (~300 lines)
- MODIFY: `zikkaron/storage.py` (new tables + schema migration)
- MODIFY: `zikkaron/retrieval.py` (profile search integration, comparison routing, convex fusion)
- MODIFY: `zikkaron/config.py` (new settings)
- CREATE: `zikkaron/tests/test_profiles.py`
