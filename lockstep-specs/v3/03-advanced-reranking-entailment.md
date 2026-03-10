# Lockstep Prompt: Advanced Reranking + Entailment Scoring

## Goal
Replace the pointwise FlashRank CE with a stronger reranking model, add NLI-based entailment scoring as a complementary signal, and prepare the embedding architecture for dual-vector (explicit/implicit) search. These changes target the remaining inference gap after index-time enrichment.

## Current Architecture Context
- **Reranking**: `zikkaron/retrieval.py` — `_cross_encoder_rerank()` uses FlashRank `ms-marco-MiniLM-L-12-v2` (33M params, ONNX)
- **Embeddings**: `zikkaron/embeddings.py` — `EmbeddingEngine` with nomic-embed-text-v1.5 (768d)
- **Storage**: sqlite-vec `memory_vectors` table with single embedding column
- **Benchmark**: open_domain MRR=0.381 with current CE. CE is proven to be "semantic BM25" — cannot do inference.
- **FlashRank CE is our biggest bottleneck**: It scores "camping trips" at 0.013 relevance to "national park" because it can't bridge behavioral→preference inference. Even with index-time enrichment adding "nature lover" to the stored text, a better CE model will further improve scoring.

## Technique 1: GTE-Reranker-ModernBERT-base (Expected: +0.03-0.05 MRR)

### What
Replace ms-marco-MiniLM-L-12-v2 (33M) with `Alibaba-NLP/gte-reranker-modernbert-base` (149M). This model uses ModernBERT architecture with better zero-shot OOD generalization, 83% Hit@1 on standard benchmarks, and is CPU-viable via ONNX.

### Implementation
1. Install model: it's available on HuggingFace as a cross-encoder compatible model
2. In `_cross_encoder_rerank()`, add support for loading this model via `sentence-transformers` CrossEncoder class (not FlashRank):
   ```python
   from sentence_transformers import CrossEncoder
   ce_model = CrossEncoder("Alibaba-NLP/gte-reranker-modernbert-base", max_length=512)
   scores = ce_model.predict([(query, doc["content"]) for doc in memories])
   ```
3. Make the model configurable via `CROSS_ENCODER_MODEL` setting (already exists, currently defaults to ms-marco-MiniLM-L-6-v2)
4. Keep FlashRank as fallback if sentence-transformers CrossEncoder fails
5. ONNX export for faster inference: attempt to use `optimum` for ONNX conversion if available

### Config
```python
CROSS_ENCODER_MODEL: str = "Alibaba-NLP/gte-reranker-modernbert-base"  # Updated default
CROSS_ENCODER_FALLBACK: str = "ms-marco-MiniLM-L-12-v2"  # FlashRank fallback
CROSS_ENCODER_MAX_LENGTH: int = 512
```

### GPU/CPU
- On GPU: use sentence-transformers directly (fast, ~10ms per pair)
- On CPU: still viable at 149M params, ~50-100ms per pair
- For 75 candidates: ~4-8 sec on CPU vs ~0.75 sec on GPU

## Technique 2: NLI Entailment Score as Reranking Signal (Expected: +0.03-0.05 MRR)

### What
Add Natural Language Inference scoring as a COMPLEMENTARY signal to the CE. Standard NLI models can detect if a document ENTAILS the answer to a query, which is exactly what we need for inference-style questions.

### Implementation
1. Model: `cross-encoder/nli-deberta-v3-base` (86M params) from HuggingFace
   - Outputs 3 logits: [contradiction, neutral, entailment]
   - We use the entailment probability as a relevance signal
2. Create `_nli_rerank()` method in HippoRetriever:
   ```python
   def _nli_rerank(self, query: str, memories: list[dict], top_k: int) -> list[dict]:
       """Score memories by NLI entailment probability with the query."""
       from sentence_transformers import CrossEncoder
       nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

       # Frame as: premise=document, hypothesis=query-as-statement
       # Convert question to statement for NLI: "Would Melanie prefer national parks?" → "Melanie prefers national parks"
       hypothesis = _question_to_statement(query)

       pairs = [(mem["content"], hypothesis) for mem in memories]
       scores = nli_model.predict(pairs)  # Returns [contradiction, neutral, entailment] logits

       for i, mem in enumerate(memories):
           entailment_score = softmax(scores[i])[2]  # Index 2 = entailment
           mem["_nli_entailment_score"] = float(entailment_score)

       return memories
   ```

3. **Question-to-statement conversion** (`_question_to_statement()`):
   - "Would Melanie prefer a national park?" → "Melanie prefers a national park"
   - "What are Melanie's personality traits?" → "Melanie has personality traits"
   - "Is Melanie a member of the LGBTQ community?" → "Melanie is a member of the LGBTQ community"
   - Use the existing `_pseudo_hyde_expand()` function or similar regex patterns

4. **Integration with scoring**:
   - Run NLI scoring ONLY on open_domain queries (detected via `analyze_query()`)
   - Combine NLI entailment score with CE score: `final_score = 0.7 * ce_score + 0.3 * nli_score`
   - For non-open_domain queries, use CE only (NLI adds nothing for factual retrieval)

### Config
```python
NLI_RERANKING_ENABLED: bool = True
NLI_MODEL: str = "cross-encoder/nli-deberta-v3-base"
NLI_WEIGHT: float = 0.3  # Blend weight with CE score
NLI_ONLY_FOR_OPEN_DOMAIN: bool = True  # Only activate for inference queries
```

## Technique 3: Set-Encoder Listwise Reranking (Expected: +0.08-0.12 MRR)

### What
The Set-Encoder (Schlatt et al., ECIR 2025) processes multiple passages simultaneously with inter-passage attention via `<INT>` tokens. This allows it to aggregate evidence from scattered memories — exactly what fails for "What are personality traits?" questions.

### Implementation Reality Check
The Set-Encoder is a research model. A pre-trained version may or may not be available on HuggingFace. Implementation strategy:

**Option A (if model available):**
- Check HuggingFace for `irs-lab/set-encoder` or similar
- Load via sentence-transformers or custom code
- Feed top-50 candidates as a batch with inter-passage attention

**Option B (practical alternative — Multi-passage CE):**
If Set-Encoder is not available as a downloadable model, implement a simplified version:
1. After standard CE scoring, take the top-10 candidates
2. Concatenate top-3 related passages and re-score as a single document
3. This simulates multi-passage evidence aggregation without a custom model:
   ```python
   # Group memories by entity/topic
   clusters = cluster_by_overlap(top_candidates)
   for cluster in clusters:
       combined_text = " ".join(m["content"] for m in cluster[:3])
       combined_score = ce_model.predict([(query, combined_text)])
       # If combined score > max individual score, boost all cluster members
   ```

**Option C (answerai-colbert-small-v1 — 33M):**
- ColBERTv2 with token-level MaxSim scoring
- Naturally handles multi-evidence via token-level matching
- Available on HuggingFace: `answerdotai/answerai-colbert-small-v1`
- Can score passages that share partial evidence better than pointwise CE

### Implementation
Try Option A first. If unavailable, implement Option B as the practical solution. Option C as backup.

```python
def _multi_passage_rerank(self, query: str, memories: list[dict], top_k: int) -> list[dict]:
    """Multi-passage evidence aggregation reranking.

    Groups related memories and re-scores clusters to detect
    when multiple weak pieces of evidence combine into strong evidence.
    """
    # 1. Standard CE scoring first
    memories = self._cross_encoder_rerank(query, memories, top_k=50)

    # 2. Cluster by entity/topic overlap
    clusters = self._cluster_memories(memories[:20])

    # 3. For each cluster with 2+ members, score concatenated text
    for cluster_mems in clusters:
        if len(cluster_mems) < 2:
            continue
        combined = " | ".join(m["content"][:200] for m in cluster_mems[:3])
        combined_score = self._score_single_pair(query, combined)
        # If combined evidence is stronger, boost individual members
        max_individual = max(m["_cross_encoder_score"] for m in cluster_mems)
        if combined_score > max_individual:
            boost = (combined_score - max_individual) * 0.5
            for m in cluster_mems:
                m["_retrieval_score"] += boost

    memories.sort(key=lambda m: m["_retrieval_score"], reverse=True)
    return memories[:top_k]
```

### Config
```python
MULTI_PASSAGE_RERANKING_ENABLED: bool = True
MULTI_PASSAGE_CLUSTER_OVERLAP_THRESHOLD: float = 0.3  # Min Jaccard overlap to cluster
MULTI_PASSAGE_MAX_CLUSTER_SIZE: int = 3
```

## Technique 4: DualCSE Preparation — Dual Vector Architecture (Expected: +0.10-0.15 MRR, FUTURE)

### What
DualCSE (Wang et al., 2025) projects two co-existing embeddings per sentence: explicit (literal) and implicit (latent meaning). "went camping" has explicit vector near "camping, tent" and implicit vector near "nature, national parks". This is the highest-impact single technique but requires model fine-tuning.

### Implementation (Architecture Prep Only)
Since fine-tuning DualCSE requires training data (INLI dataset) and GPU hours, this step only prepares the architecture:

1. **Dual-vector storage schema**:
   ```sql
   -- Add implicit embedding column to memories
   ALTER TABLE memories ADD COLUMN implicit_embedding BLOB DEFAULT NULL;
   ALTER TABLE memories ADD COLUMN implicit_embedding_model TEXT DEFAULT NULL;

   -- Create a second vec0 table for implicit embeddings
   CREATE VIRTUAL TABLE IF NOT EXISTS memory_implicit_vectors USING vec0(
       embedding float[768]
   );
   ```

2. **Dual-vector search in retrieval.py**:
   ```python
   def _dual_vector_search(self, query_embedding, top_k):
       """Search both explicit and implicit vector spaces."""
       explicit_results = self._storage.search_vectors(query_embedding, top_k)
       implicit_results = self._storage.search_implicit_vectors(query_embedding, top_k)
       # Merge with weighted combination
       return self._merge_vector_results(explicit_results, implicit_results,
                                          explicit_weight=0.5, implicit_weight=0.5)
   ```

3. **Config**:
   ```python
   DUAL_VECTORS_ENABLED: bool = False  # Off by default until DualCSE model is trained
   IMPLICIT_EMBEDDING_MODEL: str = ""  # Path to fine-tuned DualCSE model
   IMPLICIT_VECTOR_WEIGHT: float = 0.5
   ```

4. **Fallback**: Until the DualCSE model is fine-tuned, use the enriched_content embedding as a proxy implicit embedding. The enriched content (with COMET inferences + ConceptNet terms) naturally captures implied meaning:
   ```python
   if settings.DUAL_VECTORS_ENABLED and settings.IMPLICIT_EMBEDDING_MODEL:
       # Use DualCSE model for implicit embedding
       implicit_emb = dualcse_model.encode_implicit(content)
   elif enriched_content:
       # Fallback: use enriched content embedding as proxy implicit vector
       implicit_emb = embeddings.encode_document(enriched_content)
   ```

## Validation Criteria

1. GTE-Reranker loads and scores correctly (test with 5 query-document pairs)
2. NLI model loads and returns entailment probabilities
3. NLI scoring only activates for open_domain queries
4. Question-to-statement conversion works for: "Would X prefer Y?", "What are X's traits?", "Is X a member of Y?"
5. Multi-passage reranking: when 3 outdoor memories are in top-20, their combined score is higher than individual scores
6. Memory clustering groups related memories correctly (Jaccard overlap test)
7. Dual-vector schema migration adds columns without breaking existing DB
8. Dual-vector search works with proxy implicit embeddings
9. All features independently toggleable via config
10. Existing tests pass — no regressions
11. Benchmark improvement: run on LoCoMo conv 1 and verify open_domain MRR > 0.50

## Files to Modify
- MODIFY: `zikkaron/retrieval.py` (GTE-Reranker, NLI scoring, multi-passage reranking, dual-vector search)
- MODIFY: `zikkaron/storage.py` (dual-vector schema, implicit vector search)
- MODIFY: `zikkaron/config.py` (new settings)
- MODIFY: `zikkaron/embeddings.py` (dual encoding support)
- CREATE: `zikkaron/tests/test_reranking.py` (unit tests)

## Dependencies
- `sentence-transformers` (already installed, needed for CrossEncoder)
- GTE-Reranker model: auto-downloads from HuggingFace (~600MB)
- NLI model: auto-downloads from HuggingFace (~350MB)
- All models lazy-loaded
