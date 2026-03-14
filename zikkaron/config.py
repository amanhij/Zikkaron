from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PORT: int = 8742
    IDLE_THRESHOLD_SECONDS: int = 300
    DECAY_FACTOR: float = 0.95
    COLD_THRESHOLD: float = 0.05
    HOT_THRESHOLD: float = 0.7
    MAX_EPISODE_TOKENS: int = 50000
    OVERLAP_TOKENS: int = 2000
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    DAEMON_CHECK_INTERVAL: int = 30
    DB_PATH: str = "~/.zikkaron/memory.db"

    # v2 settings
    IMPORTANCE_DECAY_FACTOR: float = 0.998
    SURPRISE_BOOST: float = 0.3
    EMOTIONAL_DECAY_RESISTANCE: float = 0.5
    DREAM_REPLAY_PAIRS: int = 20
    FRACTAL_LEVELS: int = 3
    CLUSTER_SIMILARITY_THRESHOLD: float = 0.7
    PPR_DAMPING: float = 0.85
    PPR_ITERATIONS: int = 50
    CAUSAL_THRESHOLD: int = 3
    SYNAPTIC_WINDOW_MINUTES: int = 30
    SYNAPTIC_BOOST: float = 0.2
    NUM_ASTROCYTE_PROCESSES: int = 4
    NARRATIVE_INTERVAL_HOURS: int = 24
    CONTEXTUAL_PREFIX_ENABLED: bool = True
    CURATION_SIMILARITY_THRESHOLD: float = 0.85

    # v3 frontier settings
    HOPFIELD_BETA: float = 8.0  # Hopfield sharpness (low=blended, high=precise)
    HOPFIELD_MAX_PATTERNS: int = 5000  # Max patterns in Hopfield energy store
    RECONSOLIDATION_LOW_THRESHOLD: float = 0.3  # Below this: no modification on recall
    RECONSOLIDATION_HIGH_THRESHOLD: float = 0.7  # Above this: archive old + create new
    PLASTICITY_SPIKE: float = 0.3  # How much plasticity increases on access
    PLASTICITY_HALF_LIFE_HOURS: float = 6.0  # Plasticity decay half-life
    STABILITY_INCREMENT: float = 0.1  # Stability increase per successful retrieval
    EXCITABILITY_HALF_LIFE_HOURS: float = 6.0  # Engram excitability decay half-life
    EXCITABILITY_BOOST: float = 0.5  # Excitability increase on slot activation
    WRITE_GATE_THRESHOLD: float = 0.4  # Min surprisal to pass write gate
    COMPRESSION_GIST_AGE_HOURS: float = 168.0  # 7 days before gist compression
    COMPRESSION_TAG_AGE_HOURS: float = 720.0  # 30 days before tag compression
    HDC_DIMENSIONS: int = 10000  # Hyperdimensional vector size
    SR_DISCOUNT: float = 0.9  # Successor representation discount factor γ
    SR_UPDATE_RATE: float = 0.1  # Incremental SR update learning rate
    COGNITIVE_LOAD_LIMIT: int = 4  # Max chunks in active context (Cowan's 4±1)
    CRDT_AGENT_ID: str = "default"  # Agent identifier for multi-agent CRDT

    # v4: Hippocampal Replay settings
    REPLAY_MAX_RESTORE_MEMORIES: int = 8  # Max memories to include in restoration
    REPLAY_ANCHOR_HEAT: float = 1.0  # Heat assigned to anchored memories
    REPLAY_CHECKPOINT_AUTO_INTERVAL: int = 50  # Auto-checkpoint every N tool calls

    # v5: Zero-gap memory persistence settings
    WRITE_GATE_CONTINUITY_DISCOUNT: float = 0.15  # Threshold reduction for task-continuous content
    WRITE_GATE_CONTINUITY_WINDOW: int = 10  # Number of recent stores to track for continuity
    MICRO_CHECKPOINT_ENABLED: bool = True  # Auto-checkpoint on significant events
    MICRO_CHECKPOINT_COOLDOWN: int = 5  # Min tool calls between micro-checkpoints
    SESSION_COHERENCE_BONUS: float = 0.2  # Heat bonus for current-session memories
    SESSION_COHERENCE_WINDOW_HOURS: float = 4.0  # How long the session coherence lasts
    REINJECTION_ENABLED: bool = True  # Auto-surface related context on remember
    REINJECTION_MAX_RESULTS: int = 3  # Max related memories to reinject
    DECISION_AUTO_PROTECT: bool = True  # Auto-protect detected decisions from decay
    ACTION_STREAM_ENABLED: bool = True  # Capture tool actions in sensory buffer

    # v6: WRRF (Weighted Reciprocal Rank Fusion) settings
    WRRF_K: int = 60  # RRF constant k
    WRRF_CANDIDATE_MULTIPLIER: int = 10  # Candidate pool = max_results * this
    WRRF_VECTOR_WEIGHT: float = 1.0
    WRRF_FTS_WEIGHT: float = 0.0
    WRRF_PPR_WEIGHT: float = 0.5
    WRRF_SPREADING_WEIGHT: float = 0.3
    WRRF_HOPFIELD_WEIGHT: float = 0.2
    WRRF_HDC_WEIGHT: float = 0.3
    WRRF_FRACTAL_WEIGHT: float = 0.2
    WRRF_SR_WEIGHT: float = 0.3
    RERANKER_ENABLED: bool = True
    RERANKER_TOP_K: int = 50

    # v7: Query routing settings
    QUERY_ROUTING_ENABLED: bool = True
    TEMPORAL_KEYWORDS: str = "yesterday,today,last week,last month,last session,recently,before,after,when,during,while,since,until,earlier,later,previous,next,morning,evening,night,ago,back then"
    CODE_KEYWORDS: str = "function,class,method,variable,import,error,bug,fix,refactor,implement,API,endpoint,database,schema,test,deploy"
    RELATIONAL_KEYWORDS: str = "relationship,connection,related,between,link,cause,effect,impact,influence,depend,lead to,result in"

    # v8: Confidence gating settings
    CONFIDENCE_GATING_ENABLED: bool = True
    CONFIDENCE_MIN_RESULTS: int = 3
    CONFIDENCE_SCORE_SPREAD_THRESHOLD: float = 0.15
    CONFIDENCE_TOP_SCORE_THRESHOLD: float = 0.5
    CONFIDENCE_FALLBACK_STRATEGY: str = "expand"

    # v9: Temporal retrieval settings
    TEMPORAL_RETRIEVAL_ENABLED: bool = True
    TEMPORAL_BOOST_WEIGHT: float = 0.4
    TEMPORAL_DECAY_DAYS: int = 30
    TEMPORAL_EXACT_MATCH_BOOST: float = 2.0

    # v10: Cross-encoder reranking settings
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CROSS_ENCODER_ENABLED: bool = True  # FlashRank ONNX is fast enough for CPU
    CROSS_ENCODER_TOP_K: int = 20
    CROSS_ENCODER_WEIGHT: float = 0.6  # CE weight in blend (retrieval gets 1-this)

    # v11: FTS5 enhancement settings
    FTS5_PORTER_STEMMER: bool = False
    FTS5_IDENTIFIER_SPLITTING: bool = True
    FTS5_ENTITY_BOOST: bool = True
    FTS5_MIN_TERM_LENGTH: int = 2

    # v12: Graph signal optimization settings
    GRAPH_MAX_HOPS: int = 2
    GRAPH_MIN_EDGE_WEIGHT: float = 0.1
    GRAPH_SPREADING_DECAY: float = 0.5
    GRAPH_SPREADING_MAX_DEPTH: int = 2
    GRAPH_ENTITY_MIN_LENGTH: int = 3

    # v13: Adversarial protection settings
    ADVERSARIAL_DETECTION_ENABLED: bool = True
    ADVERSARIAL_SCORE_GAP_THRESHOLD: float = 0.05
    ADVERSARIAL_DIVERSITY_ENFORCEMENT: bool = True
    ADVERSARIAL_MIN_CONFIDENCE: float = 0.3

    # v14: Embedding enhancement settings
    CANDIDATE_POOL_MULTIPLIER: int = 20
    EMBEDDING_CACHE_SIZE: int = 128
    QUERY_PREFIX: str = ""

    # v16: Query expansion (pseudo-HyDE) settings
    QUERY_EXPANSION_ENABLED: bool = True

    # v15: Fusion optimization settings
    COMBMNZ_ENABLED: bool = False  # CombMNZ: multiply fused score by signal count
    FUSION_NORM: str = "zscore"  # "zscore", "minmax", or "raw"

    # v17 Index-Time Enrichment Settings
    INDEX_ENRICHMENT_ENABLED: bool = True
    CONCEPTNET_ENRICHMENT_ENABLED: bool = True
    CONCEPTNET_MIN_EDGE_WEIGHT: float = 1.0
    CONCEPTNET_MAX_TERMS: int = 10
    CONCEPTNET_RELATIONS: str = "IsA,UsedFor,HasProperty,AtLocation,MotivatedByGoal,CausesDesire,CapableOf"
    COMET_ENRICHMENT_ENABLED: bool = True
    COMET_QUERY_EXPANSION_ENABLED: bool = False  # COMET at query time for open_domain
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

    # v18 Structured Profiles (Memobase)
    PROFILE_EXTRACTION_ENABLED: bool = True
    PROFILE_CONFIDENCE_DIRECT: float = 0.7
    PROFILE_CONFIDENCE_INFERRED: float = 0.4
    PROFILE_SEARCH_WEIGHT: float = 0.8
    PROFILE_SUMMARY_ENABLED: bool = True

    # v19 Derived Beliefs (Hindsight)
    DERIVED_BELIEFS_ENABLED: bool = True
    BELIEF_MIN_CONFIDENCE: float = 0.3
    BELIEF_HIGH_CONFIDENCE_BOOST: float = 1.2
    BELIEF_SEARCH_PRIORITY_FOR_OPEN_DOMAIN: bool = True

    # v20 Comparison Query Routing
    COMPARISON_DUAL_SEARCH_ENABLED: bool = True
    COMPARISON_TOP_K_PER_OPTION: int = 10

    # v21 Fusion Method
    FUSION_METHOD: str = "convex"

    # v22 Advanced Reranking — GTE-Reranker
    GTE_RERANKER_ENABLED: bool = True
    GTE_RERANKER_MODEL: str = "Alibaba-NLP/gte-reranker-modernbert-base"
    GTE_RERANKER_MAX_LENGTH: int = 512
    GTE_RERANKER_FALLBACK_TO_FLASHRANK: bool = True

    # v23 NLI Entailment Scoring
    NLI_RERANKING_ENABLED: bool = True
    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-base"
    NLI_WEIGHT: float = 0.3
    NLI_ONLY_FOR_OPEN_DOMAIN: bool = True

    # v24 Multi-Passage Evidence Aggregation
    MULTI_PASSAGE_RERANKING_ENABLED: bool = True
    MULTI_PASSAGE_CLUSTER_OVERLAP_THRESHOLD: float = 0.3
    MULTI_PASSAGE_MAX_CLUSTER_SIZE: int = 3

    # v25 Dual-Vector Architecture (prep only, not active until DualCSE trained)
    DUAL_VECTORS_ENABLED: bool = False
    IMPLICIT_EMBEDDING_MODEL: str = ""
    IMPLICIT_VECTOR_WEIGHT: float = 0.5

    model_config = {"env_prefix": "ZIKKARON_"}

    @property
    def db_path_resolved(self) -> Path:
        return Path(self.DB_PATH).expanduser()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
