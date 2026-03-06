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

    model_config = {"env_prefix": "ZIKKARON_"}

    @property
    def db_path_resolved(self) -> Path:
        return Path(self.DB_PATH).expanduser()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
