from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Episode(BaseModel):
    id: Optional[int] = None
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    directory: str
    raw_content: str
    overlap_start: Optional[int] = None
    overlap_end: Optional[int] = None


class Entity(BaseModel):
    id: Optional[int] = None
    name: str
    type: Literal["file", "function", "variable", "dependency", "decision", "error", "solution"]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    heat: float = 1.0
    archived: bool = False
    # v2 fields
    causal_weight: float = 0.0
    domain: Optional[str] = None


class Relationship(BaseModel):
    id: Optional[int] = None
    source_entity_id: int
    target_entity_id: int
    relationship_type: str  # co_occurrence, imports, calls, debugged_with, decided_to_use, caused_by, resolved_by, preceded_by, derived_from
    weight: float = 1.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_reinforced: datetime = Field(default_factory=datetime.utcnow)
    # v2 fields
    event_time: Optional[datetime] = None
    record_time: datetime = Field(default_factory=datetime.utcnow)
    is_causal: bool = False
    confidence: float = 1.0


class Memory(BaseModel):
    id: Optional[int] = None
    content: str
    embedding: Optional[bytes] = None
    tags: list[str] = Field(default_factory=list)
    source_episode_id: Optional[int] = None
    directory_context: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    heat: float = 1.0
    is_stale: bool = False
    file_hash: Optional[str] = None
    # v2 fields
    surprise_score: float = 0.0
    importance: float = 0.5
    emotional_valence: float = 0.0
    confidence: float = 1.0
    access_count: int = 0
    useful_count: int = 0
    embedding_model: Optional[str] = None
    contextual_prefix: Optional[str] = None
    cluster_id: Optional[int] = None
    is_prospective: bool = False
    trigger_condition: Optional[str] = None
    narrative_weight: float = 0.0
    compressed: bool = False
    # v3 frontier fields
    plasticity: float = 1.0  # Spikes on access, decays with ~6h half-life (reconsolidation)
    stability: float = 0.0  # Increases with successful retrievals (reconsolidation)
    excitability: float = 1.0  # Competitive allocation score (engram)
    last_excitability_update: Optional[datetime] = None  # For excitability decay calc
    store_type: str = "episodic"  # "episodic" or "semantic" (CLS dual-store)
    compression_level: int = 0  # 0=full, 1=gist, 2=tag (rate-distortion)
    original_content: Optional[str] = None  # Full content preserved before compression
    hdc_vector: Optional[bytes] = None  # Hyperdimensional computing vector
    sr_x: float = 0.0  # Successor representation x coordinate (cognitive map)
    sr_y: float = 0.0  # Successor representation y coordinate (cognitive map)
    reconsolidation_count: int = 0  # Number of times reconsolidated
    last_reconsolidated: Optional[datetime] = None  # Timestamp of last reconsolidation
    provenance_agent: str = "default"  # Which agent created this memory (CRDT)
    vector_clock: str = "{}"  # JSON vector clock for CRDT conflict resolution
    is_protected: bool = False  # Protected from modification and compression


class ConsolidationLog(BaseModel):
    id: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    memories_added: int = 0
    memories_updated: int = 0
    memories_archived: int = 0
    memories_deleted: int = 0
    duration_ms: int = 0


class FileHash(BaseModel):
    id: Optional[int] = None
    filepath: str
    hash: str
    last_checked: datetime = Field(default_factory=datetime.utcnow)


class MemoryStats(BaseModel):
    total_memories: int
    active_count: int
    archived_count: int
    stale_count: int
    avg_heat: float
    last_consolidation: Optional[datetime] = None


# -- v2 models --


class MemoryCluster(BaseModel):
    id: Optional[int] = None
    name: str
    level: int = 0  # 0=leaf, 1=intermediate, 2=root
    parent_cluster_id: Optional[int] = None
    summary: str = ""
    centroid_embedding: Optional[bytes] = None
    member_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    heat: float = 1.0


class ProspectiveMemory(BaseModel):
    id: Optional[int] = None
    content: str
    trigger_condition: str
    trigger_type: Literal["directory_match", "keyword_match", "entity_match", "time_based"]
    target_directory: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    triggered_at: Optional[datetime] = None
    triggered_count: int = 0


class NarrativeEntry(BaseModel):
    id: Optional[int] = None
    directory_context: str
    summary: str
    period_start: datetime
    period_end: datetime
    key_decisions: list[str] = Field(default_factory=list)
    key_events: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    heat: float = 1.0


class AstrocyteProcess(BaseModel):
    id: Optional[int] = None
    name: str
    domain: str
    specialization: str = ""
    memory_ids: list[int] = Field(default_factory=list)
    entity_ids: list[int] = Field(default_factory=list)
    heat: float = 1.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)


# -- v3 frontier models --


class MemoryRule(BaseModel):
    id: Optional[int] = None
    rule_type: Literal["hard", "soft"]  # hard = must satisfy, soft = preference
    scope: Literal["global", "directory", "file"]  # where rule applies
    scope_value: Optional[str] = None  # directory path or file pattern for scoped rules
    condition: str  # e.g. "language == typescript", "tag contains architecture"
    action: str  # e.g. "filter", "boost:0.3", "penalty:0.2"
    priority: int = 0  # higher = applied first
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True


class MemoryArchive(BaseModel):
    id: Optional[int] = None
    original_memory_id: int
    content: str
    embedding: Optional[bytes] = None
    archived_at: datetime = Field(default_factory=datetime.utcnow)
    mismatch_score: float = 0.0
    archive_reason: str = ""  # "reconsolidation", "compression", "extinction"


class MemoryTransition(BaseModel):
    id: Optional[int] = None
    from_memory_id: int
    to_memory_id: int
    count: int = 1
    last_transition: datetime = Field(default_factory=datetime.utcnow)
    session_id: str = ""


class CausalDAGEdge(BaseModel):
    id: Optional[int] = None
    source_entity_id: int
    target_entity_id: int
    algorithm: str = "pc"  # "pc", "ges", "heuristic"
    confidence: float = 1.0
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    is_validated: bool = False


# -- v4 models --


class Checkpoint(BaseModel):
    id: Optional[int] = None
    session_id: str = "default"
    directory_context: str
    current_task: str = ""
    files_being_edited: list[str] = Field(default_factory=list)
    key_decisions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    active_errors: list[str] = Field(default_factory=list)
    custom_context: str = ""
    epoch: int = 0  # compaction epoch counter
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True  # only latest checkpoint is active
