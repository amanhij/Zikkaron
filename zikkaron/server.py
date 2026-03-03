"""Zikkaron MCP server — supports SSE and Streamable HTTP transports."""

import hashlib
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from zikkaron.astrocyte_pool import AstrocytePool
from zikkaron.cls_store import DualStoreCLS
from zikkaron.cognitive_map import CognitiveMap
from zikkaron.compression import MemoryCompressor
from zikkaron.config import Settings, get_settings
from zikkaron.consolidation import AstrocyteEngine
from zikkaron.crdt_sync import CRDTMemorySync
from zikkaron.engram import EngramAllocator
from zikkaron.causal_discovery import CausalDiscovery
from zikkaron.curation import MemoryCurator
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.fractal import FractalMemoryTree
from zikkaron.hdc_encoder import HDCEncoder
from zikkaron.hopfield import HopfieldMemory
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.metacognition import MetaCognition
from zikkaron.narrative import NarrativeEngine
from zikkaron.predictive_coding import PredictiveCodingGate
from zikkaron.prospective import ProspectiveMemoryEngine
from zikkaron.reconsolidation import ReconsolidationEngine
from zikkaron.retrieval import HippoRetriever
from zikkaron.rules_engine import RulesEngine
from zikkaron.sensory_buffer import SensoryBuffer
from zikkaron.sleep_compute import SleepComputeEngine
from zikkaron.staleness import StalenessDetector
from zikkaron.storage import StorageEngine
from zikkaron.thermodynamics import MemoryThermodynamics

logger = logging.getLogger(__name__)

# Global instances — initialized in main()
_storage: StorageEngine | None = None
_embeddings: EmbeddingEngine | None = None
_buffer: SensoryBuffer | None = None
_consolidation: AstrocyteEngine | None = None
_staleness: StalenessDetector | None = None
_thermo: MemoryThermodynamics | None = None
_retriever: HippoRetriever | None = None
_curator: MemoryCurator | None = None
_prospective: ProspectiveMemoryEngine | None = None
_narrative: NarrativeEngine | None = None
_sleep: SleepComputeEngine | None = None
_fractal: FractalMemoryTree | None = None
_pool: AstrocytePool | None = None
_kg: KnowledgeGraph | None = None
_reconsolidation: ReconsolidationEngine | None = None
_write_gate: PredictiveCodingGate | None = None
_engram: EngramAllocator | None = None
_rules_engine: RulesEngine | None = None
_hopfield: HopfieldMemory | None = None
_cls: DualStoreCLS | None = None
_compressor: MemoryCompressor | None = None
_hdc: HDCEncoder | None = None
_cognitive_map: CognitiveMap | None = None
_causal: CausalDiscovery | None = None
_metacognition: MetaCognition | None = None
_crdt: CRDTMemorySync | None = None

# Session state for transition tracking
_last_recalled_ids: dict[str, int] = {}  # session_id → last recalled memory_id

# Transport type used by the running server
_active_transport: str = "sse"

# Server start timestamp for uptime tracking
_start_time: float = 0.0

settings = get_settings()

mcp_server = FastMCP(
    name="zikkaron",
    instructions="Biologically-inspired persistent memory engine for Claude Code.",
    host="127.0.0.1",
    port=settings.PORT,
)


# ── Custom HTTP Endpoints ─────────────────────────────────────────────


@mcp_server.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint."""
    from zikkaron import __version__

    session_count = 0
    if mcp_server._session_manager is not None:
        session_count = len(mcp_server._session_manager._server_instances)

    return JSONResponse({
        "status": "ok",
        "version": __version__,
        "transport": _active_transport,
        "uptime_seconds": round(time.time() - _start_time, 1) if _start_time else 0,
        "active_sessions": session_count,
    })


def _get_storage() -> StorageEngine:
    assert _storage is not None, "StorageEngine not initialized"
    return _storage


def _get_embeddings() -> EmbeddingEngine:
    assert _embeddings is not None, "EmbeddingEngine not initialized"
    return _embeddings


def _get_buffer() -> SensoryBuffer:
    assert _buffer is not None, "SensoryBuffer not initialized"
    return _buffer


def _get_consolidation() -> AstrocyteEngine:
    assert _consolidation is not None, "AstrocyteEngine not initialized"
    return _consolidation


def _get_staleness() -> StalenessDetector:
    assert _staleness is not None, "StalenessDetector not initialized"
    return _staleness


def _get_thermo() -> MemoryThermodynamics:
    assert _thermo is not None, "MemoryThermodynamics not initialized"
    return _thermo


def _get_retriever() -> HippoRetriever:
    assert _retriever is not None, "HippoRetriever not initialized"
    return _retriever


def _get_reconsolidation() -> ReconsolidationEngine:
    assert _reconsolidation is not None, "ReconsolidationEngine not initialized"
    return _reconsolidation


def _get_write_gate() -> PredictiveCodingGate:
    assert _write_gate is not None, "PredictiveCodingGate not initialized"
    return _write_gate


def _get_engram() -> EngramAllocator:
    assert _engram is not None, "EngramAllocator not initialized"
    return _engram


def _get_crdt() -> CRDTMemorySync:
    assert _crdt is not None, "CRDTMemorySync not initialized"
    return _crdt


def _get_cognitive_map() -> CognitiveMap:
    assert _cognitive_map is not None, "CognitiveMap not initialized"
    return _cognitive_map


def _file_hash(filepath: str) -> str | None:
    """Compute SHA-256 hash of a file if it exists."""
    p = Path(filepath).expanduser()
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ── MCP Tools ──────────────────────────────────────────────────────────


@mcp_server.tool()
def remember(content: str, context: str, tags: list[str]) -> dict:
    """Store a new memory with embedding and optional file hash."""
    storage = _get_storage()
    embeddings = _get_embeddings()
    buffer = _get_buffer()

    # Predictive coding write gate — FIRST check before any storage
    gate_result = None
    if _write_gate is not None:
        should_store, surprisal, reason = _write_gate.should_store(
            content, context, tags
        )
        gate_result = {
            "surprisal": round(surprisal, 4),
            "gate_reason": reason,
        }
        if not should_store:
            return {
                "stored": False,
                "surprisal": round(surprisal, 4),
                "reason": reason,
                "message": "Memory below surprisal threshold, skipped",
            }

    # Generate contextual prefix for richer embedding semantics
    contextual_prefix = None
    retriever = _retriever
    if retriever is not None and settings.CONTEXTUAL_PREFIX_ENABLED:
        from datetime import datetime, timezone
        contextual_prefix = retriever.generate_contextual_prefix(
            content, context, tags, datetime.now(timezone.utc)
        )

    # Embed with contextual prefix prepended if available
    embed_text = f"{contextual_prefix}{content}" if contextual_prefix else content
    embedding = embeddings.encode(embed_text)
    fhash = _file_hash(context)

    # Compute thermodynamic scores
    thermo = _thermo
    if thermo is not None:
        surprise = thermo.compute_surprise(content, context)
        importance = thermo.compute_importance(content, tags)
        valence = thermo.compute_valence(content)
        initial_heat = thermo.apply_surprise_boost(1.0, surprise)
    else:
        surprise = 0.0
        importance = 0.5
        valence = 0.0
        initial_heat = 1.0

    # CRDT provenance tagging — stamp agent ID and vector clock
    crdt = _crdt
    crdt_provenance = {}
    if crdt is not None:
        crdt_provenance = {
            "provenance_agent": crdt.get_agent_id(),
            "vector_clock": json.dumps(crdt.increment_clock()),
        }

    # Use curator for intelligent ingestion (merge/link/create)
    curator = _curator
    if curator is not None and embedding is not None:
        curation_result = curator.curate_on_remember(
            content, context, tags, embedding,
            initial_heat=initial_heat,
            surprise=surprise,
            importance=importance,
            valence=valence,
            file_hash=fhash,
            embedding_model=embeddings.get_model_name(),
            contextual_prefix=contextual_prefix,
        )
        memory_id = curation_result["memory_id"]
        curation_action = curation_result["action"]
    else:
        # Fallback: direct insert (no curator or no embedding)
        memory_id = storage.insert_memory(
            {
                "content": content,
                "embedding": embedding,
                "tags": tags,
                "directory_context": context,
                "heat": initial_heat,
                "is_stale": False,
                "file_hash": fhash,
                "embedding_model": embeddings.get_model_name(),
            }
        )

        if contextual_prefix:
            storage._conn.execute(
                "UPDATE memories SET contextual_prefix = ? WHERE id = ?",
                (contextual_prefix, memory_id),
            )
            storage._conn.commit()

        storage.update_memory_scores(
            memory_id,
            surprise_score=surprise,
            importance=importance,
            emotional_valence=valence,
        )
        curation_action = "created"

    # Apply CRDT provenance to the stored memory
    if crdt_provenance:
        storage._conn.execute(
            "UPDATE memories SET provenance_agent = ?, vector_clock = ? WHERE id = ?",
            (crdt_provenance["provenance_agent"], crdt_provenance["vector_clock"], memory_id),
        )
        storage._conn.commit()

    # CLS dual-store: classify memory as episodic or semantic
    if _consolidation is not None and _consolidation.cls is not None:
        store_type = _consolidation.cls.classify_memory(content, tags, context)
        storage._conn.execute(
            "UPDATE memories SET store_type = ? WHERE id = ?",
            (store_type, memory_id),
        )
        storage._conn.commit()

    # Register file hash so staleness detector can find the filepath later
    if fhash is not None:
        storage.upsert_file_hash(context, fhash)

    # Capture in sensory buffer
    buffer.capture(content, context)

    # Record activity on consolidation engine
    if _consolidation is not None:
        _consolidation.record_activity()

    # Assign to astrocyte processes for domain-aware consolidation
    if _pool is not None:
        mem_data = storage.get_memory(memory_id)
        if mem_data:
            _pool.assign_memory(mem_data)

    # Synaptic boost for high-importance memories
    if thermo is not None and importance > 0.7:
        thermo.synaptic_boost(memory_id, initial_heat)

    # Prospective memory: auto-create triggers from content & check existing triggers
    triggered_memories = []
    if _prospective is not None:
        _prospective.auto_create_from_content(content, context)

        from datetime import datetime as _dt, timezone as _tz
        trigger_context = {
            "directory": context,
            "content": content,
            "entities": tags,
            "current_time": _dt.now(_tz.utc),
        }
        triggered_memories = _prospective.check_triggers(trigger_context)

    # Engram allocation — competitive slot assignment with temporal linking
    engram_result = None
    if _engram is not None:
        try:
            engram_result = _engram.allocate(memory_id)
        except Exception:
            logger.debug("Engram allocation failed for memory %s", memory_id)

    # HDC encoding — compute compositional hyperdimensional vector
    if _hdc is not None:
        try:
            from zikkaron.retrieval import _extract_query_entities
            hdc_entities = _extract_query_entities(content)
            hdc_vec = _hdc.encode_memory(
                directory=context,
                tags=tags,
                entities=hdc_entities,
                store_type="episodic",
            )
            storage._conn.execute(
                "UPDATE memories SET hdc_vector = ? WHERE id = ?",
                (_hdc.to_bytes(hdc_vec), memory_id),
            )
            storage._conn.commit()
        except Exception:
            logger.debug("HDC encoding failed for memory %s", memory_id)

    memory = storage.get_memory(memory_id)
    # Strip binary embedding from response
    memory.pop("embedding", None)
    memory["curation_action"] = curation_action
    if gate_result is not None:
        memory["surprisal"] = gate_result["surprisal"]
        memory["gate_reason"] = gate_result["gate_reason"]
    if triggered_memories:
        memory["triggered_prospective_memories"] = [
            {"id": pm["id"], "content": pm["content"]}
            for pm in triggered_memories
        ]
    if engram_result is not None:
        memory["engram_slot"] = engram_result["slot_index"]
        memory["temporal_links"] = engram_result["temporally_linked"]
        memory["temporal_link_count"] = engram_result["link_count"]
    return memory


@mcp_server.tool()
def recall(query: str, max_results: int = 5, min_heat: float = 0.1) -> list[dict]:
    """Semantic + keyword search filtered by heat. Boosts accessed memories."""
    storage = _get_storage()

    # Record activity on consolidation engine
    if _consolidation is not None:
        _consolidation.record_activity()

    # Use HippoRetriever for unified 4-signal recall
    retriever = _retriever
    if retriever is not None:
        merged = retriever.recall(query, max_results=max_results, min_heat=min_heat)
    else:
        # Fallback to basic FTS + vector if retriever not initialized
        embeddings = _get_embeddings()
        try:
            fts_results = storage.search_memories_fts(
                query, min_heat=min_heat, limit=max_results * 2
            )
        except Exception:
            fts_results = []

        semantic_results = []
        query_embedding = embeddings.encode(query)
        if query_embedding is not None:
            vec_hits = storage.search_vectors(
                query_embedding, top_k=max_results * 2, min_heat=min_heat
            )
            for mid, _distance in vec_hits:
                mem = storage.get_memory(mid)
                if mem:
                    semantic_results.append(mem)

        seen = set()
        merged = []
        for m in fts_results + semantic_results:
            if m["id"] not in seen:
                seen.add(m["id"])
                merged.append(m)

        merged.sort(
            key=lambda m: m["heat"] * m.get("confidence", 1.0),
            reverse=True,
        )
        merged = merged[:max_results]
        for m in merged:
            m.pop("embedding", None)

    # Boost heat, update last_accessed, and record metamemory access
    now = storage._now_iso()
    thermo = _thermo
    for m in merged:
        new_heat = min(m["heat"] + 0.1, 1.0)
        storage.update_memory_heat(m["id"], new_heat)
        storage._conn.execute(
            "UPDATE memories SET last_accessed = ? WHERE id = ?", (now, m["id"])
        )
        m["heat"] = new_heat
        m["last_accessed"] = now
        if thermo is not None:
            thermo.record_access(m["id"], was_useful=True)
    storage._conn.commit()

    # Record SR transitions: link previous recall → current recall
    if _cognitive_map is not None and merged:
        session_key = "default"
        top_id = merged[0]["id"]
        prev_id = _last_recalled_ids.get(session_key)
        if prev_id is not None and prev_id != top_id:
            try:
                _cognitive_map.record_transition(prev_id, top_id, session_key)
                _cognitive_map.incremental_update(prev_id, top_id)
            except Exception:
                logger.debug("SR transition recording failed")
        _last_recalled_ids[session_key] = top_id

    # Reconsolidate: retrieved memories become labile and may be updated
    # This happens AFTER scoring, so it doesn't affect the current recall
    if _reconsolidation is not None:
        for m in merged:
            try:
                _reconsolidation.reconsolidate(m["id"], query, "")
            except Exception:
                logger.debug("Reconsolidation failed for memory %s", m.get("id"))

    # Strip binary embeddings from response (in case any slipped through)
    for m in merged:
        m.pop("embedding", None)

    return merged


@mcp_server.tool()
def forget(memory_id: int) -> dict:
    """Mark a memory for deletion by setting heat to 0, then delete it."""
    storage = _get_storage()
    memory = storage.get_memory(memory_id)
    if memory is None:
        return {"memory_id": memory_id, "status": "not_found"}
    storage.delete_memory(memory_id)
    return {"memory_id": memory_id, "status": "deleted"}


@mcp_server.tool()
def validate_memory(memory_id: int) -> dict:
    """Check memory validity against current file state."""
    if _staleness is not None:
        result = _staleness.validate_memory(memory_id)
        # Normalize response format for the MCP tool
        return {
            "memory_id": memory_id,
            "is_valid": result["valid"],
            "reason": result["reason"],
        }

    # Fallback if staleness detector not initialized
    storage = _get_storage()
    memory = storage.get_memory(memory_id)
    if memory is None:
        return {"memory_id": memory_id, "is_valid": False, "reason": "memory not found"}

    if not memory.get("file_hash"):
        return {"memory_id": memory_id, "is_valid": True, "reason": "no file hash to validate"}

    current_hash = _file_hash(memory["directory_context"])
    if current_hash is None:
        storage.update_memory_staleness(memory_id, True)
        return {"memory_id": memory_id, "is_valid": False, "reason": "file no longer exists"}

    if current_hash != memory["file_hash"]:
        storage.update_memory_staleness(memory_id, True)
        return {"memory_id": memory_id, "is_valid": False, "reason": "file has changed"}

    return {"memory_id": memory_id, "is_valid": True, "reason": "file hash matches"}


@mcp_server.tool()
def get_project_context(directory: str) -> list[dict]:
    """Return all hot memories for a directory, sorted by heat descending."""
    storage = _get_storage()
    memories = storage.get_memories_for_directory(directory, min_heat=settings.HOT_THRESHOLD)
    for m in memories:
        m.pop("embedding", None)
    return memories


@mcp_server.tool()
def consolidate_now() -> dict:
    """Trigger an immediate consolidation cycle."""
    if _consolidation is not None:
        stats = _consolidation.force_consolidate()
        # Also run memify cycle (already included in force_consolidate via _consolidation_cycle)
        # Run sleep-time compute if available
        if _sleep is not None:
            try:
                sleep_stats = _sleep.run_sleep_cycle()
                stats["sleep_cycle"] = sleep_stats
            except Exception:
                logger.exception("Sleep cycle failed during consolidate_now")
        return {"status": "completed", **stats}
    return {"status": "error", "message": "Consolidation engine not initialized"}


@mcp_server.tool()
def memory_stats() -> dict:
    """Return system memory statistics."""
    storage = _get_storage()
    stats = storage.get_memory_stats()

    # Frontier metrics
    if _hopfield is not None:
        stats["hopfield_patterns"] = _hopfield.get_pattern_count()

    if _reconsolidation is not None:
        recon_count = storage._conn.execute(
            "SELECT COALESCE(SUM(reconsolidation_count), 0) FROM memories"
        ).fetchone()[0]
        stats["reconsolidation_count"] = recon_count

    if _write_gate is not None:
        # Track rejections via memories with surprisal below threshold
        stats["write_gate_rejections"] = getattr(_write_gate, "_rejection_count", 0)

    if _engram is not None:
        try:
            slot_stats = _engram.get_slot_statistics()
            total = slot_stats.get("total_slots", 1)
            occupied = slot_stats.get("occupied_slots", 0)
            stats["engram_slot_utilization"] = round(occupied / max(total, 1), 4)
        except Exception:
            stats["engram_slot_utilization"] = 0.0

    if _rules_engine is not None:
        active_rules = _rules_engine.get_all_rules()
        stats["active_rules"] = len(active_rules)

    if _cls is not None:
        ep_count = storage._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE store_type = 'episodic' AND heat > 0"
        ).fetchone()[0]
        sem_count = storage._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE store_type = 'semantic' AND heat > 0"
        ).fetchone()[0]
        stats["episodic_count"] = ep_count
        stats["semantic_count"] = sem_count

    if _compressor is not None:
        for level in (0, 1, 2):
            count = storage._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE compression_level = ? AND heat > 0",
                (level,),
            ).fetchone()[0]
            stats[f"compressed_level_{level}"] = count

    if _cognitive_map is not None:
        stats["sr_dimensions"] = "active" if _cognitive_map.has_sufficient_data() else "insufficient_data"

    if _causal is not None:
        causal_edges = storage.get_all_causal_edges()
        stats["causal_edges"] = len(causal_edges)

    if _metacognition is not None:
        # Average coverage across recent queries isn't tracked globally,
        # but we can report the chunk limit setting
        stats["cognitive_load_limit"] = _metacognition._chunk_limit

    if _crdt is not None:
        crdt_stats = _crdt.get_agent_stats()
        stats["agent_id"] = crdt_stats["agent_id"]
        stats["conflict_count"] = crdt_stats["conflicts_pending"]
        stats["crdt"] = crdt_stats

    return stats


@mcp_server.tool()
def rate_memory(memory_id: int, was_useful: bool) -> dict:
    """Rate a memory's usefulness for metamemory tracking."""
    storage = _get_storage()
    thermo = _get_thermo()

    mem = storage.get_memory(memory_id)
    if mem is None:
        return {"memory_id": memory_id, "status": "not_found"}

    thermo.record_access(memory_id, was_useful)

    # Update reconsolidation stability based on usefulness
    if _reconsolidation is not None:
        _reconsolidation.update_stability(memory_id, was_useful)

    updated = storage.get_memory(memory_id)
    return {
        "memory_id": memory_id,
        "status": "rated",
        "was_useful": was_useful,
        "access_count": updated.get("access_count", 0),
        "useful_count": updated.get("useful_count", 0),
        "confidence": updated.get("confidence", 1.0),
        "stability": updated.get("stability", 0.0),
    }


@mcp_server.tool()
def recall_hierarchical(
    query: str, level: int = None, max_results: int = 10
) -> list[dict]:
    """Retrieve memories from the fractal hierarchy at a specific level or adaptively."""
    retriever = _get_retriever()
    return retriever.recall_hierarchical(query, level=level, max_results=max_results)


@mcp_server.tool()
def drill_down(cluster_id: int) -> list[dict]:
    """Drill into a cluster to see its members."""
    retriever = _get_retriever()
    return retriever._fractal.drill_down(cluster_id)


@mcp_server.tool()
def create_trigger(
    content: str,
    trigger_condition: str,
    trigger_type: str,
    target_directory: str | None = None,
) -> dict:
    """Create a prospective memory trigger that fires on matching context."""
    if _prospective is None:
        return {"status": "error", "message": "ProspectiveMemoryEngine not initialized"}
    pm_id = _prospective.create_trigger(
        content, trigger_condition, trigger_type, target_directory,
    )
    return {"status": "created", "prospective_memory_id": pm_id}


@mcp_server.tool()
def get_project_story(directory: str) -> str:
    """Get the autobiographical narrative for a project directory."""
    if _narrative is None:
        return "NarrativeEngine not initialized"
    return _narrative.get_project_story(directory)


@mcp_server.tool()
def add_rule(
    rule_type: str,
    scope: str,
    condition: str,
    action: str,
    priority: int = 0,
    scope_value: str = "",
) -> dict:
    """Add a neuro-symbolic rule for filtering/re-ranking memories.

    rule_type: "hard" (must satisfy) or "soft" (preference).
    scope: "global", "directory", or "file".
    condition: e.g. "importance > 0.7", "tag contains architecture".
    action: "filter" for hard rules, "boost:0.3" or "penalty:0.2" for soft rules.
    priority: Higher = applied first (default 0).
    scope_value: Directory path or file pattern for scoped rules.
    """
    if _rules_engine is None:
        return {"status": "error", "message": "RulesEngine not initialized"}
    try:
        rule_id = _rules_engine.add_rule(
            rule_type=rule_type,
            scope=scope,
            condition=condition,
            action=action,
            priority=priority,
            scope_value=scope_value or None,
        )
        return {"status": "created", "rule_id": rule_id}
    except ValueError as e:
        return {"status": "error", "message": str(e)}


@mcp_server.tool()
def get_rules(directory: str = "") -> list[dict]:
    """Get active rules. If directory is provided, returns only applicable rules."""
    if _rules_engine is None:
        return []
    if directory:
        return _rules_engine.get_applicable_rules(directory)
    return _rules_engine.get_all_rules()


@mcp_server.tool()
def navigate_memory(query: str, top_k: int = 5) -> list[dict]:
    """Navigate concept space using Successor Representation cognitive maps.

    Instead of nearest-neighbor search, this navigates to the query's projected
    location in SR space — memories accessed in similar CONTEXTS cluster together,
    even if their CONTENT differs.
    """
    if _cognitive_map is None:
        return [{"error": "CognitiveMap not initialized"}]

    if not _cognitive_map.has_sufficient_data():
        return [{"info": "Insufficient transition data for SR navigation (need >= 20)"}]

    embeddings = _get_embeddings()
    query_embedding = embeddings.encode(query)
    if query_embedding is None:
        return [{"error": "Failed to encode query"}]

    results = _cognitive_map.navigate_to(query_embedding, embeddings, top_k=top_k)
    if not results:
        return []

    storage = _get_storage()
    output = []
    for mid, proximity in results:
        mem = storage.get_memory(mid)
        if mem:
            mem.pop("embedding", None)
            mem["sr_proximity"] = round(proximity, 4)
            output.append(mem)

    return output


@mcp_server.tool()
def get_causal_chain(entity: str) -> dict:
    """Get causal causes and effects for an entity from the PC algorithm DAG."""
    if _causal is None:
        return {"error": "CausalDiscovery not initialized"}
    return _causal.get_causal_chain(entity)


@mcp_server.tool()
def assess_coverage(query: str, directory: str = "") -> dict:
    """Assess how well Zikkaron knows about a topic.

    Returns coverage score (0-1), confidence, suggestion
    (sufficient/partial/insufficient), identified gaps, and signal breakdowns.
    """
    if _metacognition is None:
        return {"error": "MetaCognition not initialized"}
    return _metacognition.assess_coverage(query, directory)


@mcp_server.tool()
def detect_gaps(directory: str) -> list[dict]:
    """Detect knowledge gaps for a project directory.

    Returns list of gaps with type (isolated_entity, stale_region,
    low_confidence, missing_connection, one_sided_knowledge),
    description, severity, affected entities, and suggestions.
    """
    if _metacognition is None:
        return [{"error": "MetaCognition not initialized"}]
    return _metacognition.detect_gaps(directory)


# ── MCP Resources ──────────────────────────────────────────────────────


@mcp_server.resource("memory://stats")
def resource_stats() -> str:
    """Live memory statistics."""
    storage = _get_storage()
    return json.dumps(storage.get_memory_stats())


@mcp_server.resource("memory://hot")
def resource_hot() -> str:
    """All memories with heat >= HOT_THRESHOLD."""
    storage = _get_storage()
    memories = storage.get_memories_by_heat(settings.HOT_THRESHOLD)
    for m in memories:
        m.pop("embedding", None)
    return json.dumps(memories, default=str)


@mcp_server.resource("memory://stale")
def resource_stale() -> str:
    """All stale memories."""
    storage = _get_storage()
    memories = storage.get_stale_memories()
    for m in memories:
        m.pop("embedding", None)
    return json.dumps(memories, default=str)


@mcp_server.resource("memory://processes")
def resource_processes() -> str:
    """List of astrocyte process stats."""
    consolidation = _get_consolidation()
    pool = consolidation.pool
    if pool is None:
        return json.dumps([])
    return json.dumps(pool.get_process_stats(), default=str)


@mcp_server.resource("memory://narrative/{directory}")
def resource_narrative(directory: str) -> str:
    """Project story for a directory."""
    if _narrative is None:
        return json.dumps({"error": "NarrativeEngine not initialized"})
    return _narrative.get_project_story(directory)


# ── Startup ────────────────────────────────────────────────────────────


def init_engines(
    db_path: str | None = None,
    embedding_model: str | None = None,
    start_daemons: bool = False,
    watch_directory: str | None = None,
):
    """Initialize all engines. Returns (storage, embeddings, buffer, consolidation, staleness)."""
    global _storage, _embeddings, _buffer, _consolidation, _staleness, _thermo, _retriever, _curator
    global _prospective, _narrative, _sleep, _fractal, _pool, _kg, _reconsolidation, _write_gate, _engram
    global _rules_engine, _hopfield, _cls, _compressor, _hdc, _cognitive_map, _causal, _metacognition, _crdt

    _settings = get_settings()
    _storage = StorageEngine(db_path or _settings.DB_PATH)
    _embeddings = EmbeddingEngine(embedding_model or _settings.EMBEDDING_MODEL)
    _buffer = SensoryBuffer(_storage, _settings)
    _buffer.start_session()
    _thermo = MemoryThermodynamics(_storage, _embeddings, _settings)
    _kg = KnowledgeGraph(_storage, _settings)
    _hdc = HDCEncoder(dimensions=_settings.HDC_DIMENSIONS)
    _cognitive_map = CognitiveMap(_storage, _settings)
    _retriever = HippoRetriever(_storage, _embeddings, _kg, _settings)
    _retriever.set_hdc(_hdc)
    _retriever.set_cognitive_map(_cognitive_map)
    _curator = MemoryCurator(_storage, _embeddings, _thermo, _settings)
    _consolidation = AstrocyteEngine(_storage, _embeddings, _settings)
    _staleness = StalenessDetector(_storage, _settings)
    _prospective = ProspectiveMemoryEngine(_storage, _settings)
    _narrative = NarrativeEngine(_storage, _kg, _settings)
    _reconsolidation = ReconsolidationEngine(_storage, _embeddings, _settings)
    _write_gate = PredictiveCodingGate(_storage, _embeddings, _retriever, _settings)
    _engram = EngramAllocator(_storage, _settings)
    _rules_engine = RulesEngine(_storage, _settings)
    _causal = CausalDiscovery(_storage, _kg, _settings)
    _metacognition = MetaCognition(_storage, _embeddings, _kg, _settings)
    _crdt = CRDTMemorySync(_storage, _settings)
    _retriever.set_engram(_engram)
    _retriever.set_rules_engine(_rules_engine)
    _retriever.set_metacognition(_metacognition)

    # Expose inner engines as server-level globals for direct access
    _sleep = _consolidation._sleep_engine
    _fractal = _retriever._fractal
    _pool = _consolidation.pool
    _hopfield = _retriever._hopfield
    _cls = _consolidation.cls
    _compressor = _consolidation._compressor

    if start_daemons:
        _consolidation.start()
        if watch_directory:
            _staleness.start(watch_directory)

    return _storage, _embeddings, _buffer, _consolidation, _staleness


def shutdown():
    """Gracefully shut down all engines."""
    global _storage, _embeddings, _buffer, _consolidation, _staleness, _thermo, _retriever, _curator
    global _prospective, _narrative, _sleep, _fractal, _pool, _kg, _reconsolidation, _write_gate, _engram
    global _rules_engine, _hopfield, _cls, _compressor, _hdc, _cognitive_map, _causal, _metacognition, _crdt

    if _consolidation is not None:
        _consolidation.stop()
    if _staleness is not None:
        _staleness.stop()
    if _buffer is not None:
        _buffer.flush()
    if _storage is not None:
        _storage.close()

    _storage = None
    _embeddings = None
    _buffer = None
    _consolidation = None
    _staleness = None
    _thermo = None
    _retriever = None
    _curator = None
    _prospective = None
    _narrative = None
    _sleep = None
    _fractal = None
    _pool = None
    _kg = None
    _reconsolidation = None
    _write_gate = None
    _engram = None
    _rules_engine = None
    _hopfield = None
    _cls = None
    _compressor = None
    _hdc = None
    _cognitive_map = None
    _causal = None
    _metacognition = None
    _crdt = None


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    logger.info("Received signal %s, shutting down...", signum)
    shutdown()
    sys.exit(0)


def main(
    port: int | None = None,
    db_path: str | None = None,
    transport: str = "sse",
):
    global _active_transport, _start_time

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    _active_transport = transport
    _start_time = time.time()

    cwd = os.getcwd()
    init_engines(
        db_path=db_path,
        start_daemons=True,
        watch_directory=cwd,
    )

    if port is not None:
        mcp_server.settings.port = port

    try:
        mcp_server.run(transport=transport)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
