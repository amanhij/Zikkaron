"""HippoRAG-style retrieval using Personalized PageRank over the knowledge graph."""

import logging
import os
import re
from collections import defaultdict
from datetime import datetime

import networkx as nx

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.fractal import FractalMemoryTree
from zikkaron.hopfield import HopfieldMemory
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.storage import StorageEngine

# Lazy import to avoid circular dependency
_RulesEngine = None


def _get_rules_engine_class():
    global _RulesEngine
    if _RulesEngine is None:
        from zikkaron.rules_engine import RulesEngine
        _RulesEngine = RulesEngine
    return _RulesEngine

# Lazy import to avoid circular dependency
_EngramAllocator = None


def _get_engram_class():
    global _EngramAllocator
    if _EngramAllocator is None:
        from zikkaron.engram import EngramAllocator
        _EngramAllocator = EngramAllocator
    return _EngramAllocator

logger = logging.getLogger(__name__)

# Lightweight entity extraction for queries (subset of consolidation patterns)
_WORD_RE = re.compile(r"\b[A-Z][\w]*(?:[A-Z][\w]*)*\b")  # CamelCase
_PATH_RE = re.compile(r"(?:\.{0,2}/)?(?:[\w@.-]+/)+[\w@.-]+\.\w+")
_DOTTED_RE = re.compile(r"\b\w+(?:\.\w+){1,}\b")  # dotted.names
_ERROR_RE = re.compile(r"\b\w*(?:Error|Exception)\b")
_KEYWORD_RE = re.compile(r"\b[a-z_]\w{2,}\b")


def _extract_query_entities(query: str) -> list[str]:
    """Extract key concepts/entities from a query string."""
    entities: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name and name not in seen and len(name) > 1:
            seen.add(name)
            entities.append(name)

    for m in _PATH_RE.finditer(query):
        _add(m.group(0))
    for m in _DOTTED_RE.finditer(query):
        _add(m.group(0))
    for m in _ERROR_RE.finditer(query):
        _add(m.group(0))
    for m in _WORD_RE.finditer(query):
        _add(m.group(0))
    # Also split on whitespace for simple keyword matching
    for word in query.split():
        cleaned = word.strip(".,;:!?()[]{}\"'")
        if len(cleaned) > 2:
            _add(cleaned)

    return entities


class HippoRetriever:
    """HippoRAG-style retrieval combining PPR, spreading activation,
    vector similarity, and FTS5 keyword search."""

    def __init__(
        self,
        storage: StorageEngine,
        embeddings: EmbeddingEngine,
        knowledge_graph: KnowledgeGraph,
        settings: Settings,
    ) -> None:
        self._storage = storage
        self._embeddings = embeddings
        self._graph = knowledge_graph
        self._settings = settings
        self._fractal = FractalMemoryTree(storage, embeddings, settings)
        self._hopfield = HopfieldMemory(storage, embeddings, settings)
        self._engram = None  # Set externally via set_engram()
        self._rules_engine = None  # Set externally via set_rules_engine()
        self._hdc = None  # Set externally via set_hdc()
        self._cognitive_map = None  # Set externally via set_cognitive_map()
        self._metacognition = None  # Set externally via set_metacognition()

    def set_engram(self, engram) -> None:
        """Attach an EngramAllocator for temporal linking in recall results."""
        self._engram = engram

    def set_rules_engine(self, rules_engine) -> None:
        """Attach a RulesEngine for neuro-symbolic filtering/re-ranking."""
        self._rules_engine = rules_engine

    def set_hdc(self, hdc) -> None:
        """Attach an HDCEncoder for compositional structured queries."""
        self._hdc = hdc

    def set_cognitive_map(self, cognitive_map) -> None:
        """Attach a CognitiveMap for SR-based navigation signal."""
        self._cognitive_map = cognitive_map

    def set_metacognition(self, metacognition) -> None:
        """Attach a MetaCognition engine for cognitive load management."""
        self._metacognition = metacognition

    # -- a. Personalized PageRank Retrieval --

    def ppr_retrieve(
        self, query: str, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Run Personalized PageRank seeded by query entities.

        Returns (memory_id, ppr_score) sorted by score descending.
        """
        # 1. Extract entities from query
        query_terms = _extract_query_entities(query)
        if not query_terms:
            return []

        # 2. Find matching entities in the knowledge graph
        seed_entity_ids: list[int] = []
        for term in query_terms:
            entity = self._storage.get_entity_by_name(term)
            if entity:
                seed_entity_ids.append(entity["id"])

        if not seed_entity_ids:
            return []

        # 3. Build a networkx graph from entity-relationship data
        G = self._build_networkx_graph(seed_entity_ids)
        if len(G) == 0:
            return []

        # 4. Run Personalized PageRank
        personalization = {
            eid: 1.0 / len(seed_entity_ids) for eid in seed_entity_ids
            if eid in G
        }
        if not personalization:
            return []

        try:
            ppr_scores = nx.pagerank(
                G,
                alpha=self._settings.PPR_DAMPING,
                personalization=personalization,
                max_iter=self._settings.PPR_ITERATIONS,
            )
        except nx.PowerIterationFailedConvergence:
            ppr_scores = nx.pagerank(
                G,
                alpha=self._settings.PPR_DAMPING,
                personalization=personalization,
                max_iter=self._settings.PPR_ITERATIONS * 2,
                tol=1e-4,
            )

        # 5. Map high-PPR entities back to their associated memories
        entity_scores = sorted(
            ppr_scores.items(), key=lambda x: x[1], reverse=True
        )

        memory_scores: dict[int, float] = defaultdict(float)
        for entity_id, score in entity_scores:
            entity = self._storage._conn.execute(
                "SELECT name FROM entities WHERE id = ?", (entity_id,)
            ).fetchone()
            if not entity:
                continue
            entity_name = entity[0]
            # Find memories containing this entity name via FTS5
            associated = self._find_memories_for_entity(entity_name)
            for mid in associated:
                memory_scores[mid] = max(memory_scores[mid], score)

        # Sort by score descending, return top_k
        ranked = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # -- b. Contextual Prefix Generation --

    def generate_contextual_prefix(
        self,
        content: str,
        directory: str,
        tags: list[str],
        timestamp: datetime,
    ) -> str:
        """Generate a contextual prefix for richer embedding semantics."""
        dir_basename = os.path.basename(directory.rstrip("/")) or directory
        tags_joined = ", ".join(tags) if tags else "none"
        timestamp_human = timestamp.strftime("%Y-%m-%d %H:%M")

        # Find top co-occurring entities for context enrichment
        top_entities = self._get_top_cooccurring_entities(content, limit=5)
        entities_str = ", ".join(top_entities) if top_entities else "none"

        return (
            f"[Project: {dir_basename}] [Directory: {directory}] "
            f"[Tags: {tags_joined}] [Recorded: {timestamp_human}] "
            f"[Related entities: {entities_str}] "
        )

    # -- c. Spreading Activation --

    def spreading_activation(
        self,
        seed_memories: list[int],
        spread_factor: float = 0.5,
        max_depth: int = 2,
    ) -> list[tuple[int, float]]:
        """Activate related memories by spreading through the entity graph.

        Returns (memory_id, activation_score) for discovered memories
        (excludes seed memories).
        """
        if not seed_memories:
            return []

        # 1. Find entities associated with seed memories
        seed_entities: set[int] = set()
        for mid in seed_memories:
            mem = self._storage.get_memory(mid)
            if not mem:
                continue
            entities = self._find_entities_in_content(mem["content"])
            seed_entities.update(entities)

        if not seed_entities:
            return []

        # 2. BFS through entity graph up to max_depth
        activated: dict[int, float] = {}  # memory_id -> activation score
        seed_memory_set = set(seed_memories)

        visited_entities: set[int] = set(seed_entities)
        frontier: list[tuple[int, int]] = [(eid, 0) for eid in seed_entities]

        while frontier:
            next_frontier: list[tuple[int, int]] = []
            for entity_id, depth in frontier:
                if depth >= max_depth:
                    continue
                # Get connected entities
                neighbors = self._graph._get_adjacent(entity_id, None)
                for neighbor in neighbors:
                    nid = neighbor["entity_id"]
                    if nid in visited_entities:
                        continue
                    visited_entities.add(nid)
                    current_depth = depth + 1
                    activation = spread_factor ** current_depth

                    # Find memories for this neighbor entity
                    entity_row = self._storage._conn.execute(
                        "SELECT name FROM entities WHERE id = ?", (nid,)
                    ).fetchone()
                    if entity_row:
                        mids = self._find_memories_for_entity(entity_row[0])
                        for mid in mids:
                            if mid not in seed_memory_set:
                                activated[mid] = max(
                                    activated.get(mid, 0.0), activation
                                )

                    next_frontier.append((nid, current_depth))
            frontier = next_frontier

        # Sort by activation score descending
        return sorted(activated.items(), key=lambda x: x[1], reverse=True)

    # -- d. Unified Recall --

    def recall(
        self, query: str, max_results: int = 5, min_heat: float = 0.1
    ) -> list[dict]:
        """Combine up to eight retrieval signals into a unified ranked result.

        Signals and weights (normalized to sum=1.0):
        Without SR (default, < 20 transitions):
          vector=0.27, fts=0.11, ppr=0.20, spread=0.07,
          fractal=0.08, hopfield=0.14, hdc=0.13
        With SR (>= 20 transitions recorded):
          vector=0.28, fts=0.10, ppr=0.20, spread=0.07,
          hopfield=0.13, hdc=0.12, sr=0.10
        """
        # Check if SR signal is available
        sr_active = (
            self._cognitive_map is not None
            and self._cognitive_map.has_sufficient_data()
        )

        if sr_active:
            w_vector = 0.28
            w_fts = 0.10
            w_ppr = 0.20
            w_spread = 0.07
            w_fractal = 0.0
            w_hopfield = 0.13
            w_hdc = 0.12
            w_sr = 0.10
        else:
            w_vector = 0.27
            w_fts = 0.11
            w_ppr = 0.20
            w_spread = 0.07
            w_fractal = 0.08
            w_hopfield = 0.14
            w_hdc = 0.13
            w_sr = 0.0

        scores: dict[int, dict] = defaultdict(
            lambda: {
                "vector": 0.0, "fts": 0.0, "ppr": 0.0,
                "spread": 0.0, "fractal": 0.0, "hopfield": 0.0,
                "hdc": 0.0, "sr": 0.0,
            }
        )

        # 1. FTS5 keyword search
        try:
            fts_results = self._storage.search_memories_fts(
                query, min_heat=min_heat, limit=max_results * 3
            )
            if fts_results:
                # Assign descending scores based on FTS rank order
                for i, mem in enumerate(fts_results):
                    score = 1.0 / (1 + i)  # reciprocal rank
                    scores[mem["id"]]["fts"] = score
        except Exception:
            pass

        # 2. Vector similarity via sqlite-vec KNN
        query_embedding = self._embeddings.encode(query)
        vector_memory_ids: list[int] = []
        if query_embedding is not None:
            vec_hits = self._storage.search_vectors(
                query_embedding, top_k=max_results * 3, min_heat=min_heat
            )
            for mid, distance in vec_hits:
                # Convert distance to similarity score (lower distance = higher score)
                # sqlite-vec uses L2 distance; convert to 0-1 similarity
                similarity = 1.0 / (1.0 + distance)
                scores[mid]["vector"] = similarity
                vector_memory_ids.append(mid)

        # 3. PPR graph retrieval
        ppr_results = self.ppr_retrieve(query, top_k=max_results * 2)
        if ppr_results:
            max_ppr = max(s for _, s in ppr_results) if ppr_results else 1.0
            for mid, ppr_score in ppr_results:
                # Normalize PPR scores to 0-1 range
                normalized = ppr_score / max_ppr if max_ppr > 0 else 0.0
                scores[mid]["ppr"] = normalized

        # 4. Spreading activation from top vector results
        top_vector_seeds = vector_memory_ids[:5]
        if top_vector_seeds:
            spread_results = self.spreading_activation(
                top_vector_seeds, spread_factor=0.5, max_depth=2
            )
            if spread_results:
                max_spread = max(s for _, s in spread_results) if spread_results else 1.0
                for mid, spread_score in spread_results:
                    normalized = spread_score / max_spread if max_spread > 0 else 0.0
                    scores[mid]["spread"] = normalized

        # 5. Fractal cluster matching
        fractal_results = self._fractal.fractal_score(
            query, max_results=max_results * 3
        )
        if fractal_results:
            max_fractal = max(s for _, s in fractal_results) if fractal_results else 1.0
            for mid, fractal_score in fractal_results:
                normalized = fractal_score / max_fractal if max_fractal > 0 else 0.0
                scores[mid]["fractal"] = normalized

        # 6. Modern Hopfield energy-based retrieval
        if query_embedding is not None:
            try:
                hopfield_results = self._hopfield.retrieve(
                    query_embedding, top_k=max_results * 3
                )
                if hopfield_results:
                    max_hop = max(s for _, s in hopfield_results) if hopfield_results else 1.0
                    for mid, hop_score in hopfield_results:
                        normalized = hop_score / max_hop if max_hop > 0 else 0.0
                        scores[mid]["hopfield"] = normalized
            except Exception:
                logger.debug("Hopfield retrieval failed, skipping signal")

        # 7. HDC compositional query
        if self._hdc is not None:
            try:
                query_entities = _extract_query_entities(query)
                if query_entities:
                    hdc_query = self._hdc.encode_query(entities=query_entities)
                    # Load HDC vectors for candidate memories
                    candidate_ids = list(scores.keys()) if scores else []
                    if not candidate_ids:
                        # If no candidates from other signals, get recent memories
                        all_mems = self._storage._conn.execute(
                            "SELECT id FROM memories WHERE heat > ? LIMIT ?",
                            (min_heat, max_results * 3),
                        ).fetchall()
                        candidate_ids = [r[0] for r in all_mems]

                    hdc_candidates: list[tuple[int, "np.ndarray"]] = []
                    for mid in candidate_ids:
                        row = self._storage._conn.execute(
                            "SELECT hdc_vector FROM memories WHERE id = ?",
                            (mid,),
                        ).fetchone()
                        if row and row[0] is not None:
                            hdc_vec = self._hdc.from_bytes(row[0])
                            hdc_candidates.append((mid, hdc_vec))

                    if hdc_candidates:
                        hdc_results = self._hdc.search(
                            hdc_query, hdc_candidates, top_k=max_results * 3
                        )
                        if hdc_results:
                            max_hdc = max(s for _, s in hdc_results) if hdc_results else 1.0
                            for mid, hdc_score in hdc_results:
                                if max_hdc > 0:
                                    normalized = max(0.0, hdc_score) / max_hdc
                                else:
                                    normalized = 0.0
                                scores[mid]["hdc"] = normalized
            except Exception:
                logger.debug("HDC retrieval failed, skipping signal")

        # 8. Successor Representation navigation
        if sr_active and query_embedding is not None:
            try:
                candidate_ids = list(scores.keys()) if scores else []
                if candidate_ids:
                    sr_scores = self._cognitive_map.get_sr_scores(
                        query_embedding, self._embeddings, candidate_ids
                    )
                    if sr_scores:
                        max_sr = max(sr_scores.values()) if sr_scores else 1.0
                        for mid, sr_score in sr_scores.items():
                            normalized = sr_score / max_sr if max_sr > 0 else 0.0
                            scores[mid]["sr"] = normalized
            except Exception:
                logger.debug("SR retrieval failed, skipping signal")

        # Merge: compute combined score
        combined: list[tuple[int, float]] = []
        for mid, signal_scores in scores.items():
            total = (
                w_vector * signal_scores["vector"]
                + w_fts * signal_scores["fts"]
                + w_ppr * signal_scores["ppr"]
                + w_spread * signal_scores["spread"]
                + w_fractal * signal_scores["fractal"]
                + w_hopfield * signal_scores["hopfield"]
                + w_hdc * signal_scores["hdc"]
                + w_sr * signal_scores["sr"]
            )
            combined.append((mid, total))

        # Filter by min_heat and sort by combined score
        result_memories: list[dict] = []
        combined.sort(key=lambda x: x[1], reverse=True)

        for mid, total_score in combined:
            mem = self._storage.get_memory(mid)
            if mem and mem["heat"] >= min_heat:
                mem["_retrieval_score"] = round(total_score, 4)
                mem.pop("embedding", None)
                result_memories.append(mem)
            if len(result_memories) >= max_results:
                break

        # Apply neuro-symbolic rules (hard filter + soft re-rank) as final step
        if self._rules_engine is not None and result_memories:
            # Infer directory from first memory or use empty string
            directory = ""
            for mem in result_memories:
                if mem.get("directory_context"):
                    directory = mem["directory_context"]
                    break
            result_memories = self._rules_engine.apply_rules(result_memories, directory)
            # Re-trim to max_results after filtering
            result_memories = result_memories[:max_results]

        # Enrich with temporal links from engram allocation
        if self._engram is not None:
            for mem in result_memories:
                try:
                    linked = self._engram.get_temporally_linked(mem["id"])
                    if linked:
                        mem["temporal_links"] = linked
                except Exception:
                    pass

        # Apply cognitive load management via metacognition
        if self._metacognition is not None and result_memories:
            try:
                result_memories = self._metacognition.manage_context(
                    result_memories
                )
            except Exception:
                logger.debug("Metacognition manage_context failed, returning unoptimized")

        return result_memories

    # -- e. Hierarchical Recall --

    def recall_hierarchical(
        self, query: str, level: int | None = None, max_results: int = 10
    ) -> list[dict]:
        """Level-specific retrieval through the fractal hierarchy."""
        return self._fractal.retrieve_tree(query, target_level=level)[:max_results]

    # -- Internal helpers --

    def _build_networkx_graph(
        self, seed_entity_ids: list[int], max_hops: int = 3
    ) -> nx.DiGraph:
        """Build a networkx DiGraph around the seed entities."""
        G = nx.DiGraph()
        visited: set[int] = set()
        frontier = list(seed_entity_ids)

        for _ in range(max_hops):
            next_frontier: list[int] = []
            for eid in frontier:
                if eid in visited:
                    continue
                visited.add(eid)
                G.add_node(eid)
                neighbors = self._graph._get_adjacent(eid, None)
                for n in neighbors:
                    nid = n["entity_id"]
                    weight = n["weight"]
                    G.add_node(nid)
                    G.add_edge(eid, nid, weight=weight)
                    G.add_edge(nid, eid, weight=weight)
                    if nid not in visited:
                        next_frontier.append(nid)
            frontier = next_frontier

        return G

    def _find_memories_for_entity(self, entity_name: str) -> list[int]:
        """Find memory IDs whose content contains the entity name."""
        # Use FTS5 for efficient text search
        try:
            rows = self._storage._conn.execute(
                "SELECT m.id FROM memories m "
                "JOIN memories_fts fts ON m.id = fts.rowid "
                "WHERE memories_fts MATCH ? AND m.heat > 0",
                (f'"{entity_name}"',),
            ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            # Fallback to LIKE search if FTS5 match fails
            rows = self._storage._conn.execute(
                "SELECT id FROM memories WHERE content LIKE ? AND heat > 0",
                (f"%{entity_name}%",),
            ).fetchall()
            return [r[0] for r in rows]

    def _find_entities_in_content(self, content: str) -> set[int]:
        """Find entity IDs that appear in the given content."""
        entity_ids: set[int] = set()
        # Get all active entities and check which ones appear in the content
        entities = self._storage.get_all_entities(min_heat=0.0, include_archived=True)
        for entity in entities:
            if entity["name"] in content:
                entity_ids.add(entity["id"])
        return entity_ids

    def _get_top_cooccurring_entities(
        self, content: str, limit: int = 5
    ) -> list[str]:
        """Find entities that co-occur with entities mentioned in this content."""
        # Find entities mentioned in the content
        content_entities = self._find_entities_in_content(content)
        if not content_entities:
            return []

        # Count co-occurrence partners
        partner_counts: dict[str, float] = defaultdict(float)
        for eid in content_entities:
            neighbors = self._graph._get_adjacent(eid, None)
            for n in neighbors:
                entity_row = self._storage._conn.execute(
                    "SELECT name FROM entities WHERE id = ?",
                    (n["entity_id"],),
                ).fetchone()
                if entity_row and n["entity_id"] not in content_entities:
                    partner_counts[entity_row[0]] += n["weight"]

        # Sort by weight and return top
        sorted_partners = sorted(
            partner_counts.items(), key=lambda x: x[1], reverse=True
        )
        return [name for name, _ in sorted_partners[:limit]]
