"""Pool of specialized astrocyte processes for domain-aware memory consolidation."""

import json
import logging
import re
from datetime import datetime, timezone

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.storage import StorageEngine
from zikkaron.thermodynamics import MemoryThermodynamics

logger = logging.getLogger(__name__)

# Domain definitions: name -> (entity_types, keywords_regex, decay_multiplier)
# decay_multiplier < 1.0 = faster decay, > 1.0 = slower decay
DOMAIN_DEFINITIONS = {
    "code-patterns": {
        "entity_types": frozenset({"file", "function", "variable"}),
        "keywords": re.compile(
            r"\b(def|class|function|import|require|module|component|"
            r"refactor|implement|method|interface|struct)\b",
            re.IGNORECASE,
        ),
        "decay_multiplier": 1.0,
    },
    "decisions": {
        "entity_types": frozenset({"decision"}),
        "keywords": re.compile(
            r"\b(chose|decided|choosing|switched|migrated|opted|selected|"
            r"replaced|preferred|picked|strategy|approach|trade-?off)\b",
            re.IGNORECASE,
        ),
        "decay_multiplier": 1.5,  # decisions decay slower
    },
    "errors": {
        "entity_types": frozenset({"error", "solution"}),
        "keywords": re.compile(
            r"\b(error|exception|traceback|bug|crash|fix|resolved|"
            r"broken|failed|failure|timeout|denied|workaround)\b",
            re.IGNORECASE,
        ),
        "decay_multiplier": 0.7,  # errors decay faster
    },
    "dependencies": {
        "entity_types": frozenset({"dependency"}),
        "keywords": re.compile(
            r"\b(import|require|install|pip|npm|yarn|package|dependency|"
            r"version|upgrade|downgrade|library|framework)\b",
            re.IGNORECASE,
        ),
        "decay_multiplier": 1.2,  # dependencies decay slightly slower
    },
}


class AstrocytePool:
    """Pool of specialized astrocyte processes for domain-aware consolidation.

    Each process specializes in a memory domain (code, decisions, errors,
    dependencies) and maintains its own memory assignments, decay rates,
    and entity subgraphs.
    """

    def __init__(
        self,
        storage: StorageEngine,
        embeddings: EmbeddingEngine,
        knowledge_graph: KnowledgeGraph,
        thermodynamics: MemoryThermodynamics,
        settings: Settings,
    ) -> None:
        self._storage = storage
        self._embeddings = embeddings
        self._graph = knowledge_graph
        self._thermo = thermodynamics
        self._settings = settings
        self._processes: dict[str, dict] = {}

    # -- a. Process Initialization --

    def init_processes(self) -> None:
        """Create specialized processes, reusing existing DB records if present."""
        existing = self._storage.get_astrocyte_processes()
        existing_by_name = {p["name"]: p for p in existing}

        for domain_name, domain_def in DOMAIN_DEFINITIONS.items():
            if domain_name in existing_by_name:
                proc = existing_by_name[domain_name]
            else:
                proc_id = self._storage.insert_astrocyte_process({
                    "name": domain_name,
                    "domain": domain_name,
                    "specialization": json.dumps({
                        "entity_types": sorted(domain_def["entity_types"]),
                        "decay_multiplier": domain_def["decay_multiplier"],
                    }),
                    "memory_ids": [],
                    "entity_ids": [],
                })
                proc = self._storage._conn.execute(
                    "SELECT * FROM astrocyte_processes WHERE id = ?", (proc_id,)
                ).fetchone()
                proc = self._storage._row_to_dict(proc)

            self._processes[domain_name] = proc

    # -- b. Domain Assignment --

    def assign_memory(self, memory: dict) -> list[str]:
        """Assign a memory to relevant domain processes based on content analysis.

        Returns list of process names the memory was assigned to.
        """
        content = memory.get("content", "")
        tags = memory.get("tags", [])
        if isinstance(tags, str):
            tags = json.loads(tags)

        assigned: list[str] = []

        for domain_name, domain_def in DOMAIN_DEFINITIONS.items():
            score = 0.0

            # Check keyword matches in content
            keyword_matches = domain_def["keywords"].findall(content)
            score += len(keyword_matches) * 0.3

            # Check entity types referenced — look at entities linked to this memory
            for entity_type in domain_def["entity_types"]:
                if entity_type in content.lower():
                    score += 0.2

            # Check tags for domain relevance
            tag_text = " ".join(tags).lower()
            tag_matches = domain_def["keywords"].findall(tag_text)
            score += len(tag_matches) * 0.2

            if score >= 0.3:
                assigned.append(domain_name)

        # Default: assign to code-patterns if no match
        if not assigned:
            assigned.append("code-patterns")

        # Update process records with the memory assignment
        memory_id = memory.get("id")
        if memory_id is not None:
            for domain_name in assigned:
                proc = self._processes.get(domain_name)
                if proc is None:
                    continue
                current_ids = proc.get("memory_ids", [])
                if memory_id not in current_ids:
                    current_ids.append(memory_id)
                    self._storage.update_astrocyte_process(
                        proc["id"], {"memory_ids": current_ids}
                    )
                    proc["memory_ids"] = current_ids

        return assigned

    # -- c. Specialized Consolidation --

    def consolidate_domain(self, process_name: str) -> dict:
        """Run consolidation only on memories assigned to a specific domain process.

        Applies domain-specific decay rates and extracts domain-specific entities.
        """
        proc = self._processes.get(process_name)
        if proc is None:
            return {"error": f"unknown process: {process_name}"}

        domain_def = DOMAIN_DEFINITIONS.get(process_name)
        if domain_def is None:
            return {"error": f"no domain definition for: {process_name}"}

        stats = {
            "process": process_name,
            "memories_processed": 0,
            "memories_decayed": 0,
            "entities_extracted": 0,
        }

        now = datetime.now(timezone.utc)
        decay_multiplier = domain_def["decay_multiplier"]
        memory_ids = proc.get("memory_ids", [])

        # Apply domain-specific decay
        for mid in list(memory_ids):
            mem = self._storage.get_memory(mid)
            if mem is None:
                # Memory was deleted; clean up
                memory_ids.remove(mid)
                continue

            stats["memories_processed"] += 1

            last = datetime.fromisoformat(mem["last_accessed"])
            hours = (now - last).total_seconds() / 3600.0

            # Domain-adjusted decay: modify hours by inverse of multiplier
            # Higher multiplier = slower decay = fewer effective hours
            adjusted_hours = hours / decay_multiplier
            new_heat = self._thermo.compute_decay(mem, adjusted_hours)

            if new_heat < self._settings.COLD_THRESHOLD:
                new_heat = 0.0

            if abs(new_heat - mem["heat"]) > 1e-9:
                self._storage.update_memory_heat(mid, new_heat)
                stats["memories_decayed"] += 1

        # Extract domain-specific entities from assigned memories
        entity_types = domain_def["entity_types"]
        entity_ids = proc.get("entity_ids", [])

        for mid in memory_ids:
            mem = self._storage.get_memory(mid)
            if mem is None:
                continue
            content = mem.get("content", "")
            typed_entities = self._graph.extract_entities_typed(
                content, mem.get("directory_context", "")
            )
            for name, etype, _ctx in typed_entities:
                if etype in entity_types:
                    existing = self._storage.get_entity_by_name(name)
                    if existing:
                        eid = existing["id"]
                        self._storage.reinforce_entity(eid)
                    else:
                        eid = self._storage.insert_entity(
                            {"name": name, "type": etype}
                        )
                    if eid not in entity_ids:
                        entity_ids.append(eid)
                        stats["entities_extracted"] += 1

        # Update process record
        self._storage.update_astrocyte_process(proc["id"], {
            "memory_ids": memory_ids,
            "entity_ids": entity_ids,
        })
        proc["memory_ids"] = memory_ids
        proc["entity_ids"] = entity_ids

        return stats

    # -- d. Consensus Retrieval --

    def consensus_retrieve(
        self, query: str, top_k: int = 5
    ) -> list[dict]:
        """Each process scores the query against its domain memories, then merge with voting.

        Memories returned by multiple processes get score boosts.
        Domain-relevant queries weight that domain's results higher.
        """
        # Determine which domains are most relevant to this query
        domain_relevance: dict[str, float] = {}
        for domain_name, domain_def in DOMAIN_DEFINITIONS.items():
            matches = domain_def["keywords"].findall(query)
            domain_relevance[domain_name] = 1.0 + len(matches) * 0.5

        query_embedding = self._embeddings.encode(query)

        # Each process scores independently
        votes: dict[int, list[tuple[float, str]]] = {}  # memory_id -> [(score, domain)]

        for domain_name, proc in self._processes.items():
            memory_ids = proc.get("memory_ids", [])
            if not memory_ids:
                continue

            domain_weight = domain_relevance.get(domain_name, 1.0)

            for mid in memory_ids:
                mem = self._storage.get_memory(mid)
                if mem is None or mem["heat"] <= 0:
                    continue

                score = mem["heat"]

                # Semantic similarity if embeddings available
                if query_embedding is not None and mem.get("embedding"):
                    sim = self._embeddings.similarity(
                        query_embedding, mem["embedding"]
                    )
                    score = sim * 0.6 + mem["heat"] * 0.4

                # Weight by domain relevance
                weighted_score = score * domain_weight

                if mid not in votes:
                    votes[mid] = []
                votes[mid].append((weighted_score, domain_name))

        # Merge votes: boost memories voted by multiple processes
        results: list[tuple[int, float]] = []
        for mid, vote_list in votes.items():
            base_score = max(s for s, _ in vote_list)
            # Multi-domain boost: each additional vote adds 15% of base
            multi_domain_boost = (len(vote_list) - 1) * 0.15 * base_score
            final_score = base_score + multi_domain_boost
            results.append((mid, final_score))

        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        output = []
        for mid, score in results:
            mem = self._storage.get_memory(mid)
            if mem is None:
                continue
            mem.pop("embedding", None)
            mem["consensus_score"] = round(score, 4)
            mem["voting_domains"] = [
                d for _, d in votes.get(mid, [])
            ]
            output.append(mem)

        return output

    # -- e. Process Health --

    def get_process_stats(self) -> list[dict]:
        """Return stats for each astrocyte process."""
        stats = []
        for domain_name, proc in self._processes.items():
            memory_ids = proc.get("memory_ids", [])

            # Compute avg heat of assigned memories
            heats = []
            for mid in memory_ids:
                mem = self._storage.get_memory(mid)
                if mem is not None:
                    heats.append(mem["heat"])

            avg_heat = sum(heats) / len(heats) if heats else 0.0

            stats.append({
                "name": domain_name,
                "domain": proc.get("domain", domain_name),
                "memory_count": len(memory_ids),
                "entity_count": len(proc.get("entity_ids", [])),
                "avg_heat": round(avg_heat, 4),
                "last_active": proc.get("last_active", ""),
                "decay_multiplier": DOMAIN_DEFINITIONS.get(
                    domain_name, {}
                ).get("decay_multiplier", 1.0),
            })

        return stats
