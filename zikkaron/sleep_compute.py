"""Sleep-time compute system — offline processing for creative connections and maintenance."""

import logging
import random
import re
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np

from zikkaron.config import Settings
from zikkaron.curation import MemoryCurator
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.narrative import NarrativeEngine
from zikkaron.storage import StorageEngine
from zikkaron.thermodynamics import MemoryThermodynamics

logger = logging.getLogger(__name__)

# Sentence boundary splitter
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Entity-like patterns for identifying key sentences during compression
_ENTITY_PATTERN_RE = re.compile(
    r"(?:[\w@.-]+/[\w@.-]+\.\w+"  # file paths
    r"|\bdef\s+\w+"  # python defs
    r"|\bclass\s+\w+"  # python classes
    r"|\b\w*(?:Error|Exception)\b"  # error types
    r"|\bimport\s+\w+"  # imports
    r"|\bfrom\s+\w+)",  # from imports
)


class SleepComputeEngine:
    """Offline sleep-time processing engine.

    Runs during extended idle periods to:
    - Dream replay: discover unexpected cross-domain connections
    - Community detection: find clusters of related entities
    - Cluster summarization: generate summaries for memory groups
    - Re-embedding: update stale embeddings to current model
    - Compression: compress old verbose memories
    """

    def __init__(
        self,
        storage: StorageEngine,
        embeddings: EmbeddingEngine,
        knowledge_graph: KnowledgeGraph,
        curation: MemoryCurator,
        thermodynamics: MemoryThermodynamics,
        settings: Settings,
    ) -> None:
        self._storage = storage
        self._embeddings = embeddings
        self._graph = knowledge_graph
        self._curator = curation
        self._thermo = thermodynamics
        self._settings = settings
        self._narrative = NarrativeEngine(storage, knowledge_graph, settings)

    # -- a. Dream Replay --

    def dream_replay(self) -> dict:
        """Select random pairs of unconnected memories and discover hidden connections.

        For each pair:
        - similarity > 0.4: create a weak co_occurrence relationship (weight=0.5)
        - similarity > 0.7: also generate a synthetic "dream insight" memory
        """
        stats = {"pairs_examined": 0, "connections_found": 0, "insights_generated": 0}

        memories = self._storage.get_all_memories_with_embeddings()
        if len(memories) < 2:
            return stats

        max_pairs = len(memories) * (len(memories) - 1) // 2
        num_pairs = min(self._settings.DREAM_REPLAY_PAIRS, max_pairs)

        # Generate random unique index pairs
        pairs: set[tuple[int, int]] = set()
        attempts = 0
        max_attempts = num_pairs * 10
        while len(pairs) < num_pairs and attempts < max_attempts:
            i, j = random.sample(range(len(memories)), 2)
            pairs.add((min(i, j), max(i, j)))
            attempts += 1

        for idx_a, idx_b in pairs:
            mem_a = memories[idx_a]
            mem_b = memories[idx_b]

            if mem_a["embedding"] is None or mem_b["embedding"] is None:
                continue

            # Skip already-connected memories
            if self._memories_connected(mem_a["id"], mem_b["id"]):
                continue

            stats["pairs_examined"] += 1
            sim = self._embeddings.similarity(mem_a["embedding"], mem_b["embedding"])

            if sim > 0.7:
                # Strong unexpected connection: link + dream insight
                self._create_dream_connection(mem_a["id"], mem_b["id"])
                self._create_dream_insight(mem_a, mem_b)
                stats["connections_found"] += 1
                stats["insights_generated"] += 1
            elif sim > 0.4:
                # Moderate connection: link only
                self._create_dream_connection(mem_a["id"], mem_b["id"])
                stats["connections_found"] += 1

        return stats

    def _memories_connected(self, mem_a_id: int, mem_b_id: int) -> bool:
        """Check if two memories are already connected via entity relationships."""
        entity_a = self._storage.get_entity_by_name(f"memory:{mem_a_id}")
        entity_b = self._storage.get_entity_by_name(f"memory:{mem_b_id}")
        if entity_a and entity_b:
            rel = self._storage.get_relationship_between(
                entity_a["id"], entity_b["id"]
            )
            return rel is not None
        return False

    def _create_dream_connection(self, mem_a_id: int, mem_b_id: int) -> None:
        """Create a weak co_occurrence link between two memories."""
        src_eid = self._ensure_memory_entity(mem_a_id)
        tgt_eid = self._ensure_memory_entity(mem_b_id)

        self._storage.insert_relationship({
            "source_entity_id": src_eid,
            "target_entity_id": tgt_eid,
            "relationship_type": "co_occurrence",
            "weight": 0.5,
        })

    def _ensure_memory_entity(self, memory_id: int) -> int:
        """Get or create an entity node for a memory."""
        name = f"memory:{memory_id}"
        existing = self._storage.get_entity_by_name(name)
        if existing:
            return existing["id"]
        return self._storage.insert_entity({"name": name, "type": "file"})

    def _create_dream_insight(self, mem_a: dict, mem_b: dict) -> None:
        """Generate a synthetic dream insight memory."""
        summary_a = mem_a["content"][:100].strip()
        summary_b = mem_b["content"][:100].strip()
        content = f"Dream connection: {summary_a} may relate to {summary_b}"

        embedding = self._embeddings.encode(content)
        memory_id = self._storage.insert_memory({
            "content": content,
            "embedding": embedding,
            "tags": ["dream", "auto-generated"],
            "directory_context": "system",
            "heat": 0.5,
            "is_stale": False,
            "embedding_model": self._embeddings.get_model_name(),
        })
        self._storage.update_memory_scores(
            memory_id,
            surprise_score=0.8,
            importance=0.4,
        )

    # -- b. Community Detection --

    def detect_communities(self) -> list[dict]:
        """Build a networkx graph from entity relationships and detect communities."""
        import networkx as nx

        entities = self._storage.get_all_entities(min_heat=0.0, include_archived=False)
        if not entities:
            return []

        entity_map = {e["id"]: e for e in entities}

        G = nx.Graph()
        for e in entities:
            G.add_node(e["id"], name=e["name"], type=e["type"])

        rows = self._storage._conn.execute(
            "SELECT source_entity_id, target_entity_id, weight "
            "FROM relationships"
        ).fetchall()

        for row in rows:
            src, tgt, weight = row[0], row[1], row[2]
            if src in entity_map and tgt in entity_map:
                G.add_edge(src, tgt, weight=weight or 1.0)

        if G.number_of_edges() == 0:
            return []

        # Run Louvain community detection with label propagation fallback
        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(G, seed=42)
        except Exception:
            from networkx.algorithms.community import label_propagation_communities
            communities = list(label_propagation_communities(G))

        results = []
        for comm_idx, community in enumerate(communities):
            if len(community) < 2:
                continue

            # Collect entity names for this community
            member_names = []
            for eid in community:
                if eid in entity_map:
                    member_names.append(entity_map[eid]["name"])

            # Summary from top entity names
            top_names = sorted(member_names)[:5]
            summary = ", ".join(top_names)
            cluster_name = f"community_{comm_idx}"

            # Find memories associated with these entities
            memory_ids = self._find_memories_for_entities(member_names)

            # Create cluster record
            cluster_id = self._storage.insert_cluster({
                "name": cluster_name,
                "level": 1,
                "summary": summary,
                "member_count": len(memory_ids),
            })

            # Assign memories to this cluster
            for mid in memory_ids:
                self._storage._conn.execute(
                    "UPDATE memories SET cluster_id = ? WHERE id = ?",
                    (cluster_id, mid),
                )
            if memory_ids:
                self._storage._conn.commit()

            results.append({
                "cluster_id": cluster_id,
                "name": cluster_name,
                "entity_count": len(community),
                "member_count": len(memory_ids),
            })

        return results

    def _find_memories_for_entities(self, entity_names: list[str]) -> list[int]:
        """Find memory IDs whose content mentions any of the given entity names."""
        if not entity_names:
            return []

        memory_ids: set[int] = set()
        all_memories = self._storage._conn.execute(
            "SELECT id, content FROM memories WHERE heat > 0"
        ).fetchall()

        for mem in all_memories:
            content = mem[1]
            for name in entity_names:
                if name in content:
                    memory_ids.add(mem[0])
                    break

        return list(memory_ids)

    # -- c. Cluster Summarization --

    def generate_cluster_summaries(self) -> None:
        """Generate summaries and centroid embeddings for clusters with > 3 members."""
        clusters = self._storage.get_clusters_by_level(1)

        for cluster in clusters:
            if cluster["member_count"] <= 3:
                continue

            cluster_id = cluster["id"]
            rows = self._storage._conn.execute(
                "SELECT id, content, embedding, heat FROM memories "
                "WHERE cluster_id = ? AND heat > 0",
                (cluster_id,),
            ).fetchall()

            if len(rows) <= 3:
                continue

            # Extract entities and keywords from all member contents
            all_content = " ".join(r[1] for r in rows)
            entities = _ENTITY_PATTERN_RE.findall(all_content)
            entity_counts = Counter(entities)
            top_entities = [e for e, _ in entity_counts.most_common(10)]

            # Top keywords by frequency (excluding stop words)
            words = all_content.lower().split()
            stop_words = frozenset({
                "the", "a", "an", "is", "are", "was", "were", "and", "or",
                "to", "in", "of", "for", "with", "on", "at", "by", "from",
                "this", "that", "it", "not", "be", "as", "has", "have",
            })
            meaningful = [w for w in words if w not in stop_words and len(w) > 2]
            word_counts = Counter(meaningful)
            top_keywords = [w for w, _ in word_counts.most_common(5)]

            summary_parts = []
            if top_entities:
                summary_parts.append("Entities: " + ", ".join(top_entities[:5]))
            if top_keywords:
                summary_parts.append("Keywords: " + ", ".join(top_keywords))
            summary = "; ".join(summary_parts) if summary_parts else cluster["summary"]

            # Compute centroid embedding (average of member embeddings, normalized)
            embeddings_list = [r[2] for r in rows if r[2] is not None]
            centroid = None
            if embeddings_list:
                arrays = [np.frombuffer(e, dtype=np.float32) for e in embeddings_list]
                centroid_arr = np.mean(arrays, axis=0).astype(np.float32)
                norm = np.linalg.norm(centroid_arr)
                if norm > 0:
                    centroid_arr = centroid_arr / norm
                centroid = centroid_arr.tobytes()

            self._storage.update_cluster(cluster_id, {
                "summary": summary,
                "centroid_embedding": centroid,
            })

        # Create level 2 (root) clusters by grouping level 1 by directory_context
        self._create_root_clusters()

    def _create_root_clusters(self) -> None:
        """Group level 1 clusters by dominant directory_context into level 2 clusters."""
        clusters = self._storage.get_clusters_by_level(1)
        if not clusters:
            return

        dir_groups: dict[str, list[dict]] = {}
        for cluster in clusters:
            rows = self._storage._conn.execute(
                "SELECT directory_context, COUNT(*) as cnt FROM memories "
                "WHERE cluster_id = ? GROUP BY directory_context "
                "ORDER BY cnt DESC LIMIT 1",
                (cluster["id"],),
            ).fetchall()

            dominant_dir = rows[0][0] if rows else "unknown"
            dir_groups.setdefault(dominant_dir, []).append(cluster)

        for dir_ctx, group_clusters in dir_groups.items():
            if len(group_clusters) < 2:
                continue

            root_name = f"root_{dir_ctx.replace('/', '_').strip('_')}"
            total_members = sum(c["member_count"] for c in group_clusters)

            root_id = self._storage.insert_cluster({
                "name": root_name,
                "level": 2,
                "summary": f"Root cluster for {dir_ctx}",
                "member_count": total_members,
            })

            for child in group_clusters:
                self._storage.update_cluster(child["id"], {
                    "parent_cluster_id": root_id,
                })

    # -- d. Incremental Re-embedding --

    def reembed_stale(self) -> int:
        """Re-embed memories whose embedding_model differs from the current model."""
        current_model = self._embeddings.get_model_name()
        stale = self._storage.get_memories_needing_reembedding(current_model)

        if not stale:
            return 0

        count = 0
        batch_size = 50

        for i in range(0, len(stale), batch_size):
            batch = stale[i : i + batch_size]
            texts = [m["content"] for m in batch]
            embeddings = self._embeddings.encode_batch(texts)

            for mem, emb in zip(batch, embeddings):
                if emb is not None:
                    self._storage.update_memory_embedding(
                        mem["id"], emb, current_model
                    )
                    count += 1

        return count

    # -- e. Memory Compression --

    def compress_old_memories(self, days_threshold: int = 30) -> int:
        """Compress old verbose memories by extracting key entity-bearing sentences."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days_threshold)
        ).isoformat()

        rows = self._storage._conn.execute(
            "SELECT id, content FROM memories "
            "WHERE created_at < ? AND LENGTH(content) > 1000 "
            "AND (compressed = 0 OR compressed IS NULL)",
            (cutoff,),
        ).fetchall()

        compressed_count = 0
        for row in rows:
            mem_id, content = row[0], row[1]

            # Split into sentences and keep those containing entity patterns
            sentences = _SENTENCE_RE.split(content)
            key_sentences = [s for s in sentences if _ENTITY_PATTERN_RE.search(s)]

            if not key_sentences:
                # Fallback: keep first and last sentences
                key_sentences = [sentences[0]]
                if len(sentences) > 1:
                    key_sentences.append(sentences[-1])

            compressed_content = " ".join(key_sentences)

            # Only compress if actually shorter
            if len(compressed_content) >= len(content):
                continue

            # Update content, set compressed flag, re-embed
            new_embedding = self._embeddings.encode(compressed_content)
            self._storage._conn.execute(
                "UPDATE memories SET content = ?, compressed = 1 WHERE id = ?",
                (compressed_content, mem_id),
            )
            self._storage._conn.commit()

            if new_embedding is not None:
                self._storage.update_memory_embedding(
                    mem_id, new_embedding, self._embeddings.get_model_name()
                )

            compressed_count += 1

        return compressed_count

    # -- f. Full Sleep Cycle --

    def run_sleep_cycle(self) -> dict:
        """Orchestrate all sleep-time operations in order."""
        stats: dict = {}

        logger.info("Sleep cycle phase 1: dream replay")
        stats["dream_replay"] = self.dream_replay()

        logger.info("Sleep cycle phase 2: community detection")
        stats["communities"] = self.detect_communities()

        logger.info("Sleep cycle phase 3: cluster summarization")
        self.generate_cluster_summaries()
        stats["cluster_summaries_generated"] = True

        logger.info("Sleep cycle phase 4: re-embedding")
        stats["reembedded"] = self.reembed_stale()

        logger.info("Sleep cycle phase 5: compression")
        stats["compressed"] = self.compress_old_memories()

        logger.info("Sleep cycle phase 6: auto-narrate")
        stats["narrative"] = self._narrative.auto_narrate()

        logger.info("Sleep cycle complete: %s", stats)
        return stats
