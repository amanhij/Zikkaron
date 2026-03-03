"""Fractal/hierarchical memory organization with multi-scale retrieval."""

import logging
from collections import Counter, defaultdict

import numpy as np

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.storage import StorageEngine

logger = logging.getLogger(__name__)


class FractalMemoryTree:
    """Multi-level hierarchy for memory organization and retrieval.

    Levels:
      0 (Leaf): Individual memories
      1 (Intermediate): Cluster summaries (from embedding similarity)
      2 (Root): Project/directory-level summaries
    """

    def __init__(
        self,
        storage: StorageEngine,
        embeddings: EmbeddingEngine,
        settings: Settings,
    ) -> None:
        self._storage = storage
        self._embeddings = embeddings
        self._settings = settings

    # -- a. Tree Construction --

    def build_tree(self) -> dict:
        """Construct a multi-level hierarchy from existing memories.

        Returns stats about the tree that was built.
        """
        stats = {"level_1_clusters": 0, "level_2_clusters": 0, "memories_assigned": 0}

        all_memories = self._storage.get_all_memories_with_embeddings()
        if not all_memories:
            return stats

        # 1. Group memories by directory_context
        dir_groups: dict[str, list[dict]] = defaultdict(list)
        for mem in all_memories:
            dir_groups[mem["directory_context"]].append(mem)

        # 2. Within each directory, cluster by embedding similarity
        level_1_clusters: list[dict] = []
        for directory, memories in dir_groups.items():
            clusters = self._cluster_by_similarity(memories)
            for cluster_mems in clusters:
                cluster_info = self._create_level_1_cluster(
                    directory, cluster_mems
                )
                if cluster_info:
                    level_1_clusters.append(cluster_info)
                    stats["level_1_clusters"] += 1
                    stats["memories_assigned"] += len(cluster_mems)

        # 3. Group level 1 clusters by directory prefix into level 2 clusters
        stats["level_2_clusters"] = self._create_level_2_clusters(level_1_clusters)

        return stats

    def _cluster_by_similarity(
        self, memories: list[dict]
    ) -> list[list[dict]]:
        """Agglomerative clustering of memories by embedding similarity.

        Uses a simple single-linkage approach with CLUSTER_SIMILARITY_THRESHOLD.
        """
        if not memories:
            return []

        # Filter to memories with embeddings
        with_emb = [m for m in memories if m.get("embedding") is not None]
        if not with_emb:
            return [memories] if memories else []

        # Parse embeddings into numpy arrays
        emb_arrays = []
        for m in with_emb:
            arr = np.frombuffer(m["embedding"], dtype=np.float32).copy()
            emb_arrays.append(arr)

        threshold = self._settings.CLUSTER_SIMILARITY_THRESHOLD
        n = len(with_emb)

        # Union-Find for clustering
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Compare all pairs (fine for typical memory counts per directory)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_similarity(emb_arrays[i], emb_arrays[j])
                if sim >= threshold:
                    union(i, j)

        # Group by cluster root
        groups: dict[int, list[dict]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(with_emb[i])

        # Add memories without embeddings to the first cluster
        without_emb = [m for m in memories if m.get("embedding") is None]
        if without_emb and groups:
            first_key = next(iter(groups))
            groups[first_key].extend(without_emb)
        elif without_emb:
            groups[0] = without_emb

        return list(groups.values())

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    def _create_level_1_cluster(
        self, directory: str, memories: list[dict]
    ) -> dict | None:
        """Create or update a level 1 cluster from a group of memories."""
        if not memories:
            return None

        # Summary: top 3 entity names + most common tags
        all_tags: list[str] = []
        all_content = ""
        for m in memories:
            tags = m.get("tags", [])
            if isinstance(tags, list):
                all_tags.extend(tags)
            all_content += " " + m.get("content", "")

        # Extract entity-like names (CamelCase, paths, etc.)
        import re
        entity_re = re.compile(r"\b[A-Z][\w]*(?:[A-Z][\w]*)*\b")
        entities = entity_re.findall(all_content)
        entity_counts = Counter(entities)
        top_entities = [e for e, _ in entity_counts.most_common(3)]

        tag_counts = Counter(all_tags)
        top_tags = [t for t, _ in tag_counts.most_common(3)]

        summary_parts = []
        if top_entities:
            summary_parts.append(", ".join(top_entities))
        if top_tags:
            summary_parts.append("tags: " + ", ".join(top_tags))
        summary = "; ".join(summary_parts) if summary_parts else f"cluster in {directory}"

        # Compute centroid embedding
        emb_arrays = []
        for m in memories:
            if m.get("embedding") is not None:
                arr = np.frombuffer(m["embedding"], dtype=np.float32).copy()
                emb_arrays.append(arr)

        centroid = None
        if emb_arrays:
            centroid_arr = np.mean(emb_arrays, axis=0).astype(np.float32)
            norm = np.linalg.norm(centroid_arr)
            if norm > 0:
                centroid_arr = centroid_arr / norm
            centroid = centroid_arr.tobytes()

        cluster_name = f"cluster_{directory.replace('/', '_').strip('_')}_{len(memories)}mem"

        cluster_id = self._storage.insert_cluster({
            "name": cluster_name,
            "level": 1,
            "summary": summary,
            "centroid_embedding": centroid,
            "member_count": len(memories),
        })

        # Assign memories to this cluster
        for m in memories:
            self._storage._conn.execute(
                "UPDATE memories SET cluster_id = ? WHERE id = ?",
                (cluster_id, m["id"]),
            )
        self._storage._conn.commit()

        return {
            "cluster_id": cluster_id,
            "directory": directory,
            "member_count": len(memories),
            "summary": summary,
            "centroid": centroid,
        }

    def _create_level_2_clusters(
        self, level_1_clusters: list[dict]
    ) -> int:
        """Group level 1 clusters by directory prefix into level 2 root clusters."""
        if not level_1_clusters:
            return 0

        # Group by directory
        dir_groups: dict[str, list[dict]] = defaultdict(list)
        for cl in level_1_clusters:
            dir_groups[cl["directory"]].append(cl)

        count = 0
        for directory, clusters in dir_groups.items():
            total_members = sum(c["member_count"] for c in clusters)

            # Collect top entities from sub-clusters for the summary
            all_summaries = " ".join(c["summary"] for c in clusters)

            summary = (
                f"{directory}: {len(clusters)} sub-clusters, "
                f"{total_members} memories"
            )

            root_name = f"root_{directory.replace('/', '_').strip('_')}"

            root_id = self._storage.insert_cluster({
                "name": root_name,
                "level": 2,
                "summary": summary,
                "member_count": total_members,
            })

            # Link level 1 clusters to this root
            for cl in clusters:
                self._storage.update_cluster(cl["cluster_id"], {
                    "parent_cluster_id": root_id,
                })

            count += 1

        return count

    # -- b. Tree Retrieval --

    def retrieve_tree(
        self, query: str, target_level: int | None = None
    ) -> list[dict]:
        """Multi-scale retrieval across the fractal hierarchy.

        If target_level is specified, search only that level.
        If None, adaptively choose level weighting based on query length.
        """
        query_embedding = self._embeddings.encode(query)
        if query_embedding is None:
            return []

        query_arr = np.frombuffer(query_embedding, dtype=np.float32).copy()

        if target_level is not None:
            return self._retrieve_at_level(query_arr, target_level)

        # Adaptive level selection based on query length
        word_count = len(query.split())
        results = []

        if word_count < 10:
            # Short queries -> prefer higher levels (broad)
            level_2 = self._retrieve_at_level(query_arr, 2, weight=1.0)
            level_1 = self._retrieve_at_level(query_arr, 1, weight=0.5)
            level_0 = self._retrieve_at_level(query_arr, 0, weight=0.3)
            results = level_2 + level_1 + level_0
        elif word_count > 30:
            # Long queries -> prefer lower levels (specific)
            level_0 = self._retrieve_at_level(query_arr, 0, weight=1.0)
            level_1 = self._retrieve_at_level(query_arr, 1, weight=0.5)
            level_2 = self._retrieve_at_level(query_arr, 2, weight=0.3)
            results = level_0 + level_1 + level_2
        else:
            # Medium queries -> search all levels equally
            level_0 = self._retrieve_at_level(query_arr, 0, weight=1.0)
            level_1 = self._retrieve_at_level(query_arr, 1, weight=1.0)
            level_2 = self._retrieve_at_level(query_arr, 2, weight=1.0)
            results = level_0 + level_1 + level_2

        # Sort by relevance descending
        results.sort(key=lambda r: r.get("score", 0), reverse=True)
        return results

    def _retrieve_at_level(
        self,
        query_arr: np.ndarray,
        level: int,
        weight: float = 1.0,
        limit: int = 10,
    ) -> list[dict]:
        """Retrieve items at a specific level of the hierarchy."""
        if level == 0:
            return self._retrieve_memories(query_arr, weight, limit)
        else:
            return self._retrieve_clusters(query_arr, level, weight, limit)

    def _retrieve_memories(
        self,
        query_arr: np.ndarray,
        weight: float = 1.0,
        limit: int = 10,
    ) -> list[dict]:
        """Retrieve individual memories (level 0) by vector similarity."""
        query_bytes = query_arr.astype(np.float32).tobytes()
        vec_hits = self._storage.search_vectors(
            query_bytes, top_k=limit, min_heat=0.0
        )

        results = []
        for mid, distance in vec_hits:
            mem = self._storage.get_memory(mid)
            if mem is None:
                continue
            similarity = 1.0 / (1.0 + distance)
            mem.pop("embedding", None)
            results.append({
                "level": 0,
                "type": "memory",
                "id": mem["id"],
                "content": mem["content"],
                "score": similarity * weight,
                "directory": mem["directory_context"],
                "tags": mem.get("tags", []),
                "heat": mem["heat"],
                "cluster_id": mem.get("cluster_id"),
            })

        return results

    def _retrieve_clusters(
        self,
        query_arr: np.ndarray,
        level: int,
        weight: float = 1.0,
        limit: int = 10,
    ) -> list[dict]:
        """Retrieve clusters at a given level by centroid similarity."""
        clusters = self._storage.get_clusters_by_level(level)
        if not clusters:
            return []

        scored = []
        for cl in clusters:
            if cl.get("centroid_embedding") is None:
                # For clusters without centroids, do keyword matching on summary
                query_lower = " ".join(query_arr.tobytes().decode("latin-1", errors="ignore").split())
                # Skip clusters without centroid - they can't be similarity-searched
                continue

            cl_arr = np.frombuffer(cl["centroid_embedding"], dtype=np.float32).copy()
            sim = self._cosine_similarity(query_arr, cl_arr)

            scored.append({
                "level": level,
                "type": "cluster",
                "id": cl["id"],
                "name": cl["name"],
                "summary": cl["summary"],
                "score": sim * weight,
                "member_count": cl["member_count"],
                "parent_cluster_id": cl.get("parent_cluster_id"),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    # -- c. Drill-Down --

    def drill_down(self, cluster_id: int) -> list[dict]:
        """Given a cluster, return all its children.

        Level 2 -> returns level 1 sub-clusters
        Level 1 -> returns individual memories (level 0)
        Level 0 -> returns the memory itself
        """
        cluster = self._storage.get_cluster(cluster_id)
        if cluster is None:
            return []

        level = cluster["level"]

        if level == 2:
            # Return level 1 sub-clusters that have this as parent
            rows = self._storage._conn.execute(
                "SELECT * FROM memory_clusters WHERE parent_cluster_id = ? "
                "ORDER BY heat DESC",
                (cluster_id,),
            ).fetchall()
            return [
                {
                    "level": 1,
                    "type": "cluster",
                    "id": dict(r)["id"],
                    "name": dict(r)["name"],
                    "summary": dict(r)["summary"],
                    "member_count": dict(r)["member_count"],
                }
                for r in rows
            ]

        elif level == 1:
            # Return individual memories assigned to this cluster
            rows = self._storage._conn.execute(
                "SELECT * FROM memories WHERE cluster_id = ? AND heat > 0 "
                "ORDER BY heat DESC",
                (cluster_id,),
            ).fetchall()
            results = []
            for r in rows:
                mem = self._storage._row_to_dict(r)
                mem.pop("embedding", None)
                results.append({
                    "level": 0,
                    "type": "memory",
                    "id": mem["id"],
                    "content": mem["content"],
                    "directory": mem["directory_context"],
                    "tags": mem.get("tags", []),
                    "heat": mem["heat"],
                })
            return results

        else:
            # Level 0 doesn't exist as a cluster, but if somehow called
            return []

    # -- d. Roll-Up --

    def roll_up(self, memory_id: int) -> dict:
        """Given a memory, return its cluster hierarchy.

        Returns: {memory, level_1_cluster, level_2_cluster}
        """
        mem = self._storage.get_memory(memory_id)
        if mem is None:
            return {"memory": None, "level_1_cluster": None, "level_2_cluster": None}

        mem.pop("embedding", None)
        result: dict = {
            "memory": {
                "id": mem["id"],
                "content": mem["content"],
                "directory": mem["directory_context"],
                "tags": mem.get("tags", []),
                "heat": mem["heat"],
            },
            "level_1_cluster": None,
            "level_2_cluster": None,
        }

        cluster_id = mem.get("cluster_id")
        if cluster_id is None:
            return result

        # Get level 1 cluster
        cluster = self._storage.get_cluster(cluster_id)
        if cluster is None:
            return result

        result["level_1_cluster"] = {
            "id": cluster["id"],
            "name": cluster["name"],
            "summary": cluster["summary"],
            "member_count": cluster["member_count"],
            "level": cluster["level"],
        }

        # Get level 2 parent cluster
        parent_id = cluster.get("parent_cluster_id")
        if parent_id is not None:
            parent = self._storage.get_cluster(parent_id)
            if parent is not None:
                result["level_2_cluster"] = {
                    "id": parent["id"],
                    "name": parent["name"],
                    "summary": parent["summary"],
                    "member_count": parent["member_count"],
                    "level": parent["level"],
                }

        return result

    # -- Fractal signal for retrieval integration --

    def fractal_score(
        self, query: str, max_results: int = 10
    ) -> list[tuple[int, float]]:
        """Return (memory_id, score) pairs from fractal cluster matching.

        Used as a signal in unified recall. For broad queries, includes
        cluster-level matches that boost member memories.
        """
        query_embedding = self._embeddings.encode(query)
        if query_embedding is None:
            return []

        query_arr = np.frombuffer(query_embedding, dtype=np.float32).copy()

        # Search level 1 clusters by centroid similarity
        clusters = self._storage.get_clusters_by_level(1)
        if not clusters:
            return []

        memory_scores: dict[int, float] = {}

        for cl in clusters:
            if cl.get("centroid_embedding") is None:
                continue

            cl_arr = np.frombuffer(cl["centroid_embedding"], dtype=np.float32).copy()
            sim = self._cosine_similarity(query_arr, cl_arr)

            if sim <= 0:
                continue

            # Boost all member memories by cluster similarity score
            rows = self._storage._conn.execute(
                "SELECT id FROM memories WHERE cluster_id = ? AND heat > 0",
                (cl["id"],),
            ).fetchall()

            for row in rows:
                mid = row[0]
                memory_scores[mid] = max(memory_scores.get(mid, 0.0), sim)

        ranked = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:max_results]
