"""Formal causal discovery using the PC algorithm for Zikkaron.

Implements a simplified PC algorithm (Spirtes, Glymour, Scheines 2000)
to discover causal DAGs from coding session event logs. Uses numpy and
scipy.stats for conditional independence testing — no external causal
inference libraries required.
"""

import logging
import math
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from itertools import combinations

import numpy as np
from scipy import stats

from zikkaron.config import Settings
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.storage import StorageEngine

logger = logging.getLogger(__name__)


class CausalDiscovery:
    """Discovers causal structure from observational coding event data.

    Collects structured events (file changes, errors, decisions) from
    the knowledge graph and entity store, builds a time-aligned binary
    event matrix, then runs the PC algorithm to discover which variables
    causally influence which others.
    """

    def __init__(
        self,
        storage: StorageEngine,
        knowledge_graph: KnowledgeGraph,
        settings: Settings,
    ) -> None:
        self._storage = storage
        self._kg = knowledge_graph
        self._settings = settings

    def build_event_matrix(
        self, directory: str | None = None, hours: int = 168
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Build a time-aligned binary event matrix from recent activity.

        Rows = 1-hour time windows, Columns = entity variables.
        Values = 1 if entity was active in that window, 0 otherwise.

        Returns (data_matrix, variable_names, timestamps).
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=hours)
        cutoff_iso = cutoff.isoformat()

        # Collect entities active since cutoff
        all_entities = self._storage.get_all_entities(
            min_heat=0.0, include_archived=True
        )

        # Collect episodes within the time range
        episodes = self._storage._conn.execute(
            "SELECT id, timestamp, directory, raw_content FROM episodes "
            "WHERE timestamp >= ? ORDER BY timestamp ASC",
            (cutoff_iso,),
        ).fetchall()
        episodes = self._storage._rows_to_dicts(episodes)

        if directory:
            episodes = [e for e in episodes if e["directory"] == directory]

        if not episodes:
            return np.zeros((0, 0)), [], []

        # Build time buckets (1-hour windows)
        timestamps: list[str] = []
        bucket_start = cutoff.replace(minute=0, second=0, microsecond=0)
        while bucket_start < now:
            timestamps.append(bucket_start.isoformat())
            bucket_start += timedelta(hours=1)

        if not timestamps:
            return np.zeros((0, 0)), [], []

        # Map entity names to column indices
        entity_names: list[str] = []
        entity_name_set: set[str] = set()

        # Collect entity mentions per episode
        episode_entities: list[tuple[str, list[str]]] = []
        for ep in episodes:
            content = ep["raw_content"]
            ep_entities: list[str] = []
            for ent in all_entities:
                name = ent["name"]
                if name in content:
                    ep_entities.append(name)
                    if name not in entity_name_set:
                        entity_name_set.add(name)
                        entity_names.append(name)
            episode_entities.append((ep["timestamp"], ep_entities))

        if not entity_names:
            return np.zeros((0, 0)), [], []

        # Build the matrix
        n_windows = len(timestamps)
        n_vars = len(entity_names)
        name_to_col = {name: i for i, name in enumerate(entity_names)}
        data = np.zeros((n_windows, n_vars), dtype=np.float64)

        for ep_ts, ep_ents in episode_entities:
            # Find which time bucket this episode falls into
            try:
                ep_time = datetime.fromisoformat(ep_ts)
            except (ValueError, TypeError):
                continue
            bucket_idx = int(
                (ep_time - cutoff.replace(minute=0, second=0, microsecond=0))
                .total_seconds() / 3600
            )
            if 0 <= bucket_idx < n_windows:
                for name in ep_ents:
                    if name in name_to_col:
                        data[bucket_idx, name_to_col[name]] = 1.0

        return data, entity_names, timestamps

    def conditional_independence_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray | None = None,
        alpha: float = 0.05,
    ) -> bool:
        """Test if X is independent of Y given Z.

        Returns True if independent (p_value > alpha), False if dependent.
        """
        n = len(x)
        if n < 4:
            return True  # Not enough data to determine dependence

        if z is None:
            # Unconditional: Pearson correlation test
            r = np.corrcoef(x, y)[0, 1]
            if np.isnan(r):
                return True  # constant variable -> treat as independent
            denom = 1.0 - r * r
            if denom <= 0:
                return False  # perfect correlation -> dependent
            t_stat = r * math.sqrt((n - 2) / denom)
            p_value = 2.0 * stats.t.sf(abs(t_stat), df=n - 2)
        else:
            # Partial correlation: regress X on Z and Y on Z, correlate residuals
            if z.ndim == 1:
                z = z.reshape(-1, 1)

            # Add intercept column
            ones = np.ones((n, 1))
            Z = np.hstack([ones, z])

            # Compute residuals via least squares
            try:
                res_x = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
                res_y = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                return True  # Singular matrix -> treat as independent

            # Check for zero-variance residuals
            if np.std(res_x) < 1e-10 or np.std(res_y) < 1e-10:
                return True

            r = np.corrcoef(res_x, res_y)[0, 1]
            if np.isnan(r):
                return True

            dof = n - 2 - z.shape[1]
            if dof < 1:
                return True  # Not enough degrees of freedom

            denom = 1.0 - r * r
            if denom <= 0:
                return False

            t_stat = r * math.sqrt(dof / denom)
            p_value = 2.0 * stats.t.sf(abs(t_stat), df=dof)

        return bool(p_value > alpha)

    def pc_algorithm(
        self,
        data: np.ndarray,
        variable_names: list[str],
        alpha: float = 0.05,
        max_cond_set: int = 3,
    ) -> dict:
        """Run the PC algorithm to discover causal structure.

        Phase 1: Skeleton discovery — remove edges where conditional
        independence is detected.
        Phase 2: Edge orientation — orient v-structures and apply
        Meek's rules.

        Returns dict with nodes, directed_edges, undirected_edges,
        and separating_sets.
        """
        n_vars = data.shape[1]
        if n_vars < 2:
            return {
                "nodes": variable_names,
                "directed_edges": [],
                "undirected_edges": [],
                "separating_sets": {},
            }

        # Phase 1: Skeleton discovery
        # Start with complete undirected graph
        adjacency = [[True] * n_vars for _ in range(n_vars)]
        for i in range(n_vars):
            adjacency[i][i] = False

        sep_sets: dict[tuple[int, int], set[int]] = {}

        for k in range(max_cond_set + 1):
            edges_to_remove = []
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if not adjacency[i][j]:
                        continue

                    # Get neighbors of i (excluding j)
                    neighbors_i = [
                        n for n in range(n_vars)
                        if adjacency[i][n] and n != j
                    ]

                    if len(neighbors_i) < k:
                        continue

                    # Test all subsets of size k from neighbors of i
                    found_independent = False
                    for subset in combinations(neighbors_i, k):
                        if k == 0:
                            z = None
                        else:
                            z = data[:, list(subset)]

                        if self.conditional_independence_test(
                            data[:, i], data[:, j], z, alpha
                        ):
                            edges_to_remove.append((i, j))
                            sep_sets[(i, j)] = set(subset)
                            sep_sets[(j, i)] = set(subset)
                            found_independent = True
                            break

                    if found_independent:
                        continue

                    # Also check from j's perspective
                    neighbors_j = [
                        n for n in range(n_vars)
                        if adjacency[j][n] and n != i
                    ]

                    if len(neighbors_j) < k:
                        continue

                    for subset in combinations(neighbors_j, k):
                        if k == 0:
                            z = None
                        else:
                            z = data[:, list(subset)]

                        if self.conditional_independence_test(
                            data[:, i], data[:, j], z, alpha
                        ):
                            edges_to_remove.append((i, j))
                            sep_sets[(i, j)] = set(subset)
                            sep_sets[(j, i)] = set(subset)
                            found_independent = True
                            break

            for i, j in edges_to_remove:
                adjacency[i][j] = False
                adjacency[j][i] = False

        # Phase 2: Edge orientation
        # directed[i][j] = True means i -> j
        directed = [[False] * n_vars for _ in range(n_vars)]

        # Orient v-structures: X - Z - Y where X not adj Y, Z not in sep(X,Y)
        for z in range(n_vars):
            neighbors_z = [n for n in range(n_vars) if adjacency[z][n]]
            for xi, yi in combinations(neighbors_z, 2):
                x, y = (xi, yi) if xi < yi else (yi, xi)
                if adjacency[x][y]:
                    continue  # x and y are adjacent, skip

                sep = sep_sets.get((x, y), set())
                if z not in sep:
                    # Orient as x -> z <- y (v-structure)
                    directed[x][z] = True
                    directed[y][z] = True
                    # Remove reverse directions
                    directed[z][x] = False
                    directed[z][y] = False

        # Apply Meek's orientation rules iteratively
        changed = True
        while changed:
            changed = False

            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j or not adjacency[i][j]:
                        continue
                    if directed[i][j] or directed[j][i]:
                        continue  # already oriented

                    # R1: If X -> Y and Y - Z and not X - Z, orient Y -> Z
                    for x in range(n_vars):
                        if directed[x][i] and not adjacency[x][j]:
                            directed[i][j] = True
                            changed = True
                            break

                    if directed[i][j]:
                        continue

                    # R2: If X -> Z and Y -> Z and X - Y, orient X -> Y
                    # (i - j): if there exists z such that i -> z and j -> z
                    for z in range(n_vars):
                        if directed[i][z] and directed[j][z]:
                            directed[i][j] = True
                            changed = True
                            break

                    if directed[i][j]:
                        continue

                    # R3: If X - Y and X - Z1 and X - Z2 and Z1 -> Y and Z2 -> Y
                    # orient X -> Y
                    z_to_y = [
                        z for z in range(n_vars)
                        if z != i and z != j
                        and adjacency[i][z]
                        and directed[z][j]
                    ]
                    if len(z_to_y) >= 2:
                        directed[i][j] = True
                        changed = True

        # Build result
        directed_edges: list[tuple[str, str, float]] = []
        undirected_edges: list[tuple[str, str, float]] = []
        seen_undirected: set[tuple[int, int]] = set()

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j or not adjacency[i][j]:
                    continue
                if directed[i][j]:
                    # Compute edge confidence from correlation strength
                    r = abs(np.corrcoef(data[:, i], data[:, j])[0, 1])
                    conf = float(r) if not np.isnan(r) else 0.5
                    directed_edges.append(
                        (variable_names[i], variable_names[j], round(conf, 4))
                    )
                elif not directed[j][i]:
                    edge_key = (min(i, j), max(i, j))
                    if edge_key not in seen_undirected:
                        seen_undirected.add(edge_key)
                        r = abs(np.corrcoef(data[:, i], data[:, j])[0, 1])
                        conf = float(r) if not np.isnan(r) else 0.5
                        undirected_edges.append(
                            (variable_names[i], variable_names[j], round(conf, 4))
                        )

        # Convert separating sets to serializable form
        serializable_sep_sets: dict[str, list[str]] = {}
        for (i, j), s in sep_sets.items():
            if i < j:
                key = f"{variable_names[i]}|{variable_names[j]}"
                serializable_sep_sets[key] = [variable_names[k] for k in s]

        return {
            "nodes": variable_names,
            "directed_edges": directed_edges,
            "undirected_edges": undirected_edges,
            "separating_sets": serializable_sep_sets,
        }

    def discover_dag(
        self,
        directory: str | None = None,
        algorithm: str = "pc",
        hours: int = 168,
    ) -> dict:
        """Build event matrix and run causal discovery.

        Returns the discovered DAG with metadata, and stores directed
        edges in the causal_dag_edges table.
        """
        data, variable_names, timestamps = self.build_event_matrix(
            directory=directory, hours=hours
        )

        n_vars = len(variable_names)
        n_windows = len(timestamps)

        # Minimum data requirements
        if n_vars < 5 or n_windows < 10:
            return {
                "nodes": variable_names,
                "directed_edges": [],
                "undirected_edges": [],
                "separating_sets": {},
                "metadata": {
                    "algorithm": algorithm,
                    "variables": n_vars,
                    "time_windows": n_windows,
                    "status": "insufficient_data",
                    "reason": f"Need >= 5 variables and >= 10 time windows "
                              f"(got {n_vars} vars, {n_windows} windows)",
                },
            }

        dag = self.pc_algorithm(data, variable_names)

        # Store directed edges in causal_dag_edges table
        now_iso = datetime.now(timezone.utc).isoformat()
        stored_count = 0
        for source_name, target_name, confidence in dag["directed_edges"]:
            source_entity = self._storage.get_entity_by_name(source_name)
            target_entity = self._storage.get_entity_by_name(target_name)
            if source_entity and target_entity:
                self._storage.insert_causal_edge({
                    "source_entity_id": source_entity["id"],
                    "target_entity_id": target_entity["id"],
                    "algorithm": algorithm,
                    "confidence": confidence,
                    "discovered_at": now_iso,
                })
                stored_count += 1

        dag["metadata"] = {
            "algorithm": algorithm,
            "variables": n_vars,
            "time_windows": n_windows,
            "directed_count": len(dag["directed_edges"]),
            "undirected_count": len(dag["undirected_edges"]),
            "stored_edges": stored_count,
            "status": "completed",
        }

        return dag

    def query_causes(
        self, effect_entity: str, max_depth: int = 3
    ) -> list[dict]:
        """Find causes of an effect by traversing the DAG upstream.

        BFS from the effect node following edges in reverse direction
        (target -> source means source is a cause of target).
        """
        target = self._storage.get_entity_by_name(effect_entity)
        if not target:
            return []

        visited: set[int] = {target["id"]}
        results: list[dict] = []
        # Queue: (entity_id, entity_name, depth, path)
        queue: deque[tuple[int, str, int, list[str]]] = deque()
        queue.append((target["id"], effect_entity, 0, [effect_entity]))

        while queue:
            current_id, current_name, depth, path = queue.popleft()
            if depth >= max_depth:
                continue

            # Find edges where current entity is the TARGET (upstream causes)
            edges = self._storage._conn.execute(
                "SELECT cde.*, e.name AS source_name "
                "FROM causal_dag_edges cde "
                "JOIN entities e ON e.id = cde.source_entity_id "
                "WHERE cde.target_entity_id = ?",
                (current_id,),
            ).fetchall()
            edges = self._storage._rows_to_dicts(edges)

            for edge in edges:
                src_id = edge["source_entity_id"]
                if src_id not in visited:
                    visited.add(src_id)
                    src_name = edge["source_name"]
                    new_path = path + [src_name]
                    results.append({
                        "entity": src_name,
                        "confidence": edge["confidence"],
                        "depth": depth + 1,
                        "path": new_path,
                    })
                    queue.append((src_id, src_name, depth + 1, new_path))

        # Sort by depth (closest causes first), then by confidence
        results.sort(key=lambda r: (r["depth"], -r["confidence"]))
        return results

    def query_effects(
        self, cause_entity: str, max_depth: int = 3
    ) -> list[dict]:
        """Find effects by traversing the DAG downstream.

        BFS from the cause node following edges in forward direction
        (source -> target means target is an effect of source).
        """
        source = self._storage.get_entity_by_name(cause_entity)
        if not source:
            return []

        visited: set[int] = {source["id"]}
        results: list[dict] = []
        queue: deque[tuple[int, str, int, list[str]]] = deque()
        queue.append((source["id"], cause_entity, 0, [cause_entity]))

        while queue:
            current_id, current_name, depth, path = queue.popleft()
            if depth >= max_depth:
                continue

            # Find edges where current entity is the SOURCE (downstream effects)
            edges = self._storage._conn.execute(
                "SELECT cde.*, e.name AS target_name "
                "FROM causal_dag_edges cde "
                "JOIN entities e ON e.id = cde.target_entity_id "
                "WHERE cde.source_entity_id = ?",
                (current_id,),
            ).fetchall()
            edges = self._storage._rows_to_dicts(edges)

            for edge in edges:
                tgt_id = edge["target_entity_id"]
                if tgt_id not in visited:
                    visited.add(tgt_id)
                    tgt_name = edge["target_name"]
                    new_path = path + [tgt_name]
                    results.append({
                        "entity": tgt_name,
                        "confidence": edge["confidence"],
                        "depth": depth + 1,
                        "path": new_path,
                    })
                    queue.append((tgt_id, tgt_name, depth + 1, new_path))

        results.sort(key=lambda r: (r["depth"], -r["confidence"]))
        return results

    def get_causal_chain(self, entity: str) -> dict:
        """Return both causes and effects for an entity."""
        causes = self.query_causes(entity)
        effects = self.query_effects(entity)

        all_edges = self._storage.get_all_causal_edges()

        return {
            "entity": entity,
            "causes": causes,
            "effects": effects,
            "dag_edges_total": len(all_edges),
        }
