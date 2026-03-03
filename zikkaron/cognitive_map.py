"""Successor Representation cognitive maps — retrieval as navigation through concept space.

Based on:
- Stachenfeld et al. (Nature Neuroscience 20:1643, 2017): SR in hippocampus
- Whittington et al. "Tolman-Eichenbaum Machine" (Cell 183:1249, 2020)
- Yan "External Hippocampus" (arXiv:2512.18190, 2025)

Key math:
  T[i,j] = P(access j right after i)
  M = (I - γ·T)^{-1}  — Successor Representation matrix
  M[i,j] = expected discounted future visits to j starting from i
  Eigenvectors of M ≈ "grid cell coordinates" in concept space
"""

import logging
from typing import Optional

import numpy as np

from zikkaron.config import Settings
from zikkaron.storage import StorageEngine

logger = logging.getLogger(__name__)

# Minimum transitions before SR is considered useful
_MIN_TRANSITIONS = 20


class CognitiveMap:
    """Navigate memory space via Successor Representation coordinates."""

    def __init__(self, storage: StorageEngine, settings: Settings) -> None:
        self._storage = storage
        self._discount = settings.SR_DISCOUNT  # γ
        self._lr = settings.SR_UPDATE_RATE  # TD learning rate
        self._sr_matrix: Optional[np.ndarray] = None
        self._memory_index: dict[int, int] = {}  # memory_id → row index
        self._index_memory: dict[int, int] = {}  # row index → memory_id
        self._dirty = True

    # -- Recording --

    def record_transition(
        self, from_memory_id: int, to_memory_id: int, session_id: str = ""
    ) -> None:
        """Record that memory 'to' was accessed right after memory 'from'."""
        existing = self._storage.get_transition(from_memory_id, to_memory_id)
        if existing:
            self._storage.increment_transition(from_memory_id, to_memory_id)
        else:
            self._storage.insert_transition({
                "from_memory_id": from_memory_id,
                "to_memory_id": to_memory_id,
                "count": 1,
                "session_id": session_id,
            })
        self._dirty = True

    # -- Transition matrix --

    def build_transition_matrix(self) -> np.ndarray:
        """Build row-normalized transition matrix T from stored transitions.

        T[i,j] = count(i→j) / sum(count(i→*))
        """
        transitions = self._storage.get_all_transitions()
        if not transitions:
            self._memory_index = {}
            self._index_memory = {}
            return np.zeros((0, 0), dtype=np.float64)

        # Collect unique memory IDs
        ids: set[int] = set()
        for t in transitions:
            ids.add(t["from_memory_id"])
            ids.add(t["to_memory_id"])

        sorted_ids = sorted(ids)
        self._memory_index = {mid: idx for idx, mid in enumerate(sorted_ids)}
        self._index_memory = {idx: mid for mid, idx in self._memory_index.items()}

        n = len(sorted_ids)
        T = np.zeros((n, n), dtype=np.float64)

        for t in transitions:
            i = self._memory_index[t["from_memory_id"]]
            j = self._memory_index[t["to_memory_id"]]
            T[i, j] = t["count"]

        # Row-normalize
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid division by zero
        T = T / row_sums

        return T

    # -- SR matrix --

    def compute_sr_matrix(self) -> np.ndarray:
        """Compute M = (I - γ·T)^{-1}, the Successor Representation matrix."""
        T = self.build_transition_matrix()
        n = T.shape[0]

        if n == 0:
            self._sr_matrix = np.zeros((0, 0), dtype=np.float64)
            self._dirty = False
            return self._sr_matrix

        if n > 5000:
            # Iterative: M ≈ I + γT + γ²T² + ... (truncate at 20 terms)
            M = np.eye(n, dtype=np.float64)
            power = np.eye(n, dtype=np.float64)
            gamma_k = 1.0
            for _ in range(20):
                gamma_k *= self._discount
                power = power @ T
                M += gamma_k * power
        else:
            # Direct inversion with epsilon for numerical stability
            eps = 1e-10
            A = np.eye(n, dtype=np.float64) - self._discount * T
            A += eps * np.eye(n, dtype=np.float64)
            M = np.linalg.inv(A)

        self._sr_matrix = M
        self._dirty = False
        return M

    # -- Coordinate extraction --

    def extract_coordinates(self, n_dims: int = 2) -> dict[int, tuple]:
        """Extract low-dimensional coordinates from SR matrix eigenvectors.

        Returns {memory_id: (x, y, ...)} mapping.
        """
        if self._sr_matrix is None or self._dirty:
            self.compute_sr_matrix()

        M = self._sr_matrix
        if M.size == 0:
            return {}

        n = M.shape[0]
        n_dims = min(n_dims, n)

        # Symmetrize for real eigenvalues
        M_sym = (M + M.T) / 2.0

        eigenvalues, eigenvectors = np.linalg.eigh(M_sym)

        # Sort by eigenvalue magnitude descending
        order = np.argsort(-np.abs(eigenvalues))
        # Take top n_dims eigenvectors (columns of eigenvectors)
        top_vecs = eigenvectors[:, order[:n_dims]]  # shape (n, n_dims)

        coords: dict[int, tuple] = {}
        for idx in range(n):
            mid = self._index_memory[idx]
            coords[mid] = tuple(float(top_vecs[idx, d]) for d in range(n_dims))

        return coords

    def update_memory_coordinates(self) -> int:
        """Compute SR coordinates and update sr_x, sr_y in storage."""
        coords = self.extract_coordinates(n_dims=2)
        count = 0
        for mid, (x, y) in coords.items():
            self._storage.update_memory_sr_coords(mid, x, y)
            count += 1
        return count

    # -- Navigation --

    def navigate_to(
        self,
        query_embedding: bytes,
        embeddings_engine,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Project query into SR space and find nearest memories.

        Returns (memory_id, proximity_score) sorted by proximity descending.
        """
        coords = self.extract_coordinates(n_dims=2)
        if not coords:
            return []

        # Find top-5 most similar memories by embedding to seed SR position
        all_memory_ids = list(coords.keys())
        vec_hits = self._storage.search_vectors(
            query_embedding, top_k=min(5, len(all_memory_ids)), min_heat=0.0
        )

        if not vec_hits:
            return []

        # Average SR coordinates of embedding-similar memories
        seed_coords = []
        for mid, _dist in vec_hits:
            if mid in coords:
                seed_coords.append(np.array(coords[mid]))

        if not seed_coords:
            return []

        query_pos = np.mean(seed_coords, axis=0)

        # Find top_k nearest in SR space
        distances: list[tuple[int, float]] = []
        for mid, c in coords.items():
            dist = float(np.linalg.norm(np.array(c) - query_pos))
            proximity = 1.0 / (1.0 + dist)
            distances.append((mid, proximity))

        distances.sort(key=lambda x: x[1], reverse=True)
        return distances[:top_k]

    def get_neighborhood(
        self, memory_id: int, radius: float = 0.5
    ) -> list[dict]:
        """Find memories within Euclidean distance 'radius' in SR space."""
        coords = self.extract_coordinates(n_dims=2)
        if memory_id not in coords:
            return []

        center = np.array(coords[memory_id])
        neighbors: list[dict] = []

        for mid, c in coords.items():
            if mid == memory_id:
                continue
            dist = float(np.linalg.norm(np.array(c) - center))
            if dist <= radius:
                mem = self._storage.get_memory(mid)
                if mem:
                    mem.pop("embedding", None)
                    mem["sr_distance"] = round(dist, 6)
                    neighbors.append(mem)

        neighbors.sort(key=lambda x: x["sr_distance"])
        return neighbors

    # -- Incremental TD update --

    def incremental_update(self, from_id: int, to_id: int) -> None:
        """TD-learning update: M[from] += lr * (e_to + γ·M[to] - M[from]).

        Only updates if both IDs are in the current index.
        """
        if self._sr_matrix is None or self._sr_matrix.size == 0:
            return
        if from_id not in self._memory_index or to_id not in self._memory_index:
            return

        i = self._memory_index[from_id]
        j = self._memory_index[to_id]
        n = self._sr_matrix.shape[0]

        e_to = np.zeros(n, dtype=np.float64)
        e_to[j] = 1.0

        delta = e_to + self._discount * self._sr_matrix[j] - self._sr_matrix[i]
        self._sr_matrix[i] += self._lr * delta

    # -- Query helpers --

    def get_sr_scores(
        self, query_embedding: bytes, embeddings_engine, candidate_ids: list[int]
    ) -> dict[int, float]:
        """Get SR proximity scores for candidate memories given a query.

        Used as a retrieval signal in HippoRetriever.recall().
        """
        if self._sr_matrix is None or self._dirty:
            self.compute_sr_matrix()

        if self._sr_matrix is None or self._sr_matrix.size == 0:
            return {}

        coords = self.extract_coordinates(n_dims=2)
        if not coords:
            return {}

        # Find query position via embedding similarity
        vec_hits = self._storage.search_vectors(
            query_embedding, top_k=5, min_heat=0.0
        )

        seed_coords = []
        for mid, _dist in vec_hits:
            if mid in coords:
                seed_coords.append(np.array(coords[mid]))

        if not seed_coords:
            return {}

        query_pos = np.mean(seed_coords, axis=0)

        # Score candidates by proximity
        result: dict[int, float] = {}
        for mid in candidate_ids:
            if mid in coords:
                dist = float(np.linalg.norm(np.array(coords[mid]) - query_pos))
                result[mid] = 1.0 / (1.0 + dist)

        return result

    def has_sufficient_data(self) -> bool:
        """Check if enough transitions exist for meaningful SR computation."""
        transitions = self._storage.get_all_transitions()
        total_count = sum(t["count"] for t in transitions) if transitions else 0
        return total_count >= _MIN_TRANSITIONS

    @property
    def is_dirty(self) -> bool:
        return self._dirty
