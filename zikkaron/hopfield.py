"""Modern Hopfield Networks for energy-based associative memory retrieval.

Implements the continuous Hopfield model from Ramsauer et al. (2021),
"Hopfield Networks is All You Need" (arXiv:2008.02217). Retrieval is
equivalent to transformer attention: softmax(β · Xᵀ · query), where X
is the stored pattern matrix and β controls sharpness.

Key capabilities:
  - Dense retrieval via softmax attention over all stored patterns
  - Sparse retrieval via Hopfield-Fenchel-Young (sparsemax)
  - Pattern completion via iterative Hopfield dynamics
  - Energy-based novelty detection
"""

import logging
from typing import Optional

import numpy as np

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.storage import StorageEngine

logger = logging.getLogger(__name__)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()


def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp using numpy."""
    max_x = np.max(x)
    return float(max_x + np.log(np.sum(np.exp(x - max_x))))


def _sparsemax(logits: np.ndarray) -> np.ndarray:
    """Sparsemax: projects logits onto the probability simplex.

    Produces exact zeros for irrelevant entries, unlike softmax.
    Algorithm from Martins & Astudillo (2016).
    """
    n = len(logits)
    if n == 0:
        return logits

    # Sort in descending order
    sorted_logits = np.sort(logits)[::-1]
    cumsum = np.cumsum(sorted_logits)

    # Find the threshold index k: largest k where sorted[k] > (cumsum[k] - 1) / (k+1)
    k_range = np.arange(1, n + 1, dtype=np.float64)
    thresholds = (cumsum - 1.0) / k_range
    support = sorted_logits > thresholds
    k = int(np.sum(support))
    if k == 0:
        k = 1  # at least one non-zero entry

    tau = (cumsum[k - 1] - 1.0) / k
    weights = np.maximum(logits - tau, 0.0)
    return weights


class HopfieldMemory:
    """Modern Hopfield Network for energy-based associative memory retrieval.

    The stored pattern matrix X (N×d) is cached and rebuilt lazily when
    memories change. Retrieval computes attention = softmax(β · X · query),
    which is mathematically equivalent to single-head transformer attention.
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
        self._beta: float = settings.HOPFIELD_BETA
        self._max_patterns: int = settings.HOPFIELD_MAX_PATTERNS
        self._pattern_matrix: Optional[np.ndarray] = None  # Cached N×d matrix
        self._pattern_ids: list[int] = []  # Memory IDs for each row
        self._dirty: bool = True  # Flag to rebuild cache

    def _build_pattern_matrix(self) -> None:
        """Fetch hot memories and build the N×d pattern matrix."""
        memories = self._storage.get_all_memories_with_embeddings()

        # Filter by heat > COLD_THRESHOLD
        cold = self._settings.COLD_THRESHOLD
        hot_memories = [m for m in memories if m.get("heat", 0) > cold]

        # Sort by heat descending, cap at max_patterns
        hot_memories.sort(key=lambda m: m.get("heat", 0), reverse=True)
        hot_memories = hot_memories[: self._max_patterns]

        if not hot_memories:
            self._pattern_matrix = np.empty((0, 0), dtype=np.float32)
            self._pattern_ids = []
            self._dirty = False
            return

        rows = []
        ids = []
        for mem in hot_memories:
            emb = mem.get("embedding")
            if emb is None:
                continue
            vec = np.frombuffer(emb, dtype=np.float32).copy()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            rows.append(vec)
            ids.append(mem["id"])

        if rows:
            self._pattern_matrix = np.stack(rows)  # shape (N, d)
        else:
            self._pattern_matrix = np.empty((0, 0), dtype=np.float32)

        self._pattern_ids = ids
        self._dirty = False

    def retrieve(
        self, query_embedding: bytes, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Retrieve memories using Modern Hopfield attention.

        Computes: attention = softmax(β · X · query)
        Returns top_k (memory_id, attention_weight) sorted by weight descending.

        β controls sharpness:
          - low β (1-4): blended context from many memories
          - medium β (4-12): balanced retrieval
          - high β (>12): sharp single-memory retrieval
        """
        if self._dirty or self._pattern_matrix is None:
            self._build_pattern_matrix()

        if self._pattern_matrix.size == 0:
            return []

        # Convert query to numpy, L2-normalize
        query = np.frombuffer(query_embedding, dtype=np.float32).copy()
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # logits = β * (X @ query), shape (N,)
        logits = self._beta * (self._pattern_matrix @ query)

        # attention = softmax(logits)
        attention = _softmax(logits)

        # Get top-k by attention weight
        top_indices = np.argsort(attention)[::-1][:top_k]
        results = [
            (self._pattern_ids[i], float(attention[i]))
            for i in top_indices
            if attention[i] > 0
        ]

        return results

    def retrieve_sparse(
        self, query_embedding: bytes, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Hopfield-Fenchel-Young retrieval using sparsemax.

        Replaces softmax with sparsemax, which projects onto the probability
        simplex and produces EXACT zeros for irrelevant memories.
        """
        if self._dirty or self._pattern_matrix is None:
            self._build_pattern_matrix()

        if self._pattern_matrix.size == 0:
            return []

        query = np.frombuffer(query_embedding, dtype=np.float32).copy()
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        logits = self._beta * (self._pattern_matrix @ query)
        weights = _sparsemax(logits)

        # Filter exact zeros and sort descending
        nonzero_indices = np.nonzero(weights)[0]
        if len(nonzero_indices) == 0:
            return []

        # Sort nonzero by weight descending
        sorted_nz = nonzero_indices[np.argsort(weights[nonzero_indices])[::-1]]
        results = [
            (self._pattern_ids[i], float(weights[i]))
            for i in sorted_nz[:top_k]
        ]

        return results

    def pattern_completion(
        self, partial_embedding: bytes, iterations: int = 5
    ) -> bytes:
        """Iterative Hopfield dynamics for completing partial/noisy queries.

        For each iteration: ξ_new = Xᵀ @ softmax(β · X @ ξ_old)
        Converges to a stored pattern (or blend of nearby patterns).
        """
        if self._dirty or self._pattern_matrix is None:
            self._build_pattern_matrix()

        xi = np.frombuffer(partial_embedding, dtype=np.float32).copy()
        norm = np.linalg.norm(xi)
        if norm > 0:
            xi = xi / norm

        if self._pattern_matrix.size == 0:
            return xi.astype(np.float32).tobytes()

        X = self._pattern_matrix  # (N, d)

        for _ in range(iterations):
            # logits = β · X · ξ, shape (N,)
            logits = self._beta * (X @ xi)
            # attention = softmax(logits), shape (N,)
            attn = _softmax(logits)
            # ξ_new = Xᵀ · attention = sum of weighted patterns, shape (d,)
            xi_new = X.T @ attn
            # L2-normalize the result
            norm = np.linalg.norm(xi_new)
            if norm > 0:
                xi_new = xi_new / norm
            xi = xi_new

        return xi.astype(np.float32).tobytes()

    def invalidate_cache(self) -> None:
        """Mark the pattern matrix as stale (call when memories change)."""
        self._dirty = True

    def get_energy(self, query_embedding: bytes) -> float:
        """Compute Hopfield energy for a query.

        E(ξ, X) = -log(Σ exp(β · xᵢᵀ · ξ)) + 0.5 · |ξ|²

        Lower energy = query is well-represented by stored patterns.
        Uses logsumexp for numerical stability.
        """
        if self._dirty or self._pattern_matrix is None:
            self._build_pattern_matrix()

        query = np.frombuffer(query_embedding, dtype=np.float32).copy()
        norm_sq = float(np.dot(query, query))

        if self._pattern_matrix.size == 0:
            # No patterns: energy is just the norm term
            return 0.5 * norm_sq

        # Normalize query for dot products
        norm = np.linalg.norm(query)
        if norm > 0:
            query_normed = query / norm
        else:
            query_normed = query

        # β · Xᵀ · ξ, shape (N,)
        logits = self._beta * (self._pattern_matrix @ query_normed)

        # E = -logsumexp(logits) + 0.5 * |ξ|²
        energy = -_logsumexp(logits) + 0.5 * norm_sq

        return energy

    def get_pattern_count(self) -> int:
        """Return current number of patterns in the matrix."""
        if self._dirty or self._pattern_matrix is None:
            self._build_pattern_matrix()
        return len(self._pattern_ids)
