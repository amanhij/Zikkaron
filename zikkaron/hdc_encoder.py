"""Hyperdimensional Computing (HDC) / Vector Symbolic Architecture (VSA) encoder.

Encodes memories as compositions of role-filler bindings in high-dimensional
bipolar space, enabling structured queries that vector similarity alone cannot
handle.  Based on Kanerva (1988) Sparse Distributed Memory and Frady et al.
(Neural Computation, 2020) resonator networks.

Operations (pure numpy, no external deps):
  bind(a, b)     = element-wise multiply → creates association
  bundle(a, ...) = element-wise sum + sign → creates superposition
  permute(a, k)  = circular shift → encodes sequence / order
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class HDCEncoder:
    """Hyperdimensional computing encoder for compositional memory representation."""

    def __init__(self, dimensions: int = 10000, seed: int = 42) -> None:
        self._dim = dimensions
        self._rng = np.random.default_rng(seed)
        self._codebook: dict[str, np.ndarray] = {}

        # Pre-generate role vectors for known roles
        self._roles: dict[str, np.ndarray] = {
            "directory": self._random_vector(),
            "entity": self._random_vector(),
            "tag": self._random_vector(),
            "type": self._random_vector(),
            "purpose": self._random_vector(),
            "time_bucket": self._random_vector(),
        }

    @property
    def dimensions(self) -> int:
        return self._dim

    def _random_vector(self) -> np.ndarray:
        """Generate a random bipolar vector {-1, +1}^dimensions."""
        return self._rng.choice([-1.0, 1.0], size=self._dim)

    def get_or_create_atom(self, name: str) -> np.ndarray:
        """Get or create an atomic symbol vector for a given name.

        Atoms are the atomic symbols — "auth.py", "bug", "/myproject".
        Each unique name gets a unique random bipolar vector, created
        deterministically from the seeded RNG on first access.
        """
        if name not in self._codebook:
            self._codebook[name] = self._random_vector()
        return self._codebook[name]

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind two vectors via element-wise multiplication.

        The result is dissimilar to both inputs (key property of binding).
        bind(role, filler) creates a role-filler association.
        """
        return a * b

    def bundle(self, *vectors: np.ndarray) -> np.ndarray:
        """Bundle vectors via element-wise sum + sign normalization.

        Creates a superposition that is similar to each component.
        Returns bipolar vector in {-1, 0, +1}; zeros are replaced
        with random ±1 to maintain full bipolarity.
        """
        if len(vectors) == 0:
            return self._random_vector()
        if len(vectors) == 1:
            return vectors[0].copy()

        result = np.sum(vectors, axis=0)
        signed = np.sign(result)

        # Replace zeros with random ±1 to maintain bipolarity
        zero_mask = signed == 0
        if np.any(zero_mask):
            n_zeros = int(np.sum(zero_mask))
            signed[zero_mask] = self._rng.choice([-1.0, 1.0], size=n_zeros)

        return signed

    def permute(self, vector: np.ndarray, shift: int = 1) -> np.ndarray:
        """Circular right shift by 'shift' positions.

        Used to encode sequence/order — permuted vectors are
        dissimilar to the original.
        """
        return np.roll(vector, shift)

    def encode_memory(
        self,
        directory: str,
        tags: list[str],
        entities: list[str],
        store_type: str = "episodic",
    ) -> np.ndarray:
        """Build a compositional HDC vector for a memory.

        Encodes structured attributes as role-filler bindings and
        bundles them into a single superposition vector.
        """
        components: list[np.ndarray] = []

        # Directory binding
        components.append(
            self.bind(self._roles["directory"], self.get_or_create_atom(directory))
        )

        # Tag bindings
        for tag in tags:
            components.append(
                self.bind(self._roles["tag"], self.get_or_create_atom(tag))
            )

        # Entity bindings
        for entity in entities:
            components.append(
                self.bind(self._roles["entity"], self.get_or_create_atom(entity))
            )

        # Store type binding
        components.append(
            self.bind(self._roles["type"], self.get_or_create_atom(store_type))
        )

        if not components:
            return self._random_vector()

        return self.bundle(*components)

    def encode_query(
        self,
        directory: str | None = None,
        tags: list[str] | None = None,
        entities: list[str] | None = None,
        store_type: str | None = None,
    ) -> np.ndarray:
        """Build a query vector from specified attributes only.

        Only binds specified fields (None = don't include in query).
        This enables partial queries: encode_query(entities=["auth.py"])
        matches ALL memories about auth.py regardless of directory or tags.
        """
        components: list[np.ndarray] = []

        if directory is not None:
            components.append(
                self.bind(self._roles["directory"], self.get_or_create_atom(directory))
            )

        if tags is not None:
            for tag in tags:
                components.append(
                    self.bind(self._roles["tag"], self.get_or_create_atom(tag))
                )

        if entities is not None:
            for entity in entities:
                components.append(
                    self.bind(self._roles["entity"], self.get_or_create_atom(entity))
                )

        if store_type is not None:
            components.append(
                self.bind(self._roles["type"], self.get_or_create_atom(store_type))
            )

        if not components:
            return self._random_vector()

        return self.bundle(*components)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity in HDC space."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b + 1e-8))

    def search(
        self,
        query_vec: np.ndarray,
        candidates: list[tuple[int, np.ndarray]],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Search candidates by HDC similarity.

        Returns top_k (id, similarity) pairs sorted descending.
        """
        scored = [
            (cid, self.similarity(query_vec, cvec))
            for cid, cvec in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def to_bytes(self, vector: np.ndarray) -> bytes:
        """Serialize HDC vector for SQLite BLOB storage."""
        return vector.astype(np.float32).tobytes()

    def from_bytes(self, data: bytes) -> np.ndarray:
        """Deserialize HDC vector from SQLite BLOB."""
        return np.frombuffer(data, dtype=np.float32).copy()
