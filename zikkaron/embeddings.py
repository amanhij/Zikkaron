"""Local embedding engine wrapping sentence-transformers for semantic operations."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


MODEL_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-MiniLM-L6-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "nomic-ai/nomic-embed-text-v1.5": 768,
}

# Backward-compatible alias
_MODEL_DIMENSIONS = MODEL_DIMENSIONS


class EmbeddingEngine:
    """Lazy-loading wrapper around SentenceTransformer for local embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._unavailable = False

    def _ensure_model(self) -> None:
        """Load the SentenceTransformer model if not already loaded."""
        if self._model is not None or self._unavailable:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers is not installed; "
                "embedding operations will return None"
            )
            self._unavailable = True

    def get_model_name(self) -> str:
        """Return the current model name."""
        return self.model_name

    def get_dimensions(self) -> int:
        """Return the embedding dimensionality for the current model."""
        if self.model_name in MODEL_DIMENSIONS:
            return MODEL_DIMENSIONS[self.model_name]
        # Fallback: load model and check
        self._ensure_model()
        if self._model is not None:
            dim = self._model.get_sentence_embedding_dimension()
            return dim
        return 384  # safe default

    def needs_reembedding(self, stored_model: str) -> bool:
        """Check if a memory's stored model differs from the current model."""
        if stored_model is None:
            return True
        return stored_model != self.model_name

    def encode_adaptive(self, text: str, dimensions: int = None) -> Optional[bytes]:
        """Encode text with Matryoshka adaptive dimensionality.

        If dimensions < the model's native dimensions, truncate and re-normalize
        the embedding vector. This produces compact embeddings for less important
        memories while preserving directional quality.
        """
        self._ensure_model()
        if self._unavailable:
            return None
        vec = self._model.encode(text)
        arr = np.asarray(vec, dtype=np.float32)
        native_dim = len(arr)
        if dimensions is not None and dimensions < native_dim:
            arr = arr[:dimensions]
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
        return arr.tobytes()

    def batch_reembed(self, texts: list[str]) -> list[Optional[bytes]]:
        """Efficiently re-embed a batch of texts with the current model."""
        return self.encode_batch(texts)

    @staticmethod
    def quantize(embedding: bytes, bits: int = 8) -> bytes:
        """Quantize float32 embedding to int8 for storage efficiency."""
        arr = np.frombuffer(embedding, dtype=np.float32)
        if bits == 8:
            max_val = np.max(np.abs(arr))
            if max_val == 0:
                return np.zeros(len(arr), dtype=np.int8).tobytes()
            scaled = np.clip(arr / max_val * 127, -127, 127)
            return scaled.astype(np.int8).tobytes()
        raise ValueError(f"Unsupported quantization bits: {bits}")

    @staticmethod
    def dequantize(quantized: bytes, bits: int = 8) -> bytes:
        """Dequantize int8 back to float32 (approximate)."""
        if bits == 8:
            arr = np.frombuffer(quantized, dtype=np.int8).astype(np.float32)
            arr = arr / 127.0
            return arr.astype(np.float32).tobytes()
        raise ValueError(f"Unsupported dequantization bits: {bits}")

    def encode(self, text: str) -> Optional[bytes]:
        """Encode text to a float32 byte blob for SQLite BLOB storage."""
        self._ensure_model()
        if self._unavailable:
            return None
        vec = self._model.encode(text)
        return np.asarray(vec, dtype=np.float32).tobytes()

    def encode_batch(self, texts: list[str]) -> list[Optional[bytes]]:
        """Batch encode texts for efficiency during consolidation."""
        self._ensure_model()
        if self._unavailable:
            return [None] * len(texts)
        vecs = self._model.encode(texts)
        return [np.asarray(v, dtype=np.float32).tobytes() for v in vecs]

    def similarity(self, embedding_a: bytes, embedding_b: bytes) -> float:
        """Compute cosine similarity between two embedding blobs."""
        a = np.frombuffer(embedding_a, dtype=np.float32)
        b = np.frombuffer(embedding_b, dtype=np.float32)
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    def search(
        self,
        query_embedding: bytes,
        candidate_embeddings: list[tuple[int, bytes]],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Rank candidates by similarity to query, return top_k (id, score) pairs."""
        scored = [
            (mid, self.similarity(query_embedding, emb))
            for mid, emb in candidate_embeddings
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
