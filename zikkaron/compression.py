"""Rate-distortion compression — memories progressively degrade from full fidelity
to gist to tags as they age, following information-theoretic optimal forgetting.

Based on:
- Tóth et al. (PLoS Computational Biology, 2020): Optimal forgetting = rate-distortion curve
- MemFly (arXiv:2602.07885, 2025): IB optimization for agentic memory
- Tishby (1999): Information Bottleneck — min I(All;Memory) - β·I(Memory;Future)

Three compression levels:
  Level 0 (recent, < 7 days): Full fidelity — complete content preserved
  Level 1 (medium, 7-30 days): Gist — key sentences + code snippets + entities
  Level 2 (old, > 30 days): Tag — one-line summary + semantic tags only
"""

import logging
import re
from datetime import datetime, timezone

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.storage import StorageEngine

logger = logging.getLogger(__name__)

# Patterns for identifying high-information sentences
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_FILE_PATH_RE = re.compile(
    r"(?:\.{0,2}/)?(?:[\w@.-]+/)+[\w@.-]+\.\w+"
)
_ERROR_RE = re.compile(r"\b\w*(?:Error|Exception|Traceback)\b")
_DECISION_RE = re.compile(
    r"\b(?:decided|chose|choosing|using|switched|migrated|replaced|selected|adopted)\b",
    re.IGNORECASE,
)
_NUMBER_VERSION_RE = re.compile(r"\b\d+(?:\.\d+)+\b")
_ENTITY_NAME_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:[A-Z][a-zA-Z]+)+\b")  # CamelCase


class MemoryCompressor:
    """Compresses memories along the rate-distortion curve.

    High-surprise and high-importance memories resist compression (get more bits).
    Protected memories are never compressed. Semantic store memories are never compressed.
    Original content is always preserved in memory_archives for potential restoration.
    """

    def __init__(
        self,
        storage: StorageEngine,
        embeddings: EmbeddingEngine,
        settings: Settings,
    ) -> None:
        self._storage = storage
        self._embeddings = embeddings
        self._gist_age_hours = settings.COMPRESSION_GIST_AGE_HOURS
        self._tag_age_hours = settings.COMPRESSION_TAG_AGE_HOURS

    def get_compression_schedule(self, memory: dict) -> int:
        """Calculate target compression level based on age and importance.

        Returns:
            0 = full fidelity, 1 = gist, 2 = tag
        """
        # Protected memories: never compress
        if memory.get("is_protected", False):
            return 0

        # Semantic store memories: never compress (schemas should stay intact)
        if memory.get("store_type", "episodic") == "semantic":
            return 0

        created_at_str = memory.get("created_at", "")
        if not created_at_str:
            return 0

        try:
            created_at = datetime.fromisoformat(created_at_str)
        except (ValueError, TypeError):
            return 0

        now = datetime.now(timezone.utc)
        # Handle naive datetimes by assuming UTC
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        hours_elapsed = (now - created_at).total_seconds() / 3600.0

        # Compute resistance multiplier from memory properties
        resistance = 1.0
        if memory.get("importance", 0.5) > 0.7:
            resistance *= 2.0
        if memory.get("surprise_score", 0.0) > 0.6:
            resistance *= 1.5
        if memory.get("confidence", 1.0) > 0.8:
            resistance *= 1.3
        if memory.get("access_count", 0) > 10:
            resistance *= 1.5

        gist_threshold = self._gist_age_hours * resistance
        tag_threshold = self._tag_age_hours * resistance

        if hours_elapsed < gist_threshold:
            return 0
        elif hours_elapsed < tag_threshold:
            return 1
        else:
            return 2

    def compress_to_gist(self, memory_id: int) -> str:
        """Compress a memory to gist level (level 1).

        Extracts key sentences containing entities, code blocks, file paths,
        error messages, decision keywords, and numbers/versions.
        Preserves first and last sentences (primacy-recency effect).
        Targets ~30% of original length. Preserves ALL code blocks verbatim.

        Returns the gist content.
        """
        memory = self._storage.get_memory(memory_id)
        if memory is None:
            return ""

        if memory.get("compression_level", 0) >= 1:
            return memory["content"]

        original_content = memory["content"]

        # Archive original content
        self._storage.insert_archive({
            "original_memory_id": memory_id,
            "content": original_content,
            "embedding": memory.get("embedding"),
            "archive_reason": "compression",
        })

        # Generate gist
        gist = self._extract_gist(original_content)

        # Re-encode embedding for gist content
        new_embedding = self._embeddings.encode(gist)

        # Update memory
        self._storage.update_memory_compression(
            memory_id=memory_id,
            content=gist,
            embedding=new_embedding,
            compression_level=1,
            original_content=original_content,
        )

        return gist

    def compress_to_tag(self, memory_id: int) -> str:
        """Compress a memory to tag level (level 2).

        Generates a one-line summary + semantic tags.
        Format: "[summary] | Tags: [entity1, entity2, ...] | Created: [date]"
        Target: < 200 characters.

        Returns the tag content.
        """
        memory = self._storage.get_memory(memory_id)
        if memory is None:
            return ""

        if memory.get("compression_level", 0) >= 2:
            return memory["content"]

        # If at level 0, compress to gist first
        if memory.get("compression_level", 0) == 0:
            self.compress_to_gist(memory_id)
            # Re-fetch the now-gist memory
            memory = self._storage.get_memory(memory_id)
            if memory is None:
                return ""

        gist_content = memory["content"]

        # Archive gist content
        self._storage.insert_archive({
            "original_memory_id": memory_id,
            "content": gist_content,
            "embedding": memory.get("embedding"),
            "archive_reason": "compression",
        })

        # Generate tag representation
        tag_repr = self._generate_tag(gist_content, memory)

        # Re-encode embedding for tag content
        new_embedding = self._embeddings.encode(tag_repr)

        # Update memory
        self._storage.update_memory_compression(
            memory_id=memory_id,
            content=tag_repr,
            embedding=new_embedding,
            compression_level=2,
        )

        return tag_repr

    def compression_cycle(self) -> dict:
        """Run compression on all eligible memories.

        Returns stats dict with counts of compressed, skipped, etc.
        """
        stats = {
            "compressed_to_gist": 0,
            "compressed_to_tag": 0,
            "already_compressed": 0,
            "protected_skipped": 0,
            "semantic_skipped": 0,
        }

        memories = self._storage.get_all_memories_for_decay()
        for mem in memories:
            if mem.get("is_protected", False):
                stats["protected_skipped"] += 1
                continue

            if mem.get("store_type", "episodic") == "semantic":
                stats["semantic_skipped"] += 1
                continue

            current_level = mem.get("compression_level", 0)
            target_level = self.get_compression_schedule(mem)

            if target_level <= current_level:
                if current_level > 0:
                    stats["already_compressed"] += 1
                continue

            try:
                if target_level == 1 and current_level == 0:
                    self.compress_to_gist(mem["id"])
                    stats["compressed_to_gist"] += 1
                elif target_level == 2:
                    if current_level == 0:
                        self.compress_to_gist(mem["id"])
                        stats["compressed_to_gist"] += 1
                    self.compress_to_tag(mem["id"])
                    stats["compressed_to_tag"] += 1
            except Exception:
                logger.exception("Failed to compress memory %d", mem["id"])

        return stats

    def decompress(self, memory_id: int) -> str:
        """Attempt to restore original content from archive.

        Checks memory_archives first (earliest entry = original full content),
        then falls back to the memory's original_content field,
        then returns current content with a note.
        """
        archives = self._storage.get_archives_for_memory(memory_id)
        if archives:
            # get_archives_for_memory returns DESC order, so last = earliest
            earliest = archives[-1]
            return earliest["content"]

        memory = self._storage.get_memory(memory_id)
        if memory is None:
            return "(no original available)"

        if memory.get("original_content"):
            return memory["original_content"]

        return memory["content"] + " (no original available)"

    # -- Internal helpers --

    def _extract_gist(self, content: str) -> str:
        """Extract gist from full content.

        Strategy:
        1. Extract and preserve all code blocks verbatim
        2. Split remaining text into sentences
        3. Score sentences by information density
        4. Keep first and last sentence (primacy-recency)
        5. Target ~30% of original length
        """
        # Extract code blocks first — they're preserved verbatim
        code_blocks = _CODE_BLOCK_RE.findall(content)
        text_without_code = _CODE_BLOCK_RE.sub("", content)

        # Split into sentences
        sentences = self._split_sentences(text_without_code)
        if not sentences:
            # Only code blocks or empty
            return "\n\n".join(code_blocks) if code_blocks else content

        if len(sentences) <= 3:
            # Short content — keep everything
            parts = list(sentences)
            if code_blocks:
                parts.extend(code_blocks)
            return "\n".join(parts)

        # Score each sentence
        scored = []
        for i, sent in enumerate(sentences):
            score = self._score_sentence(sent)
            # Primacy boost for first sentence
            if i == 0:
                score += 10.0
            # Recency boost for last sentence
            if i == len(sentences) - 1:
                score += 8.0
            scored.append((i, sent, score))

        # Sort by score descending, take enough to reach ~30% of original length
        target_length = max(len(content) * 0.3, 50)
        scored.sort(key=lambda x: x[2], reverse=True)

        selected_indices = set()
        current_length = sum(len(cb) for cb in code_blocks)
        for idx, sent, _score in scored:
            if current_length >= target_length:
                break
            selected_indices.add(idx)
            current_length += len(sent)

        # Always include first and last
        selected_indices.add(0)
        selected_indices.add(len(sentences) - 1)

        # Reconstruct in original order
        gist_sentences = [sentences[i] for i in sorted(selected_indices)]
        parts = gist_sentences
        if code_blocks:
            parts.extend(code_blocks)

        return "\n".join(parts)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences, filtering out empty lines."""
        # Split on sentence-ending punctuation or newlines
        raw = re.split(r'(?<=[.!?])\s+|\n+', text)
        return [s.strip() for s in raw if s.strip()]

    def _score_sentence(self, sentence: str) -> float:
        """Score a sentence by information density for gist extraction."""
        score = 0.0

        # File paths
        if _FILE_PATH_RE.search(sentence):
            score += 3.0

        # Error references
        if _ERROR_RE.search(sentence):
            score += 4.0

        # Decision keywords
        if _DECISION_RE.search(sentence):
            score += 3.0

        # Numbers/versions (specific factual info)
        if _NUMBER_VERSION_RE.search(sentence):
            score += 2.0

        # CamelCase entity names
        if _ENTITY_NAME_RE.search(sentence):
            score += 2.0

        # Inline code (backtick)
        if "`" in sentence:
            score += 2.0

        return score

    def _generate_tag(self, content: str, memory: dict) -> str:
        """Generate a tag representation from content.

        Format: "[summary] | Tags: [entities] | Created: [date]"
        Target: < 200 characters.
        """
        # Extract entity names from content
        entities = set()
        for m in _ENTITY_NAME_RE.finditer(content):
            entities.add(m.group(0))
        for m in _FILE_PATH_RE.finditer(content):
            entities.add(m.group(0))

        # Get first sentence as summary
        sentences = self._split_sentences(content)
        summary = sentences[0] if sentences else content[:80]

        # Get created date
        created = memory.get("created_at", "")
        if created:
            try:
                dt = datetime.fromisoformat(created)
                date_str = dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                date_str = created[:10]
        else:
            date_str = "unknown"

        # Also include memory tags if available
        mem_tags = memory.get("tags", [])
        if isinstance(mem_tags, list):
            entities.update(mem_tags)

        # Build tag representation
        entity_list = sorted(entities)[:5]  # Limit to 5 entities
        tag_part = ", ".join(entity_list) if entity_list else "general"

        tag_repr = f"{summary} | Tags: {tag_part} | Created: {date_str}"

        # Truncate to < 200 chars
        if len(tag_repr) > 200:
            # Shorten summary to fit
            available = 200 - len(f" | Tags: {tag_part} | Created: {date_str}")
            if available > 10:
                summary = summary[:available - 3] + "..."
            else:
                summary = summary[:30] + "..."
                tag_part = ", ".join(entity_list[:2]) if entity_list else "general"
            tag_repr = f"{summary} | Tags: {tag_part} | Created: {date_str}"

        # Final truncation safety
        if len(tag_repr) > 200:
            tag_repr = tag_repr[:197] + "..."

        return tag_repr
