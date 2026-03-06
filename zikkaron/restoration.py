"""Hippocampal Replay — intelligent context restoration after compaction."""

import json
import logging
from datetime import datetime, timezone

from zikkaron.config import Settings
from zikkaron.storage import StorageEngine
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.retrieval import HippoRetriever
from zikkaron.cognitive_map import CognitiveMap
from zikkaron.metacognition import MetaCognition
from zikkaron.fractal import FractalMemoryTree

logger = logging.getLogger(__name__)


class HippocampalReplay:
    """Reconstructs context after Claude Code compaction events.

    Named after the neuroscience phenomenon where the hippocampus
    replays important experiences during sleep to consolidate them.
    Context compaction IS the 'sleep' — we replay what matters when
    Claude 'wakes up'.
    """

    def __init__(
        self,
        storage: StorageEngine,
        embeddings: EmbeddingEngine,
        retriever: HippoRetriever | None = None,
        cognitive_map: CognitiveMap | None = None,
        metacognition: MetaCognition | None = None,
        fractal: FractalMemoryTree | None = None,
        settings: Settings | None = None,
    ):
        self._storage = storage
        self._embeddings = embeddings
        self._retriever = retriever
        self._cognitive_map = cognitive_map
        self._metacognition = metacognition
        self._fractal = fractal
        self._settings = settings or Settings()
        self._tool_call_count = 0

    def record_tool_call(self):
        """Track tool calls for auto-checkpoint threshold."""
        self._tool_call_count += 1

    def should_auto_checkpoint(self) -> bool:
        """Check if we've hit the auto-checkpoint interval."""
        interval = self._settings.REPLAY_CHECKPOINT_AUTO_INTERVAL
        if interval <= 0:
            return False
        return self._tool_call_count > 0 and self._tool_call_count % interval == 0

    def reset_tool_count(self):
        """Reset after checkpoint."""
        self._tool_call_count = 0

    def create_checkpoint(
        self,
        directory: str,
        current_task: str = "",
        files_being_edited: list[str] | None = None,
        key_decisions: list[str] | None = None,
        open_questions: list[str] | None = None,
        next_steps: list[str] | None = None,
        active_errors: list[str] | None = None,
        custom_context: str = "",
        session_id: str = "default",
    ) -> dict:
        """Create a working state checkpoint for post-compaction recovery."""
        epoch = self._storage.get_current_epoch()
        checkpoint_id = self._storage.insert_checkpoint({
            "session_id": session_id,
            "directory_context": directory,
            "current_task": current_task,
            "files_being_edited": files_being_edited or [],
            "key_decisions": key_decisions or [],
            "open_questions": open_questions or [],
            "next_steps": next_steps or [],
            "active_errors": active_errors or [],
            "custom_context": custom_context,
            "epoch": epoch,
        })
        self.reset_tool_count()
        return {
            "checkpoint_id": checkpoint_id,
            "epoch": epoch,
            "status": "created",
        }

    def anchor_memory(self, content: str, context: str, tags: list[str], reason: str = "") -> int:
        """Store a memory with maximum protection — survives compaction restoration.

        Anchored memories get heat=1.0, is_protected=True, importance=1.0.
        They are ALWAYS included in restoration regardless of other scoring.
        """
        embedding = self._embeddings.encode(content)
        memory_id = self._storage.insert_memory({
            "content": content,
            "embedding": embedding,
            "tags": tags + ["_anchor"],
            "directory_context": context,
            "heat": self._settings.REPLAY_ANCHOR_HEAT,
            "is_stale": False,
            "file_hash": None,
            "embedding_model": self._embeddings.get_model_name(),
        })
        # Set protection and importance flags
        self._storage._conn.execute(
            "UPDATE memories SET is_protected = 1, importance = 1.0 WHERE id = ?",
            (memory_id,),
        )
        if reason:
            self._storage._conn.execute(
                "UPDATE memories SET contextual_prefix = ? WHERE id = ?",
                (f"[ANCHOR: {reason}] ", memory_id),
            )
        self._storage._conn.commit()
        return memory_id

    def pre_compact_drain(self, directory: str) -> dict:
        """Emergency context capture before compaction.

        Called by PreCompact hook. Triggers:
        1. Auto-checkpoint from sensory buffer
        2. Epoch increment (marks compaction boundary)
        3. Emergency consolidation
        """
        new_epoch = self._storage.increment_epoch()

        # Create an auto-checkpoint if no recent one exists
        active = self._storage.get_active_checkpoint()
        auto_created = False
        if active is None or active.get("epoch", 0) < new_epoch - 1:
            self._storage.insert_checkpoint({
                "session_id": "auto-drain",
                "directory_context": directory,
                "current_task": "[auto-captured before compaction]",
                "epoch": new_epoch,
            })
            auto_created = True
        else:
            # Update existing checkpoint with new epoch
            self._storage._conn.execute(
                "UPDATE checkpoints SET epoch = ? WHERE id = ?",
                (new_epoch, active["id"]),
            )
            self._storage._conn.commit()

        return {
            "status": "drained",
            "epoch": new_epoch,
            "auto_checkpoint_created": auto_created,
        }

    def restore(self, directory: str = "") -> dict:
        """Intelligent context reconstruction after compaction.

        Combines:
        1. Latest checkpoint (what you were doing)
        2. Anchored memories (critical facts, always included)
        3. Hot project memories (thermodynamic ranking)
        4. Predictive retrieval via SR (what you'll likely need next)
        5. Gap detection (what might have been lost)

        Returns structured data + formatted markdown for injection.
        """
        max_memories = self._settings.REPLAY_MAX_RESTORE_MEMORIES

        # 1. Get latest checkpoint
        checkpoint = self._storage.get_active_checkpoint()

        # 2. Get anchored memories (always included)
        anchored = self._storage._conn.execute(
            """SELECT * FROM memories
               WHERE is_protected = 1 AND heat > 0
               AND tags LIKE '%_anchor%'
               ORDER BY created_at DESC LIMIT ?""",
            (max_memories,),
        ).fetchall()
        anchored = [dict(r) for r in anchored]
        for m in anchored:
            m.pop("embedding", None)
            m.pop("hdc_vector", None)
            if isinstance(m.get("tags"), str):
                m["tags"] = json.loads(m["tags"])

        # 3. Hot project memories
        hot_memories = []
        if directory:
            hot_memories = self._storage.get_memories_for_directory(
                directory, min_heat=self._settings.HOT_THRESHOLD
            )
        else:
            hot_memories = self._storage.get_memories_by_heat(
                self._settings.HOT_THRESHOLD
            )
        for m in hot_memories:
            m.pop("embedding", None)
            m.pop("hdc_vector", None)

        # Exclude anchored IDs from hot to avoid duplicates
        anchor_ids = {m["id"] for m in anchored}
        hot_memories = [m for m in hot_memories if m["id"] not in anchor_ids]
        hot_memories = hot_memories[:max_memories]

        # 4. Predictive retrieval via SR cognitive map
        predicted = []
        if self._cognitive_map is not None and self._cognitive_map.has_sufficient_data():
            # Use checkpoint task as query for SR navigation
            query = ""
            if checkpoint:
                query = checkpoint.get("current_task", "")
            if not query and directory:
                query = f"project work in {directory}"
            if query:
                query_emb = self._embeddings.encode(query)
                if query_emb is not None:
                    sr_results = self._cognitive_map.navigate_to(
                        query_emb, self._embeddings, top_k=max_memories // 2
                    )
                    seen_ids = anchor_ids | {m["id"] for m in hot_memories}
                    for mid, proximity in sr_results:
                        if mid not in seen_ids:
                            mem = self._storage.get_memory(mid)
                            if mem:
                                mem.pop("embedding", None)
                                mem.pop("hdc_vector", None)
                                mem["_sr_proximity"] = round(proximity, 4)
                                predicted.append(mem)
                                seen_ids.add(mid)

        # 5. Gap detection
        gaps = []
        if self._metacognition is not None and directory:
            try:
                gaps = self._metacognition.detect_gaps(directory)[:3]
            except Exception:
                logger.debug("Gap detection failed during restore")

        # Build formatted markdown for hook injection
        markdown = self._format_restoration(
            checkpoint, anchored, hot_memories, predicted, gaps, directory
        )

        return {
            "checkpoint": checkpoint,
            "anchored_memories": len(anchored),
            "hot_memories": len(hot_memories),
            "predicted_memories": len(predicted),
            "gaps_detected": len(gaps),
            "epoch": checkpoint.get("epoch", 0) if checkpoint else 0,
            "formatted": markdown,
        }

    def _format_restoration(
        self,
        checkpoint: dict | None,
        anchored: list[dict],
        hot: list[dict],
        predicted: list[dict],
        gaps: list[dict],
        directory: str,
    ) -> str:
        """Format restoration data as injectable markdown."""
        lines = []
        lines.append("# Zikkaron Context Restoration (Hippocampal Replay)")
        lines.append("")

        # Checkpoint section
        if checkpoint:
            lines.append("## What You Were Doing")
            if checkpoint.get("current_task"):
                lines.append(f"**Task:** {checkpoint['current_task']}")
            if checkpoint.get("files_being_edited"):
                files = checkpoint["files_being_edited"]
                if isinstance(files, str):
                    files = json.loads(files)
                if files:
                    lines.append(f"**Files:** {', '.join(files)}")
            if checkpoint.get("key_decisions"):
                decisions = checkpoint["key_decisions"]
                if isinstance(decisions, str):
                    decisions = json.loads(decisions)
                if decisions:
                    lines.append("**Decisions:**")
                    for d in decisions:
                        lines.append(f"- {d}")
            if checkpoint.get("open_questions"):
                questions = checkpoint["open_questions"]
                if isinstance(questions, str):
                    questions = json.loads(questions)
                if questions:
                    lines.append("**Open questions:**")
                    for q in questions:
                        lines.append(f"- {q}")
            if checkpoint.get("next_steps"):
                steps = checkpoint["next_steps"]
                if isinstance(steps, str):
                    steps = json.loads(steps)
                if steps:
                    lines.append("**Next steps:**")
                    for s in steps:
                        lines.append(f"- {s}")
            if checkpoint.get("active_errors"):
                errors = checkpoint["active_errors"]
                if isinstance(errors, str):
                    errors = json.loads(errors)
                if errors:
                    lines.append("**Active errors:**")
                    for e in errors:
                        lines.append(f"- {e}")
            if checkpoint.get("custom_context"):
                lines.append(f"\n{checkpoint['custom_context']}")
            lines.append("")

        # Anchored facts
        if anchored:
            lines.append("## Critical Facts (Anchored)")
            for m in anchored:
                prefix = m.get("contextual_prefix", "")
                content = m.get("content", "")
                # Strip the [ANCHOR: reason] prefix for cleaner display
                lines.append(f"- {content}")
            lines.append("")

        # Hot memories
        if hot:
            lines.append("## Active Project Context")
            for m in hot[:6]:
                content = m.get("content", "")
                # Truncate long memories for restoration
                if len(content) > 200:
                    content = content[:200] + "..."
                heat = m.get("heat", 0)
                lines.append(f"- [{heat:.1f}] {content}")
            lines.append("")

        # Predicted memories
        if predicted:
            lines.append("## Predicted Context (SR Navigation)")
            for m in predicted[:4]:
                content = m.get("content", "")
                if len(content) > 200:
                    content = content[:200] + "..."
                lines.append(f"- {content}")
            lines.append("")

        # Gaps
        if gaps:
            lines.append("## Knowledge Gaps Detected")
            for g in gaps:
                lines.append(f"- **{g.get('type', 'unknown')}**: {g.get('description', '')}")
            lines.append("")

        if directory:
            lines.append(f"*Restored for directory: {directory}*")

        return "\n".join(lines)
