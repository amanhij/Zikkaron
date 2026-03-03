"""CRDT-based multi-agent memory sharing with automatic conflict resolution.

Uses three CRDT types:
- OR-Set (Observed-Remove Set): For the memory collection — any agent can add,
  deprecation requires the adding agent's consent.
- LWW-Register (Last-Writer-Wins): For scalar metadata (heat, scores) —
  latest timestamp wins.
- MV-Register (Multi-Value): For content conflicts — preserve ALL versions
  until application-level resolution.

Each memory carries immutable provenance (which agent created/modified it).
"""

import copy
import json
import re
from datetime import datetime, timezone

from zikkaron.config import Settings
from zikkaron.storage import StorageEngine


# Marker pattern used to detect multi-value content conflicts
_CONFLICT_MARKER = "--- [Agent: {}] ---"
_CONFLICT_PATTERN = re.compile(r"--- \[Agent: ([^\]]+)\] ---")


class CRDTMemorySync:
    """Manages CRDT-based multi-agent memory synchronization."""

    def __init__(self, storage: StorageEngine, settings: Settings):
        self._storage = storage
        self._settings = settings
        self._agent_id: str = settings.CRDT_AGENT_ID
        self._vector_clock: dict[str, int] = {self._agent_id: 0}

    def get_agent_id(self) -> str:
        return self._agent_id

    def increment_clock(self) -> dict[str, int]:
        self._vector_clock[self._agent_id] = (
            self._vector_clock.get(self._agent_id, 0) + 1
        )
        return dict(self._vector_clock)

    def tag_provenance(self, memory_dict: dict) -> dict:
        """Add provenance metadata to a new memory before storage."""
        memory_dict["provenance_agent"] = self._agent_id
        memory_dict["vector_clock"] = json.dumps(self.increment_clock())
        return memory_dict

    def compare_clocks(self, clock_a: dict, clock_b: dict) -> str:
        """Compare two vector clocks.

        Returns:
            "before"  — A happened-before B (all A[k] <= B[k], at least one <)
            "after"   — A happened-after B
            "concurrent" — neither dominates (some A[k] > B[k] and some B[k] > A[k])
            "equal"   — identical clocks
        """
        all_keys = set(clock_a) | set(clock_b)
        a_le_b = True  # all A <= B?
        b_le_a = True  # all B <= A?

        for k in all_keys:
            va = clock_a.get(k, 0)
            vb = clock_b.get(k, 0)
            if va > vb:
                a_le_b = False
            if vb > va:
                b_le_a = False

        if a_le_b and b_le_a:
            return "equal"
        if a_le_b:
            return "before"
        if b_le_a:
            return "after"
        return "concurrent"

    def merge_memory(self, local: dict, remote: dict) -> dict:
        """Merge two versions of the same memory using CRDT semantics.

        Content (MV-Register): if content differs AND clocks are concurrent,
            preserve both versions with conflict markers.
        Heat/scores (LWW-Register): take version with later last_accessed.
        Tags (OR-Set): union of both tag sets.
        Vector clocks: take max of each agent's counter.
        """
        local_clock = json.loads(local.get("vector_clock", "{}"))
        remote_clock = json.loads(remote.get("vector_clock", "{}"))

        relation = self.compare_clocks(local_clock, remote_clock)

        merged = dict(local)
        has_conflict = False

        # --- Content: MV-Register ---
        local_content = local.get("content", "")
        remote_content = remote.get("content", "")

        if local_content != remote_content:
            if relation == "concurrent":
                remote_agent = remote.get("provenance_agent", "unknown")
                merged["content"] = (
                    local_content
                    + "\n"
                    + _CONFLICT_MARKER.format(remote_agent)
                    + "\n"
                    + remote_content
                )
                has_conflict = True
            elif relation == "before":
                # Remote is newer — take remote content
                merged["content"] = remote_content
            # else relation == "after" — keep local (already in merged)

        # --- Tags: OR-Set (union) ---
        local_tags = local.get("tags", [])
        remote_tags = remote.get("tags", [])
        if isinstance(local_tags, str):
            local_tags = json.loads(local_tags)
        if isinstance(remote_tags, str):
            remote_tags = json.loads(remote_tags)
        merged["tags"] = sorted(set(local_tags) | set(remote_tags))

        # --- Heat/scores: LWW-Register (latest timestamp wins) ---
        local_ts = local.get("last_accessed", "")
        remote_ts = remote.get("last_accessed", "")
        lww_fields = [
            "heat", "surprise_score", "importance", "emotional_valence",
            "confidence", "access_count", "useful_count", "plasticity",
            "stability", "excitability",
        ]
        if remote_ts > local_ts:
            for field in lww_fields:
                if field in remote:
                    merged[field] = remote[field]
            merged["last_accessed"] = remote_ts

        # --- Merge vector clocks: take max per agent ---
        all_agents = set(local_clock) | set(remote_clock)
        merged_clock = {}
        for agent in all_agents:
            merged_clock[agent] = max(
                local_clock.get(agent, 0),
                remote_clock.get(agent, 0),
            )
        merged["vector_clock"] = json.dumps(merged_clock)

        # --- Conflict flag ---
        merged["_conflict"] = has_conflict

        return merged

    def detect_conflicts(self) -> list[dict]:
        """Scan all memories for concurrent modifications.

        A memory is conflicted if its content contains conflict markers.
        """
        all_memories = self._storage.get_all_memories_for_decay()
        conflicts = []

        for mem in all_memories:
            content = mem.get("content", "")
            agents = _CONFLICT_PATTERN.findall(content)
            if agents:
                # Include the provenance agent (author of the first section)
                provenance = mem.get("provenance_agent", "unknown")
                all_agents = [provenance] + agents
                # Deduplicate while preserving order
                seen = set()
                unique_agents = []
                for a in all_agents:
                    if a not in seen:
                        seen.add(a)
                        unique_agents.append(a)
                conflicts.append({
                    "memory_id": mem["id"],
                    "agents": unique_agents,
                    "versions": len(unique_agents),
                })
        return conflicts

    def resolve_conflict(self, memory_id: int, strategy: str = "latest") -> dict:
        """Resolve a content conflict on a memory.

        Strategies:
            "latest"     — keep version from most recent timestamp
            "merge"      — keep merged content with markers cleaned up
            "agent:<id>" — keep specific agent's version
            "longest"    — keep longest content (most information)
        """
        mem = self._storage.get_memory(memory_id)
        if mem is None:
            return {"error": "memory not found", "memory_id": memory_id}

        content = mem.get("content", "")
        if not _CONFLICT_PATTERN.search(content):
            return {"memory_id": memory_id, "status": "no_conflict"}

        # Split content into versions
        parts = _CONFLICT_PATTERN.split(content)
        # parts = [first_section, agent1, section_after_agent1, agent2, ...]
        # Extract version tuples: (agent, text)
        provenance = mem.get("provenance_agent", "unknown")
        versions = []
        # First section belongs to provenance agent
        versions.append((provenance, parts[0].strip()))
        # Subsequent pairs: (agent, text)
        for i in range(1, len(parts), 2):
            agent = parts[i]
            text = parts[i + 1].strip() if i + 1 < len(parts) else ""
            versions.append((agent, text))

        resolved_content = content  # default fallback

        if strategy == "latest":
            # Keep the last version (most recently appended)
            resolved_content = versions[-1][1]
        elif strategy == "merge":
            # Clean up markers but keep all text concatenated
            resolved_content = "\n\n".join(text for _, text in versions if text)
        elif strategy.startswith("agent:"):
            target_agent = strategy[6:]
            for agent, text in versions:
                if agent == target_agent:
                    resolved_content = text
                    break
        elif strategy == "longest":
            resolved_content = max(
                (text for _, text in versions), key=len, default=content
            )

        # Update the memory in storage
        self._storage._conn.execute(
            "UPDATE memories SET content = ?, vector_clock = ? WHERE id = ?",
            (resolved_content, json.dumps(self.increment_clock()), memory_id),
        )
        self._storage._conn.commit()

        updated = self._storage.get_memory(memory_id)
        updated["_resolved"] = True
        updated["_strategy"] = strategy
        return updated

    def sync_memories(self, remote_memories: list[dict]) -> dict:
        """Synchronize with a list of memories from another agent.

        For each remote memory:
          a) Find matching local memory by content similarity
          b) If no match: insert as new (OR-Set add)
          c) If match: compare vector clocks and merge if needed
        """
        stats = {
            "new_from_remote": 0,
            "merged": 0,
            "conflicted": 0,
            "unchanged": 0,
            "total_processed": len(remote_memories),
        }

        local_memories = self._storage.get_all_memories_for_decay()
        local_by_content = {}
        for lm in local_memories:
            local_by_content[lm.get("content", "")] = lm

        for remote in remote_memories:
            remote_content = remote.get("content", "")
            local_match = local_by_content.get(remote_content)

            if local_match is None:
                # No match — insert as new memory (OR-Set add)
                new_mem = {
                    "content": remote_content,
                    "embedding": remote.get("embedding"),
                    "tags": remote.get("tags", []),
                    "directory_context": remote.get("directory_context", ""),
                    "heat": remote.get("heat", 1.0),
                    "is_stale": remote.get("is_stale", False),
                    "file_hash": remote.get("file_hash"),
                    "embedding_model": remote.get("embedding_model"),
                }
                mid = self._storage.insert_memory(new_mem)
                # Set provenance fields
                self._storage._conn.execute(
                    "UPDATE memories SET provenance_agent = ?, vector_clock = ? WHERE id = ?",
                    (
                        remote.get("provenance_agent", "unknown"),
                        remote.get("vector_clock", "{}"),
                        mid,
                    ),
                )
                self._storage._conn.commit()
                stats["new_from_remote"] += 1
            else:
                # Match found — compare and possibly merge
                local_clock = json.loads(local_match.get("vector_clock", "{}"))
                remote_clock = json.loads(remote.get("vector_clock", "{}"))
                relation = self.compare_clocks(local_clock, remote_clock)

                if relation == "equal":
                    stats["unchanged"] += 1
                elif relation == "after":
                    # Local is strictly newer — no update needed
                    stats["unchanged"] += 1
                else:
                    # "before" or "concurrent" — merge needed
                    merged = self.merge_memory(local_match, remote)
                    # Write merged content back
                    self._storage._conn.execute(
                        "UPDATE memories SET content = ?, tags = ?, "
                        "heat = ?, vector_clock = ?, last_accessed = ? WHERE id = ?",
                        (
                            merged["content"],
                            json.dumps(merged["tags"]),
                            merged.get("heat", local_match["heat"]),
                            merged["vector_clock"],
                            merged.get("last_accessed", local_match.get("last_accessed")),
                            local_match["id"],
                        ),
                    )
                    self._storage._conn.commit()
                    if merged.get("_conflict"):
                        stats["conflicted"] += 1
                    stats["merged"] += 1

        return stats

    def get_agent_stats(self) -> dict:
        """Return agent-specific statistics."""
        all_memories = self._storage.get_all_memories_for_decay()
        authored = sum(
            1 for m in all_memories
            if m.get("provenance_agent") == self._agent_id
        )
        conflicts = self.detect_conflicts()
        return {
            "agent_id": self._agent_id,
            "vector_clock": dict(self._vector_clock),
            "memories_authored": authored,
            "conflicts_pending": len(conflicts),
        }
