#!/usr/bin/env python3
"""Zikkaron session context — SessionStart hook handler.

Injects recent project context into Claude's conversation on every
session start. Uses lightweight DB queries only — no ML model loading.

Output goes to stdout and is injected into Claude's context window.
Works in both stdio and HTTP transport modes (reads SQLite directly).
"""

import json
import os
import sqlite3
import sys
from pathlib import Path


def main():
    db_path = Path("~/.zikkaron/memory.db").expanduser()
    if not db_path.exists():
        return

    # Read hook input from stdin to get cwd
    try:
        data = json.load(sys.stdin)
        cwd = data.get("cwd", os.getcwd())
    except Exception:
        cwd = os.getcwd()

    try:
        conn = sqlite3.connect(str(db_path), timeout=2)
        conn.row_factory = sqlite3.Row
    except Exception:
        return

    try:
        # 1. Get latest checkpoint
        checkpoint = conn.execute(
            "SELECT current_task, key_decisions, custom_context, created_at "
            "FROM checkpoints WHERE is_active = 1 "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        # 2. Get hot memories for this directory
        hot = conn.execute(
            "SELECT content, heat, created_at "
            "FROM memories "
            "WHERE directory_context = ? AND heat > 0.5 "
            "ORDER BY heat DESC LIMIT 6",
            (cwd,),
        ).fetchall()

        # 3. Get anchored memories
        anchored = conn.execute(
            "SELECT content FROM memories "
            "WHERE is_protected = 1 AND heat > 0 "
            "AND tags LIKE '%_anchor%' "
            "ORDER BY created_at DESC LIMIT 4"
        ).fetchall()

        # 4. Get recent actions (last 10)
        actions = conn.execute(
            "SELECT tool_name, tool_input_summary, timestamp "
            "FROM action_log "
            "ORDER BY timestamp DESC LIMIT 10"
        ).fetchall()

        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return

    # Only output if we have something useful
    if not hot and not checkpoint and not anchored:
        return

    lines = []
    lines.append("# Zikkaron — Session Context")
    lines.append("")

    if checkpoint and checkpoint["current_task"]:
        task = checkpoint["current_task"]
        if not task.startswith("[auto-captured"):
            lines.append(f"**Last task:** {task}")
            if checkpoint["key_decisions"]:
                try:
                    decisions = json.loads(checkpoint["key_decisions"])
                    if decisions:
                        for d in decisions:
                            lines.append(f"  - {d}")
                except (json.JSONDecodeError, TypeError):
                    pass
            lines.append("")

    if anchored:
        lines.append("## Critical Facts")
        for row in anchored:
            lines.append(f"- {row['content'][:200]}")
        lines.append("")

    if hot:
        lines.append("## Project Context")
        for row in hot:
            content = row["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"- [{row['heat']:.1f}] {content}")
        lines.append("")

    if actions:
        lines.append("## Recent Actions")
        for a in reversed(list(actions)):
            summary = a["tool_input_summary"]
            if len(summary) > 80:
                summary = summary[:80] + "..."
            lines.append(f"- {a['tool_name']}: {summary}")
        lines.append("")

    lines.append(f"*Context for: {cwd}*")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
