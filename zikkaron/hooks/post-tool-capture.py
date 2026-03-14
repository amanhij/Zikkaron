#!/usr/bin/env python3
"""Zikkaron auto-capture — PostToolCall hook handler.

Reads tool call JSON from stdin, writes to action_log table.
Only imports stdlib (sqlite3, json, sys) — no ML model loading.
Runs in <100ms. Backgrounded by the shell wrapper for zero latency.

Works in both stdio and HTTP transport modes because it writes
directly to the shared SQLite database.
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


# Tools to skip (Zikkaron's own tools — prevents infinite loops)
_SKIP_PREFIXES = ("mcp__zikkaron__",)

# High-value tool input fields to extract as summary
_SUMMARY_FIELDS = (
    "command", "content", "query", "file_path", "pattern",
    "prompt", "old_string", "skill", "description",
)


def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return

    tool_name = data.get("tool_name", "unknown")

    # Skip Zikkaron's own tools to prevent capture loops
    for prefix in _SKIP_PREFIXES:
        if tool_name.startswith(prefix):
            return

    cwd = data.get("cwd", "")
    session_id = data.get("session_id", "")

    # Extract a brief summary from the tool input
    tool_input = data.get("tool_input", {})
    summary = ""
    if isinstance(tool_input, dict):
        for field in _SUMMARY_FIELDS:
            val = tool_input.get(field)
            if val:
                summary = str(val)[:200]
                break
        if not summary:
            summary = str(tool_input)[:200]
    else:
        summary = str(tool_input)[:200]

    # Find the database
    db_path = Path("~/.zikkaron/memory.db").expanduser()
    if not db_path.exists():
        return

    try:
        conn = sqlite3.connect(str(db_path), timeout=1)
        conn.execute(
            "INSERT INTO action_log (tool_name, tool_input_summary, directory, session_id, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (tool_name, summary, cwd, session_id, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception:
        # Never fail the hook — swallow all errors
        pass


if __name__ == "__main__":
    main()
