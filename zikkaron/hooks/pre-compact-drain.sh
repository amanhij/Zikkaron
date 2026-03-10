#!/usr/bin/env bash
# Zikkaron Hippocampal Replay — PreCompact Hook
# Drains context into Zikkaron before Claude Code compacts the conversation.
# Reads hook input from stdin (JSON with session_id, cwd, trigger).

# Read hook input from stdin
INPUT=$(cat)

# Extract cwd from hook input, fallback to current directory
CWD=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('cwd',''))" 2>/dev/null || echo "")
if [ -z "$CWD" ]; then
    CWD=$(pwd)
fi

# Drain context directly via CLI (works in both stdio and SSE mode)
zikkaron drain "$CWD" > /dev/null 2>&1

exit 0
