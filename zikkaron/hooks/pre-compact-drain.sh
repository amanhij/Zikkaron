#!/usr/bin/env bash
# Zikkaron Hippocampal Replay — PreCompact Hook
# Drains context into Zikkaron before Claude Code compacts the conversation.
# Reads hook input from stdin (JSON with session_id, cwd, trigger).

ZIKKARON_PORT="${ZIKKARON_PORT:-8742}"
ZIKKARON_URL="http://localhost:${ZIKKARON_PORT}"

# Read hook input from stdin
INPUT=$(cat)

# Extract cwd from hook input, fallback to current directory
CWD=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('cwd',''))" 2>/dev/null || echo "")
if [ -z "$CWD" ]; then
    CWD=$(pwd)
fi

# Call Zikkaron's pre-compact endpoint
curl -s -X POST "${ZIKKARON_URL}/hooks/pre-compact" \
    -H "Content-Type: application/json" \
    -d "{\"cwd\": \"${CWD}\"}" \
    -m 5 > /dev/null 2>&1

exit 0
