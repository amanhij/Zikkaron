#!/usr/bin/env bash
# Zikkaron Hippocampal Replay — PostCompact Rehydration Hook
# Restores context from Zikkaron after Claude Code compacts the conversation.
# stdout is injected into Claude's context.

ZIKKARON_PORT="${ZIKKARON_PORT:-8742}"
ZIKKARON_URL="http://localhost:${ZIKKARON_PORT}"

# Get current directory for context
CWD=$(pwd)

# Call Zikkaron's post-compact endpoint and extract the formatted markdown
RESPONSE=$(curl -s "${ZIKKARON_URL}/hooks/post-compact?directory=${CWD}" -m 10 2>/dev/null)

if [ $? -eq 0 ] && [ -n "$RESPONSE" ]; then
    # Extract the "formatted" field from JSON response
    FORMATTED=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('formatted',''))" 2>/dev/null)
    if [ -n "$FORMATTED" ]; then
        echo "$FORMATTED"
    fi
fi

exit 0
