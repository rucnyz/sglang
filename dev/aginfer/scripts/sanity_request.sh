#!/usr/bin/env bash
# Send a single chat completion to verify the server is alive and producing output.
# Default port 30000 (V4-Flash). Pass PORT=30001 to hit the smoke server instead.

set -euo pipefail
source "$(dirname "$0")/env.sh"

PORT="${PORT:-30000}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
PROMPT="${PROMPT:-Explain in two sentences what a hierarchical KV cache is.}"

curl -sS "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(cat <<EOF
{
  "model": "$MODEL",
  "messages": [{"role": "user", "content": "$PROMPT"}],
  "max_tokens": 128,
  "temperature": 0.2
}
EOF
)" | python -m json.tool
