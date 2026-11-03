#!/bin/sh
set -eu

if [ -z "${API_KEY:-}" ] && [ -z "${LLAMA_API_KEY:-}" ]; then
  echo "ERROR: Set API_KEY in Render environment variables." >&2
  exit 1
fi

export LLAMA_API_KEY="${LLAMA_API_KEY:-$API_KEY}"
exec /app/llama-server "$@"
