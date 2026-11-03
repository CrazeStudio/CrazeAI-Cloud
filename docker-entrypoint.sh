#!/bin/sh
set -eu

if [ -z "${API_KEY:-}" ] && [ -z "${LLAMA_API_KEY:-}" ]; then
  echo "ERROR: Set API_KEY in Render environment variables." >&2
  exit 1
fi

export LLAMA_API_KEY="${LLAMA_API_KEY:-$API_KEY}"

exec /app/llama-server \
  --host "${LLAMA_ARG_HOST:-0.0.0.0}" \
  --port "${PORT:-${LLAMA_ARG_PORT:-8080}}" \
  --model-url "${LLAMA_ARG_MODEL_URL:?LLAMA_ARG_MODEL_URL is required}" \
  --alias "${LLAMA_ARG_ALIAS:-HuggingFaceTB/SmolLM2-135M-Instruct}" \
  --ctx-size "${LLAMA_ARG_CTX_SIZE:-1024}" \
  --threads "${LLAMA_ARG_THREADS:-1}" \
  --threads-batch "${LLAMA_ARG_THREADS:-1}" \
  --batch-size "${LLAMA_ARG_BATCH:-64}" \
  --ubatch-size "${LLAMA_ARG_UBATCH:-32}" \
  --parallel "${LLAMA_ARG_N_PARALLEL:-1}" \
  --cache-type-k "${LLAMA_ARG_CACHE_TYPE_K:-q8_0}" \
  --cache-type-v "${LLAMA_ARG_CACHE_TYPE_V:-q8_0}" \
  --no-warmup \
  --no-webui
