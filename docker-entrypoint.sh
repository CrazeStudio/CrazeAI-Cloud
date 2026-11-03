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
  --model "/models/Qwen2.5-0.5B-Instruct-Q3_K_M.gguf" \
  --alias "${LLAMA_ARG_ALIAS:-Qwen/Qwen2.5-0.5B-Instruct}" \
  --ctx-size "${LLAMA_ARG_CTX_SIZE:-2048}" \
  --threads "${LLAMA_ARG_THREADS:-1}" \
  --threads-batch "${LLAMA_ARG_THREADS:-1}" \
  --batch-size "${LLAMA_ARG_BATCH:-64}" \
  --ubatch-size "${LLAMA_ARG_UBATCH:-32}" \
  --parallel "${LLAMA_ARG_N_PARALLEL:-2}" \
  --cache-type-k "${LLAMA_ARG_CACHE_TYPE_K:-q8_0}" \
  --cache-type-v "${LLAMA_ARG_CACHE_TYPE_V:-q8_0}" \
  --temp "${LLAMA_ARG_TEMP:-0.2}" \
  --top-p "${LLAMA_ARG_TOP_P:-0.85}" \
  --top-k "${LLAMA_ARG_TOP_K:-20}" \
  --repeat-penalty "${LLAMA_ARG_REPEAT_PENALTY:-1.15}" \
  --repeat-last-n "${LLAMA_ARG_REPEAT_LAST_N:-128}" \
  --no-webui
