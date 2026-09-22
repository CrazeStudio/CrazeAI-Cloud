#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/tmp/crazeai-models}"
MODEL_FILE="${MODEL_FILE:-${MODEL_DIR}/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf}"
LLAMA_BIN="${LLAMA_BIN:-./llama.cpp/build/bin/llama-server}"

mkdir -p "${MODEL_DIR}"

if [ ! -s "${MODEL_FILE}" ]; then
  echo "Downloading CrazeAI model..."
  curl -L --fail --retry 5 --retry-delay 3 --progress-bar "${MODEL_URL}" -o "${MODEL_FILE}"
fi

if [ ! -x "${LLAMA_BIN}" ]; then
  echo "llama-server not found at ${LLAMA_BIN}"
  exit 1
fi

echo "Starting llama.cpp..."
"${LLAMA_BIN}"   --model "${MODEL_FILE}"   --host 127.0.0.1   --port "${LLAMA_PORT}"   --ctx-size "${LLAMA_CTX}"   --threads "${LLAMA_THREADS}"   --n-predict "${MAX_TOKENS}"   --parallel 1   > /tmp/llama.log 2>&1 &

for i in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${LLAMA_PORT}/health" >/dev/null 2>&1; then
    echo "llama.cpp is ready."
    break
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:${LLAMA_PORT}/health" >/dev/null 2>&1; then
  cat /tmp/llama.log || true
  exit 1
fi

exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-10000}"
