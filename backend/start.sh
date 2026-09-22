#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/tmp/crazeai-models}"
MODEL_FILE="${MODEL_FILE:-${MODEL_DIR}/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf}"
LLAMA_BIN="${LLAMA_BIN:-./llama.cpp/build/bin/llama-server}"
DOWNLOAD_FILE="${MODEL_FILE}.part"

mkdir -p "${MODEL_DIR}"

if [ ! -s "${MODEL_FILE}" ]; then
  echo "Downloading CrazeAI model..."
  rm -f "${DOWNLOAD_FILE}"
  curl -L --fail --retry 8 --retry-delay 3 --connect-timeout 20 --progress-bar     "${MODEL_URL}" -o "${DOWNLOAD_FILE}"
  test -s "${DOWNLOAD_FILE}"
  mv "${DOWNLOAD_FILE}" "${MODEL_FILE}"
  echo "Model download complete."
else
  echo "CrazeAI model already exists."
fi

if [ ! -x "${LLAMA_BIN}" ]; then
  echo "ERROR: llama-server not found at ${LLAMA_BIN}"
  exit 1
fi

echo "Starting llama.cpp..."
"${LLAMA_BIN}"   --model "${MODEL_FILE}"   --host 127.0.0.1   --port "${LLAMA_PORT}"   --ctx-size "${LLAMA_CTX}"   --threads "${LLAMA_THREADS}"   --n-predict "${MAX_TOKENS}"   --parallel 1   > /tmp/llama.log 2>&1 &

LLAMA_PID=$!

echo "Waiting for llama.cpp to load the model..."
for i in $(seq 1 600); do
  if ! kill -0 "${LLAMA_PID}" 2>/dev/null; then
    echo "ERROR: llama-server exited during startup."
    cat /tmp/llama.log || true
    exit 1
  fi

  if curl -fsS "http://127.0.0.1:${LLAMA_PORT}/health" >/dev/null 2>&1; then
    echo "llama.cpp is ready."
    break
  fi

  if [ "$((i % 10))" -eq 0 ]; then
    echo "Still loading model... ${i}s"
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:${LLAMA_PORT}/health" >/dev/null 2>&1; then
  echo "ERROR: llama.cpp did not become ready within 10 minutes."
  cat /tmp/llama.log || true
  exit 1
fi

exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-10000}"
