#!/usr/bin/env bash
set -euo pipefail

mkdir -p /opt/models

if [ ! -s "${MODEL_FILE}" ]; then
  echo "Downloading CrazeAI model..."
  curl -L --fail --retry 5 --retry-delay 3 --progress-bar "${MODEL_URL}" -o "${MODEL_FILE}"
fi

echo "Starting llama.cpp..."
/opt/llama.cpp/build/bin/llama-server \
  --model "${MODEL_FILE}" \
  --host 127.0.0.1 \
  --port "${LLAMA_PORT}" \
  --ctx-size "${LLAMA_CTX}" \
  --threads "${LLAMA_THREADS}" \
  --n-predict "${MAX_TOKENS}" \
  --parallel 1 \
  > /tmp/llama.log 2>&1 &

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
