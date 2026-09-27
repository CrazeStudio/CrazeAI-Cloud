#!/usr/bin/env bash
set -euo pipefail

if [ -x "llama.cpp/build/bin/llama-server" ]; then
  echo "[CrazeAI] Using cached llama-server"
  exit 0
fi

echo "[CrazeAI] Cloning llama.cpp source..."
if [ ! -d llama.cpp/.git ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp
else
  git -C llama.cpp pull --ff-only
fi

echo "[CrazeAI] Building llama-server (CPU)..."
cmake -S llama.cpp -B llama.cpp/build -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --config Release --target llama-server -j 2

test -x llama.cpp/build/bin/llama-server
echo "[CrazeAI] llama-server ready: llama.cpp/build/bin/llama-server"
