# Download the GGUF during image build, not every time the free service wakes.
# This keeps Hugging Face network download time out of container startup.
FROM python:3.12-slim AS model-download

ARG MODEL_REPO=bartowski/Qwen2.5-0.5B-Instruct-GGUF
ARG MODEL_FILE=Qwen2.5-0.5B-Instruct-Q3_K_M.gguf

RUN mkdir -p /models && MODEL_REPO="${MODEL_REPO}" MODEL_FILE="${MODEL_FILE}" python - <<'PY'
import os
import time
import urllib.request

repo = os.environ["MODEL_REPO"]
filename = os.environ["MODEL_FILE"]
url = f"https://huggingface.co/{repo}/resolve/main/{filename}?download=true"
target = f"/models/{filename}"
request = urllib.request.Request(url, headers={"User-Agent": "CrazeAI-Model-Builder/1.0"})
last_error = None

for attempt in range(1, 4):
    try:
        with urllib.request.urlopen(request, timeout=90) as response, open(target, "wb") as output:
            total = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"Model download: {downloaded}/{total} bytes", flush=True)
        if os.path.getsize(target) < 100_000_000:
            raise RuntimeError("Downloaded model is unexpectedly small; refusing to build.")
        print(f"Model cached in image: {target} ({os.path.getsize(target)} bytes)", flush=True)
        break
    except Exception as exc:
        last_error = exc
        try:
            os.remove(target)
        except FileNotFoundError:
            pass
        if attempt < 3:
            print(f"Download attempt {attempt} failed: {exc}; retrying.", flush=True)
            time.sleep(3 * attempt)
else:
    raise SystemExit(f"Could not download GGUF model: {last_error}")
PY

FROM ghcr.io/ggml-org/llama.cpp:server

WORKDIR /app
COPY --from=model-download /models /models
COPY docker-entrypoint.sh /usr/local/bin/crazeai-entrypoint
RUN chmod +x /usr/local/bin/crazeai-entrypoint

ENV LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8080 \
    LLAMA_ARG_CTX_SIZE=2048 \
    LLAMA_ARG_N_PARALLEL=2 \
    LLAMA_ARG_THREADS=1 \
    LLAMA_ARG_BATCH=64 \
    LLAMA_ARG_UBATCH=32 \
    LLAMA_ARG_CACHE_TYPE_K=q8_0 \
    LLAMA_ARG_CACHE_TYPE_V=q8_0 \
    LLAMA_ARG_TEMP=0.2 \
    LLAMA_ARG_TOP_P=0.85 \
    LLAMA_ARG_TOP_K=20 \
    LLAMA_ARG_REPEAT_PENALTY=1.15 \
    LLAMA_ARG_REPEAT_LAST_N=128 \
    LLAMA_ARG_UI=false \
    LLAMA_ARG_ALIAS=Qwen/Qwen2.5-0.5B-Instruct

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/crazeai-entrypoint"]
