# CrazeAI Cloud

A lightweight Java 25 GGUF inference API for Render. This replaces the previous Python/FastAPI + llama.cpp implementation with in-process JVM inference using [Qxotic jinfer](https://qxotic.ai/).

## Model

- Model family: SmolLM2-360M-Instruct
- Hugging Face GGUF repository: [tensorblock/SmolLM2-360M-Instruct-GGUF](https://huggingface.co/tensorblock/SmolLM2-360M-Instruct-GGUF)
- Selected quantization: **Q4_0** (approximately 229 MB)
- Runtime reference: `tensorblock/SmolLM2-360M-Instruct-GGUF:Q4_0`
- Context: 256 tokens by default; configurable with `CRAZEAI_CONTEXT`

**Why Q4_0 instead of Q2_K?** The smallest Q2_K file is not in jinfer's documented supported quantization list. Q4_0 is the smallest listed quantization that the selected JVM runtime documents. The model-family adapter is jinfer's Llama family; SmolLM2 uses a Llama-style GGUF architecture, but the first successful model load on Render is the compatibility check. The API reports a clear startup error if the checkpoint is rejected.

## Runtime

- Java 25
- Maven
- Qxotic jinfer 0.3.1 (in-process JVM inference; no llama.cpp process, no Python, no external inference server)
- JDK HttpServer
- OpenAI-compatible endpoints
- Render Docker deployment

## API

### Health
`GET /health` — process health and model-load status. Does not expose the API key.

### Readiness
`GET /ready` — returns 200 only after the GGUF model is loaded.

### Models
`GET /v1/models`

### Chat completions
`POST /v1/chat/completions`

Example:
```bash
curl https://YOUR-SERVICE.onrender.com/v1/chat/completions \
  -H "Authorization: Bearer YOUR_CRAZEAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "crazeai-smollm2-360m",
    "messages": [
      {"role": "system", "content": "You are CrazeAI."},
      {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 96
  }'
```

Streaming is supported with `"stream": true` using Server-Sent Events.

## Render deployment

1. Open Render and choose **New → Blueprint**.
2. Select `CrazeStudio/CrazeAI-Cloud`.
3. Render detects `render.yaml` and builds the Docker image.
4. Render generates `CRAZEAI_API_KEY`; copy its value from the service environment settings.
5. Wait for `/ready` to return 200. The first deployment downloads the model into the instance cache.

## Resource warning

The model file alone is approximately 229 MB. Total resident memory also includes the JVM, inference buffers, KV cache and request handling. The free Render instance has a tight memory budget; this configuration uses a short context and small heap, but successful inference on the free tier is **not guaranteed**. If Render terminates the service for memory pressure, use a larger instance or a smaller compatible model. Render's ephemeral filesystem means the model may need to be downloaded again after a fresh instance.

## Local build

Requires JDK 25 and Maven 3.9+.

```bash
mvn -B package
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xms32m -Xmx128m -jar target/crazeai-cloud.jar
```

Set `CRAZEAI_API_KEY` before exposing the API publicly. The API key is generated automatically by the Render Blueprint.
