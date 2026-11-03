# CrazeAI Cloud

A low-memory, OpenAI-compatible inference API running **llama.cpp** in its official CPU server container.

## Runtime and model

- Runtime: official `ghcr.io/ggml-org/llama.cpp:server` image.
- Model: `bartowski/SmolLM2-135M-Instruct-GGUF`
- Quantization: `Q3_K_M` (approximately 94 MB model file).
- Context: 1,024 tokens.
- Parallel requests: 1.
- CPU threads: 1.
- Batch / micro-batch: 64 / 32.
- The model is downloaded automatically from Hugging Face at startup.

This deliberately replaces the previous Java/Jlama + BF16 loading path. Jlama was failing with `UnsupportedOperationException: BF16`; llama.cpp loads a pre-quantized GGUF model directly and avoids the BF16 conversion path. The 135M model is smaller than the previous 360M model, trading some response quality for lower memory usage.

## API

- `GET /health` — server health.
- `GET /v1/models` — model list.
- `POST /v1/chat/completions` — OpenAI-compatible chat completions, including streaming.

All API routes require `Authorization: Bearer YOUR_API_KEY`.

Example:

```bash
curl https://YOUR-SERVICE.onrender.com/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"HuggingFaceTB/SmolLM2-135M-Instruct","messages":[{"role":"user","content":"Reply exactly: CrazeAI works."}],"max_tokens":48}'
```

## Render deployment

1. Deploy this repository as a Docker Web Service using `render.yaml`.
2. Keep the existing `API_KEY` environment variable; the entrypoint forwards it securely to llama.cpp as `LLAMA_API_KEY`.
3. Wait for the Hugging Face model download and model-load messages in Render logs.
4. Test `GET /health`, then `POST /v1/chat/completions`.

The service uses Render's ephemeral filesystem, so a restart or redeploy may download the model again. Render free instances can sleep when idle. The 1,024-token context and single request slot are intentional memory-saving settings; increase them only if the service has enough RAM.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `API_KEY` | required | Existing bearer secret; never expose it in source code |
| `LLAMA_ARG_MODEL_URL` | Q3_K_M GGUF URL | Model file downloaded at startup |
| `LLAMA_ARG_CTX_SIZE` | 1024 | Maximum context allocation |
| `LLAMA_ARG_N_PARALLEL` | 1 | One active request slot |
| `LLAMA_ARG_THREADS` | 1 | CPU inference threads |
| `LLAMA_ARG_BATCH` | 64 | Prompt batch size |
| `LLAMA_ARG_UBATCH` | 32 | Physical micro-batch size |

The previous Gradle/Jlama Java source remains in the repository for reference, but the Docker deployment no longer builds or starts it.
