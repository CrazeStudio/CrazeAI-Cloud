# CrazeAI Cloud

A lightweight, OpenAI-compatible inference API powered by the official **llama.cpp server** and a small Qwen2.5 GGUF model. The repository also includes a standard-library-only streaming terminal client for Termux.

## Production architecture

- **Inference engine:** official `ghcr.io/ggml-org/llama.cpp:server` container.
- **Model:** `bartowski/Qwen2.5-0.5B-Instruct-GGUF`.
- **Quantization:** `Q3_K_M`.
- **Default context:** 2,048 tokens.
- **Default parallel slots:** 2; use 1 on memory-constrained instances.
- **Model delivery:** downloaded during Docker image build and copied into the runtime image. Container restarts do not download it again.

This is a small model intended for low-resource hosting. It can make mistakes and should not be treated as a reliable source of facts.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/ready` | Model readiness, where supported by the server |
| GET | `/v1/models` | List the configured model |
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions and SSE streaming |

Protected endpoints require:

```http
Authorization: Bearer YOUR_API_KEY
```

## Deploy on Render

1. Create a Docker Web Service connected to this repository.
2. Use the repository-root `Dockerfile`.
3. Set the health check path to `/health`.
4. Add `API_KEY` as a private environment variable in the Render dashboard.
5. Deploy and inspect the logs for model download and server startup.

Do not put API keys in source files, Docker build arguments, or committed configuration. The entrypoint requires `API_KEY` (or `LLAMA_API_KEY`) at runtime.

The model file is roughly 355 MB before runtime overhead. A 512 MB instance may not have enough memory for reliable inference. Monitor memory and reduce parallel slots to 1 if the service is killed for out-of-memory use.

## Deploy on Voroa

See [VOROA.md](VOROA.md) for the dashboard-based Docker deployment steps, environment variables, and verification commands. The entrypoint respects the hosting provider's injected `PORT`.

## Termux client

The `crazeai.py` client uses only Python's standard library and supports streaming, bounded recent chat history, and optional local conversation memory.

```sh
pkg install python
curl -L https://raw.githubusercontent.com/CrazeStudio/CrazeAI-Cloud/main/crazeai.py -o crazeai.py
export CRAZEAI_URL=https://crazeai-cloud.onrender.com/v1/chat/completions
export CRAZEAI_API_KEY='your-private-api-key'
python crazeai.py
```

If the key is not set in the environment, the client prompts for it without echoing it.

Commands:
- `/remember TEXT` — save a local note.
- `/memory` — show recent saved memory entries.
- `/forget` — delete local memory.
- `/clear` — archive the current chat and clear active history.
- `exit` — quit.

Memory is stored on the local device at `~/.crazeai_memory.jsonl`; it is not uploaded as a separate memory store.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `API_KEY` | required | Bearer authentication secret |
| `PORT` | provider-defined / 8080 | HTTP listening port |
| `LLAMA_ARG_CTX_SIZE` | 2048 | Context allocation |
| `LLAMA_ARG_N_PARALLEL` | 2 | Concurrent inference slots |
| `LLAMA_ARG_THREADS` | 1 | CPU threads |
| `LLAMA_ARG_BATCH` | 64 | Prompt batch size |
| `LLAMA_ARG_UBATCH` | 32 | Micro-batch size |
| `LLAMA_ARG_CACHE_TYPE_K` | q8_0 | K cache quantization |
| `LLAMA_ARG_CACHE_TYPE_V` | q8_0 | V cache quantization |
| `LLAMA_ARG_TEMP` | 0.2 | Sampling temperature |
| `LLAMA_ARG_TOP_P` | 0.85 | Nucleus sampling |
| `LLAMA_ARG_TOP_K` | 20 | Top-k sampling |
| `LLAMA_ARG_REPEAT_PENALTY` | 1.15 | Repetition penalty |
| `LLAMA_ARG_REPEAT_LAST_N` | 128 | Repetition lookback |
| `LLAMA_ARG_ALIAS` | Qwen/Qwen2.5-0.5B-Instruct | Public model identifier |

## Repository checks

GitHub Actions validates Python syntax and the shell entrypoint on pushes and pull requests. The Docker image build downloads the model, so it is intentionally not repeated as part of the lightweight syntax-validation workflow.
