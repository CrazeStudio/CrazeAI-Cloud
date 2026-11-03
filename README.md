# CrazeAI Cloud

A low-memory, OpenAI-compatible inference API running **llama.cpp** in its official CPU server container, with a Termux-friendly streaming client.

## Runtime and model

- Runtime: official `ghcr.io/ggml-org/llama.cpp:server` image.
- Model: `bartowski/Qwen2.5-0.5B-Instruct-GGUF`
- Quantization: `Q3_K_M` (approximately 355 MB model file).
- Context: 1,024 tokens (kept conservative for the Render free instance).
- Parallel requests: 1.
- CPU threads: 1.
- Batch / micro-batch: 64 / 32.
- Model is downloaded automatically from Hugging Face at startup.

The model is deliberately small to reduce RAM. It can still hallucinate or misunderstand questions; sampling controls and a system prompt improve consistency but cannot make a 135M model reliably reason like a much larger model.

## API

- `GET /health` — server health.
- `GET /ready` — readiness, where supported by the deployed server.
- `GET /v1/models` — model list.
- `POST /v1/chat/completions` — OpenAI-compatible chat completions, including SSE streaming.

The API uses `Authorization: Bearer YOUR_API_KEY`.

## Improved client

The repository includes `crazeai.py`, a standard-library-only terminal client suitable for Termux.

Features:
- Streams tokens as the server sends them.
- Sends a strict CrazeAI identity and relevance prompt.
- Uses conservative sampling settings.
- Keeps at most the latest 10 messages in active chat history.
- Archives older messages locally and retrieves only keyword-relevant excerpts.
- Retries context-overflow errors with progressively shorter history.
- Keeps the API key out of the source file.

Run on Termux:

```sh
pkg install python
curl -L https://raw.githubusercontent.com/CrazeStudio/CrazeAI-Cloud/main/crazeai.py -o crazeai.py
python crazeai.py
```

Optional environment variables:

```sh
export CRAZEAI_URL=https://crazeai-cloud.onrender.com/v1/chat/completions
export CRAZEAI_API_KEY='your-api-key'
python crazeai.py
```

Local memory commands:
- `/remember TEXT` — save a note explicitly.
- `/memory` — show recent saved memory entries.
- `/forget` — delete local memory.
- `/clear` — archive the current chat and clear active history.
- `exit` — quit.

Memory is stored in `~/.crazeai_memory.jsonl` on the device, not on Render. Old conversation excerpts are treated as unverified context, not guaranteed facts.

## Render deployment

Deploy this repository as a Docker Web Service using `render.yaml`. Keep the existing `API_KEY` environment variable configured in Render. The entrypoint forwards it to llama.cpp as `LLAMA_API_KEY`.

The server defaults to temperature 0.2, top-p 0.85, top-k 20 and repeat penalty 1.15. The client also sends these settings explicitly. They reduce randomness and repetition but do not eliminate model hallucinations.

The service uses Render's ephemeral filesystem, so a restart or redeploy may download the model again. Render free instances can sleep when idle. The 1,024-token context and single request slot are intentional memory-saving settings; the client manages history rather than increasing server RAM requirements.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `API_KEY` | required | Bearer secret; never expose in source code |
| `LLAMA_ARG_CTX_SIZE` | 1024 | Maximum context allocation |
| `LLAMA_ARG_N_PARALLEL` | 1 | One active request slot |
| `LLAMA_ARG_THREADS` | 1 | CPU inference threads |
| `LLAMA_ARG_BATCH` | 64 | Prompt batch size |
| `LLAMA_ARG_UBATCH` | 32 | Physical micro-batch size |
| `LLAMA_ARG_TEMP` | 0.2 | Default sampling temperature |
| `LLAMA_ARG_TOP_P` | 0.85 | Nucleus sampling |
| `LLAMA_ARG_TOP_K` | 20 | Top-k sampling |
| `LLAMA_ARG_REPEAT_PENALTY` | 1.15 | Repetition penalty |
| `LLAMA_ARG_REPEAT_LAST_N` | 128 | Repetition lookback window |

The previous Gradle/Jlama Java source remains in the repository for reference; Docker runs llama.cpp.


### Faster startup

The Qwen GGUF is downloaded during the Docker image build and copied into the final image. The running service loads `/models/Qwen2.5-0.5B-Instruct-Q3_K_M.gguf` locally, so it does not need to download the model again whenever the free Render instance wakes or restarts. This shifts the one-time model download into the image-build step; it does not prevent Render free instances from sleeping, and model initialization still takes time.
