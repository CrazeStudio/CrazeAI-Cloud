# Deploy CrazeAI Cloud on Voroa

Voroa currently configures Docker services from its dashboard. It does not document a repository manifest equivalent to Render's `render.yaml`, so this file is the deployment checklist; **do not rename it to `voroa.yaml` and expect Voroa to read it**.

## Create the service

1. In Voroa, create **New → Web Service** and connect this repository.
2. Select branch `main`.
3. Under Build & runtime, choose **Docker (Dockerfile)**.
4. Set the Dockerfile path to `/Dockerfile` (repository root).
5. Create the service and add the environment variable below before the first successful start.

## Service settings

| Setting | Value |
|---|---|
| Service type | Web Service |
| Runtime | Docker (Dockerfile) |
| Branch | `main` |
| Dockerfile path | `/Dockerfile` |
| Health check path | `/health` |
| Port | Read `PORT` supplied by Voroa (currently `3000`) |

The entrypoint already uses `${PORT:-${LLAMA_ARG_PORT:-8080}}`, so it binds to Voroa's injected port. Do not hard-code port 8080 in the Voroa service.

## Environment variables

| Name | Value |
|---|---|
| `API_KEY` | Required; set a strong private secret in Voroa's Environment tab |
| `LLAMA_ARG_CTX_SIZE` | `2048` |
| `LLAMA_ARG_N_PARALLEL` | `2` |
| `LLAMA_ARG_THREADS` | `1` |
| `LLAMA_ARG_BATCH` | `64` |
| `LLAMA_ARG_UBATCH` | `32` |
| `LLAMA_ARG_CACHE_TYPE_K` | `q8_0` |
| `LLAMA_ARG_CACHE_TYPE_V` | `q8_0` |
| `LLAMA_ARG_TEMP` | `0.2` |
| `LLAMA_ARG_TOP_P` | `0.85` |
| `LLAMA_ARG_TOP_K` | `20` |
| `LLAMA_ARG_REPEAT_PENALTY` | `1.15` |
| `LLAMA_ARG_REPEAT_LAST_N` | `128` |
| `LLAMA_ARG_ALIAS` | `Qwen/Qwen2.5-0.5B-Instruct` |

Do not set `API_KEY` as a Docker build argument. It is a runtime secret and must stay in Voroa's encrypted environment-variable settings. The entrypoint forwards it to llama.cpp as `LLAMA_API_KEY`.

## Important resource note

The Qwen2.5 0.5B Q3_K_M model file is approximately 355 MB before runtime overhead. Voroa's 512 MB free instance may not have enough memory for reliable inference once the model and server are loaded. Start with a larger memory tier if available; otherwise test carefully and watch memory usage and restart logs. A successful image build does not prove the model fits in the running instance.

The Docker build downloads the GGUF into the image, so model download happens during image build rather than every container start. This makes the image larger and means a rebuild is needed when changing the model.

## Verify after deployment

Replace the URL and secret locally; never commit the secret:

```sh
curl https://YOUR-SERVICE.getvoroa.com/health
curl https://YOUR-SERVICE.getvoroa.com/ready
curl https://YOUR-SERVICE.getvoroa.com/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Then test `POST /v1/chat/completions` with the same bearer token. Check Voroa deploy logs for model-load completion and check the service's memory graph for OOM restarts.

## Official Voroa documentation

- [Docker deployments](https://getvoroa.com/docs/docker/)
- [Deploy from GitHub](https://getvoroa.com/docs/deploy-from-github/)
- [Environment variables](https://getvoroa.com/docs/)
