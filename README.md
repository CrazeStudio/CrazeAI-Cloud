# CrazeAI Cloud

Cloud-hosted CrazeAI using a Qwen2.5-Coder-1.5B-Instruct GGUF model downloaded by Render at runtime.

## Architecture

- `backend/`: FastAPI gateway + llama.cpp server
- `frontend/`: minimal Android client
- Model is **not stored in Git**. Render downloads the configured GGUF file during startup.

## Model

Default:
`bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF:Q4_K_M`

The Q4_K_M file is about 986 MB according to the current Hugging Face model listing.

## API

`POST /api/chat`

Request:
```json
{"message":"Write a Python function that reverses a string."}
```

Response:
```json
{"response":"..."}
```

Health:
`GET /health`

## Render

Set the service as a Web Service and use `render.yaml`, or create it manually with the build/start commands in that file.

The Android client waits for the backend to wake from sleep by polling `/health` before sending the original message.
