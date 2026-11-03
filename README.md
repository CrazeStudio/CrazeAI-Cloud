# CrazeAI Cloud

Java 21 OpenAI-compatible inference API powered by Jlama. No llama.cpp or Python inference process is used.

## Model format compatibility

Jlama loads Hugging Face Transformers/SafeTensors model directories and supports its own Q4/Q8 quantization path. It does not directly load GGUF Q2_K files. The default model is HuggingFaceTB/SmolLM2-360M-Instruct; the service asks Jlama to quantize model weights to Q4 while loading. If the model configuration or tokenizer template is not accepted by this Jlama release, startup fails rather than silently switching engines.

## API endpoints

- GET /health — public Render health check
- GET /v1/models — authenticated model list
- POST /v1/chat/completions — OpenAI-compatible chat completion, including stream=true SSE

All /v1 routes require Authorization: Bearer YOUR_API_KEY.

Example request body:

    {"model":"HuggingFaceTB/SmolLM2-360M-Instruct","messages":[{"role":"user","content":"Hello"}],"max_tokens":96}

## Render deployment

1. Create a Blueprint in Render from this repository and apply render.yaml.
2. Set API_KEY to a long random secret in Render environment variables. The service refuses to start without it.
3. The model downloads on first boot into MODEL_DIR. HF_TOKEN is optional for gated/private models.

Render free filesystems are ephemeral, so restarts/redeploys may download the model again. Free instances also have tight memory limits: loading and quantizing a 360M BF16 checkpoint may exceed free-instance RAM. A paid instance with sufficient memory or a pre-quantized Jlama-compatible checkpoint is likely required for reliable inference. The 10–30 MB target applies to application overhead, not model weights or inference working memory.

## Local run

Requires JDK 21 and Gradle 8. Run: export API_KEY='replace-with-a-long-random-secret' then gradle run.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| PORT | 8080 | HTTP port |
| API_KEY | required | Bearer authentication secret |
| MODEL_ID | HuggingFaceTB/SmolLM2-360M-Instruct | Hugging Face model ID |
| MODEL_DIR | ./models | Model download/cache directory |
| MODEL_QUANTIZATION | Q4 | Jlama weight quantization (Q4 or Q8) |
| MAX_TOKENS | 256 | Hard cap per completion |
| HF_TOKEN | unset | Optional Hugging Face token |

Jlama requires Java 21 preview/vector JVM flags; the Gradle and Docker configurations supply them.
