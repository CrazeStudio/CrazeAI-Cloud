import asyncio
import gc
import os
import threading
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf?download=true",
)
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/Qwen3-1.7B-Q8_0.gguf"))
IDLE_SECONDS = int(os.getenv("MODEL_IDLE_SECONDS", "300"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
N_CTX = int(os.getenv("N_CTX", "1024"))
N_THREADS = int(os.getenv("N_THREADS", str(max(1, (os.cpu_count() or 2) - 1))))

app = FastAPI(title="CrazeAI Cloud", version="2.0.0")

_model = None
_model_lock = threading.Lock()
_download_lock = threading.Lock()
_unload_timer: Optional[threading.Timer] = None
_last_used = 0.0


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)


class ChatResponse(BaseModel):
    response: str


def download_model() -> None:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".part")

    with _download_lock:
        if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
            return

        print("[CrazeAI] Downloading GGUF model...", flush=True)
        with httpx.stream("GET", MODEL_URL, follow_redirects=True, timeout=None) as r:
            r.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        temp_path.replace(MODEL_PATH)
        print(f"[CrazeAI] Model downloaded: {MODEL_PATH}", flush=True)


def schedule_unload() -> None:
    global _unload_timer
    if _unload_timer is not None:
        _unload_timer.cancel()
    _unload_timer = threading.Timer(IDLE_SECONDS, unload_model_if_idle)
    _unload_timer.daemon = True
    _unload_timer.start()


def unload_model_if_idle() -> None:
    global _model
    with _model_lock:
        if _model is None:
            return
        if time.monotonic() - _last_used < IDLE_SECONDS:
            schedule_unload()
            return

        print("[CrazeAI] Idle timeout reached; unloading model from RAM...", flush=True)
        old_model = _model
        _model = None
        del old_model
        gc.collect()
        print("[CrazeAI] Model unloaded.", flush=True)


def get_model():
    global _model, _last_used

    with _model_lock:
        _last_used = time.monotonic()
        if _model is not None:
            schedule_unload()
            return _model

        download_model()

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("llama-cpp-python is not installed") from exc

        print("[CrazeAI] Loading GGUF into RAM...", flush=True)
        _model = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_batch=128,
            verbose=False,
        )
        print("[CrazeAI] Model ready.", flush=True)
        schedule_unload()
        return _model


def generate(message: str) -> str:
    model = get_model()
    global _last_used
    _last_used = time.monotonic()
    schedule_unload()

    result = model.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are CrazeAI, a helpful coding-focused AI assistant. Give accurate, concise answers.",
            },
            {"role": "user", "content": message},
        ],
        temperature=0.2,
        top_p=0.9,
        max_tokens=MAX_TOKENS,
    )

    _last_used = time.monotonic()
    schedule_unload()
    return result["choices"][0]["message"]["content"].strip()


@app.get("/")
async def root():
    return {"name": "CrazeAI Cloud", "status": "online"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_downloaded": MODEL_PATH.exists(),
        "model_loaded": _model is not None,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        answer = await asyncio.to_thread(generate, req.message)
        return ChatResponse(response=answer)
    except Exception as exc:
        print(f"[CrazeAI] Request failed: {exc}", flush=True)
        raise HTTPException(status_code=503, detail=f"CrazeAI unavailable: {exc}") from exc


@app.post("/api/unload")
async def unload():
    global _model
    with _model_lock:
        old_model = _model
        _model = None
        if old_model is not None:
            del old_model
            gc.collect()
    return {"status": "ok", "model_loaded": False}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
