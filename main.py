import asyncio
import os
import subprocess
import tarfile
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
LLAMA_DIR = Path(os.getenv("LLAMA_DIR", "llama"))
LLAMA_SERVER = LLAMA_DIR / "llama-server"
LLAMA_VERSION = os.getenv("LLAMA_VERSION", "b10964")
LLAMA_ARCHIVE = LLAMA_DIR / "llama-server.tar.gz"
LLAMA_PORT = int(os.getenv("LLAMA_PORT", "8081"))
IDLE_SECONDS = int(os.getenv("MODEL_IDLE_SECONDS", "300"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
N_CTX = int(os.getenv("N_CTX", "1024"))
N_THREADS = int(os.getenv("N_THREADS", str(max(1, (os.cpu_count() or 2) - 1))))

app = FastAPI(title="CrazeAI Cloud", version="2.1.0")

_process: Optional[subprocess.Popen] = None
_process_lock = threading.Lock()
_download_lock = threading.Lock()
_unload_timer: Optional[threading.Timer] = None
_last_used = 0.0


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)


class ChatResponse(BaseModel):
    response: str


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        with open(temp, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    temp.replace(destination)


def ensure_model_downloaded() -> None:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return
    with _download_lock:
        if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
            return
        print("[CrazeAI] Downloading GGUF model...", flush=True)
        download_file(MODEL_URL, MODEL_PATH)
        print("[CrazeAI] GGUF download complete.", flush=True)


def ensure_llama_server() -> None:
    if LLAMA_SERVER.exists():
        LLAMA_SERVER.chmod(0o755)
        return

    with _download_lock:
        if LLAMA_SERVER.exists():
            return
        LLAMA_DIR.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://github.com/ggml-org/llama.cpp/releases/download/"
            f"v0.4.1/llama-v0.4.1-bin-ubuntu-x64.tar.gz"
        )
        print("[CrazeAI] Downloading prebuilt llama.cpp server...", flush=True)
        download_file(url, LLAMA_ARCHIVE)
        with tarfile.open(LLAMA_ARCHIVE, "r:gz") as tar:
            tar.extractall(LLAMA_DIR, filter="data")

        candidates = list(LLAMA_DIR.rglob("llama-server"))
        if not candidates:
            raise RuntimeError("llama-server binary was not found in the release archive")
        candidates[0].chmod(0o755)
        if candidates[0] != LLAMA_SERVER:
            candidates[0].replace(LLAMA_SERVER)
        print("[CrazeAI] llama-server ready.", flush=True)


def server_ready() -> bool:
    try:
        with httpx.Client(timeout=2) as client:
            r = client.get(f"http://127.0.0.1:{LLAMA_PORT}/health")
            return r.status_code == 200
    except Exception:
        return False


def stop_model() -> None:
    global _process
    with _process_lock:
        if _process is None:
            return
        if time.monotonic() - _last_used < IDLE_SECONDS:
            schedule_unload()
            return
        print("[CrazeAI] Idle timeout; stopping llama-server and freeing model RAM...", flush=True)
        _process.terminate()
        try:
            _process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _process.kill()
            _process.wait(timeout=5)
        _process = None
        print("[CrazeAI] Model unloaded.", flush=True)


def schedule_unload() -> None:
    global _unload_timer
    if _unload_timer is not None:
        _unload_timer.cancel()
    _unload_timer = threading.Timer(IDLE_SECONDS, stop_model)
    _unload_timer.daemon = True
    _unload_timer.start()


def start_model() -> None:
    global _process, _last_used
    with _process_lock:
        _last_used = time.monotonic()
        if _process is not None and _process.poll() is None:
            schedule_unload()
            return

        ensure_model_downloaded()
        ensure_llama_server()

        print("[CrazeAI] Starting llama-server and loading GGUF into RAM...", flush=True)
        _process = subprocess.Popen([
            str(LLAMA_SERVER),
            "-m", str(MODEL_PATH),
            "--host", "127.0.0.1",
            "--port", str(LLAMA_PORT),
            "-c", str(N_CTX),
            "-t", str(N_THREADS),
            "-b", "128",
            "--no-webui",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if _process is not None and _process.poll() is not None:
            raise RuntimeError("llama-server exited while loading the model")
        if server_ready():
            _last_used = time.monotonic()
            schedule_unload()
            print("[CrazeAI] Model ready.", flush=True)
            return
        time.sleep(1)
    raise RuntimeError("Timed out while loading the GGUF model")


def generate(message: str) -> str:
    start_model()
    global _last_used
    _last_used = time.monotonic()

    payload = {
        "messages": [
            {"role": "system", "content": "You are CrazeAI, a helpful coding-focused AI assistant. Give accurate, concise answers."},
            {"role": "user", "content": message},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    try:
        with httpx.Client(timeout=180) as client:
            r = client.post(f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
            _last_used = time.monotonic()
            schedule_unload()
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        schedule_unload()
        raise


@app.get("/")
async def root():
    return {"name": "CrazeAI Cloud", "status": "online"}


@app.get("/health")
async def health():
    running = _process is not None and _process.poll() is None
    return {
        "status": "ok",
        "model_downloaded": MODEL_PATH.exists(),
        "model_loaded": running and server_ready(),
        "lazy_loading": True,
        "idle_seconds": IDLE_SECONDS,
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
    global _last_used
    _last_used = 0
    stop_model()
    return {"status": "ok", "model_loaded": False}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
