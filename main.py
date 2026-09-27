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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

MODEL_URL = os.getenv("MODEL_URL", "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf?download=true")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/Qwen3-1.7B-Q8_0.gguf"))
LLAMA_DIR = Path(os.getenv("LLAMA_DIR", "llama"))
LLAMA_SERVER = LLAMA_DIR / "llama-server"
LLAMA_ARCHIVE = LLAMA_DIR / "llama-server.tar.gz"
LLAMA_PORT = int(os.getenv("LLAMA_PORT", "8081"))
IDLE_SECONDS = int(os.getenv("MODEL_IDLE_SECONDS", "300"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
N_CTX = int(os.getenv("N_CTX", "1024"))
N_THREADS = int(os.getenv("N_THREADS", str(max(1, (os.cpu_count() or 2) - 1))))

app = FastAPI(title="CrazeAI Cloud", version="2.5.0")
_process: Optional[subprocess.Popen] = None
_process_lock = threading.Lock()
_download_lock = threading.Lock()
_unload_timer: Optional[threading.Timer] = None
_last_used = 0.0

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)

class ChatResponse(BaseModel):
    response: str

def download_file(url: str, destination: Path, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")
    started = time.monotonic()
    last_report = -1
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", "0"))
        downloaded = 0
        with open(temp, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = int(downloaded * 100 / total)
                    if percent >= last_report + 5 or percent == 100:
                        elapsed = max(time.monotonic() - started, 0.001)
                        speed = downloaded / elapsed / 1024 / 1024
                        print(f"[CrazeAI] {label}: {percent}% ({downloaded / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB) {speed:.1f} MB/s", flush=True)
                        last_report = percent
    temp.replace(destination)
    elapsed = max(time.monotonic() - started, 0.001)
    print(f"[CrazeAI] {label}: download complete in {elapsed:.1f}s", flush=True)

def ensure_model_downloaded() -> None:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return
    with _download_lock:
        if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
            return
        print("[CrazeAI] First chat request: downloading GGUF...", flush=True)
        download_file(MODEL_URL, MODEL_PATH, "GGUF")
        print("[CrazeAI] GGUF ready.", flush=True)

def get_llama_release_url() -> str:
    api = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        r = client.get(api, headers={"Accept": "application/vnd.github+json", "User-Agent": "CrazeAI-Cloud"})
        r.raise_for_status()
        release = r.json()
        assets = release.get("assets", [])
        for asset in assets:
            if asset.get("name") == "nightly-tag.txt":
                tag_url = asset.get("browser_download_url")
                if tag_url:
                    tag = client.get(tag_url).text.strip()
                    if tag:
                        nightly = client.get(f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}")
                        nightly.raise_for_status()
                        assets = nightly.json().get("assets", [])
                        break
    for asset in assets:
        name = asset.get("name", "")
        if name.startswith("llama-") and name.endswith("-bin-ubuntu-x64.tar.gz"):
            return asset["browser_download_url"]
    return "https://github.com/ggml-org/llama.cpp/releases/download/b11146/llama-b11146-bin-ubuntu-x64.tar.gz"

def ensure_llama_server() -> None:
    if LLAMA_SERVER.exists():
        LLAMA_SERVER.chmod(0o755)
        return
    with _download_lock:
        if LLAMA_SERVER.exists():
            return
        LLAMA_DIR.mkdir(parents=True, exist_ok=True)
        url = get_llama_release_url()
        print("[CrazeAI] First chat request: downloading llama.cpp...", flush=True)
        download_file(url, LLAMA_ARCHIVE, "llama.cpp")
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
            return client.get(f"http://127.0.0.1:{LLAMA_PORT}/health").status_code == 200
    except Exception:
        return False

def schedule_unload() -> None:
    global _unload_timer
    if _unload_timer is not None:
        _unload_timer.cancel()
    _unload_timer = threading.Timer(IDLE_SECONDS, stop_model)
    _unload_timer.daemon = True
    _unload_timer.start()

def stop_model() -> None:
    global _process
    with _process_lock:
        if _process is None:
            return
        if time.monotonic() - _last_used < IDLE_SECONDS:
            schedule_unload()
            return
        print("[CrazeAI] Idle timeout: unloading model...", flush=True)
        _process.terminate()
        try:
            _process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _process.kill()
            _process.wait(timeout=5)
        _process = None
        print("[CrazeAI] Model unloaded; RAM released.", flush=True)

def start_model() -> None:
    global _process, _last_used
    with _process_lock:
        if _process is not None and _process.poll() is None:
            _last_used = time.monotonic()
            schedule_unload()
            return
        ensure_model_downloaded()
        ensure_llama_server()
        print(f"[CrazeAI] Starting llama-server: model={MODEL_PATH} ctx={N_CTX} threads={N_THREADS}", flush=True)
        log_path = LLAMA_DIR / "llama-server.log"
        log_file = open(log_path, "a", buffering=1)
        _process = subprocess.Popen([
            str(LLAMA_SERVER), "-m", str(MODEL_PATH),
            "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
            "-c", str(N_CTX), "-t", str(N_THREADS), "-b", "64", "--no-webui"
        ], stdout=log_file, stderr=subprocess.STDOUT, text=True)
        _last_used = time.monotonic()
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if _process is not None and _process.poll() is not None:
            try:
                details = log_path.read_text(errors="replace")[-6000:]
            except Exception:
                details = "(unable to read llama-server.log)"
            raise RuntimeError("llama-server exited while loading the model. Last llama-server log:\n" + details)
        if server_ready():
            _last_used = time.monotonic()
            schedule_unload()
            return
        time.sleep(1)
    raise RuntimeError("Timed out while loading the GGUF model. Check llama-server.log in the Render filesystem.")

def generate(message: str) -> str:
    start_model()
    global _last_used
    payload = {"messages": [{"role": "system", "content": "You are CrazeAI, a helpful coding-focused AI assistant. Give accurate, concise answers."}, {"role": "user", "content": message}], "temperature": 0.2, "top_p": 0.9, "max_tokens": MAX_TOKENS, "stream": False}
    with httpx.Client(timeout=300) as client:
        r = client.post(f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    _last_used = time.monotonic()
    schedule_unload()
    return data["choices"][0]["message"]["content"].strip()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'><title>CrazeAI Cloud</title><style>body{font-family:Arial,sans-serif;background:#0b0b0b;color:#fff;display:grid;place-items:center;min-height:100vh;margin:0}main{max-width:620px;padding:32px;text-align:center}h1{color:#ff8a00}code{background:#171717;padding:4px 7px;border-radius:6px}</style></head><body><main><h1>CrazeAI Cloud</h1><p>API is online.</p><p>Model uses lazy loading and is only downloaded/loaded when <code>/api/chat</code> is called.</p><p><a href='/health' style='color:#ff8a00'>Health status</a></p></main></body></html>"""

@app.get("/health")
async def health():
    running = _process is not None and _process.poll() is None
    return {"status": "ok", "service": "online", "model_downloaded": MODEL_PATH.exists(), "model_loaded": running and server_ready(), "lazy_loading": True, "idle_seconds": IDLE_SECONDS}

async def handle_chat(req: ChatRequest):
    try:
        return ChatResponse(response=await asyncio.to_thread(generate, req.message))
    except Exception as exc:
        print(f"[CrazeAI] Request failed: {exc}", flush=True)
        raise HTTPException(status_code=503, detail=f"CrazeAI unavailable: {exc}") from exc

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    return await handle_chat(req)

@app.post("/chat", response_model=ChatResponse)
async def chat_alias(req: ChatRequest):
    return await handle_chat(req)

@app.post("/api/unload")
async def unload():
    global _last_used
    _last_used = 0
    stop_model()
    return {"status": "ok", "model_loaded": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
