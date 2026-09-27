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
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/Qwen3-1.7B-Q8_0.gguf")).resolve()
LLAMA_DIR = Path(os.getenv("LLAMA_DIR", "llama")).resolve()
LLAMA_PORT = int(os.getenv("LLAMA_PORT", "8081"))
IDLE_SECONDS = int(os.getenv("MODEL_IDLE_SECONDS", "300"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
N_CTX = int(os.getenv("N_CTX", "1024"))
N_THREADS = int(os.getenv("N_THREADS", str(max(1, (os.cpu_count() or 2) - 1))))

app = FastAPI(title="CrazeAI Cloud", version="2.7.1")
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
    with httpx.stream("GET", url, follow_redirects=True, timeout=None, headers={"User-Agent": "CrazeAI-Cloud"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", "0"))
        downloaded = 0
        with open(temp, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        percent = int(downloaded * 100 / total)
                        if percent >= last_report + 5 or percent == 100:
                            elapsed = max(time.monotonic() - started, 0.001)
                            speed = downloaded / elapsed / 1024 / 1024
                            print(f"[CrazeAI] {label}: {percent}% ({downloaded/1048576:.1f}/{total/1048576:.1f} MB) {speed:.1f} MB/s", flush=True)
                            last_report = percent
    temp.replace(destination)
    print(f"[CrazeAI] {label}: download complete in {time.monotonic()-started:.1f}s", flush=True)

def ensure_model_downloaded():
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 100_000_000:
        print(f"[CrazeAI] Using cached GGUF: {MODEL_PATH}", flush=True)
        return
    with _download_lock:
        if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 100_000_000:
            return
        print("[CrazeAI] No cached GGUF found; downloading...", flush=True)
        download_file(MODEL_URL, MODEL_PATH, "GGUF")

def llama_asset_url():
    api = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    with httpx.Client(timeout=30, follow_redirects=True, headers={"Accept":"application/vnd.github+json","User-Agent":"CrazeAI-Cloud"}) as c:
        response = c.get(api)
        response.raise_for_status()
        release = response.json()
        assets = release.get("assets", [])
        tag_file = next((a for a in assets if a.get("name") == "nightly-tag.txt"), None)
        if tag_file:
            tag_response = c.get(tag_file["browser_download_url"])
            tag_response.raise_for_status()
            tag = tag_response.text.strip()
            if tag:
                rr = c.get(f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}")
                rr.raise_for_status()
                assets = rr.json().get("assets", [])
        for a in assets:
            name = a.get("name", "")
            if name.endswith("-bin-ubuntu-x64.tar.gz"):
                return a["browser_download_url"]
    raise RuntimeError("No Ubuntu x64 llama.cpp release asset was found")

def find_server():
    candidates = [p for p in LLAMA_DIR.rglob("llama-server") if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.stat().st_size == 0, len(str(p))))
    return candidates[0].resolve()

def ensure_llama_server():
    existing = find_server()
    if existing:
        existing.chmod(existing.stat().st_mode | 0o111)
        return existing
    with _download_lock:
        existing = find_server()
        if existing:
            return existing
        LLAMA_DIR.mkdir(parents=True, exist_ok=True)
        archive = LLAMA_DIR / "llama-server.tar.gz"
        print("[CrazeAI] No cached llama-server found; downloading...", flush=True)
        download_file(llama_asset_url(), archive, "llama.cpp")
        print("[CrazeAI] Extracting llama.cpp...", flush=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(LLAMA_DIR, filter="data")
        server = find_server()
        if not server:
            raise RuntimeError("llama-server executable was not found after extraction")
        server.chmod(server.stat().st_mode | 0o111)
        print(f"[CrazeAI] llama-server found: {server}", flush=True)
        return server

def server_ready():
    try:
        with httpx.Client(timeout=2) as c:
            return c.get(f"http://127.0.0.1:{LLAMA_PORT}/health").status_code == 200
    except Exception:
        return False

def schedule_unload():
    global _unload_timer
    if _unload_timer: _unload_timer.cancel()
    _unload_timer = threading.Timer(IDLE_SECONDS, stop_model)
    _unload_timer.daemon = True
    _unload_timer.start()

def stop_model():
    global _process
    with _process_lock:
        if _process is None: return
        if time.monotonic() - _last_used < IDLE_SECONDS:
            schedule_unload(); return
        print("[CrazeAI] Idle timeout: unloading model...", flush=True)
        _process.terminate()
        try: _process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _process.kill(); _process.wait(timeout=5)
        _process = None
        print("[CrazeAI] Model unloaded; RAM released.", flush=True)

def start_model():
    global _process, _last_used
    with _process_lock:
        if _process is not None and _process.poll() is None:
            _last_used = time.monotonic(); schedule_unload(); return
        ensure_model_downloaded()
        server = ensure_llama_server()
        log_path = LLAMA_DIR / "llama-server.log"
        log_file = open(log_path, "a", buffering=1)
        env = os.environ.copy()
        lib_dirs = {str(server.parent), str(LLAMA_DIR)}
        for p in LLAMA_DIR.rglob("*.so*"):
            lib_dirs.add(str(p.parent))
        if env.get("LD_LIBRARY_PATH"):
            lib_dirs.add(env["LD_LIBRARY_PATH"])
        env["LD_LIBRARY_PATH"] = ":".join(sorted(lib_dirs))
        print(f"[CrazeAI] Starting {server} with {MODEL_PATH}", flush=True)
        _process = subprocess.Popen([str(server), "-m", str(MODEL_PATH), "--host", "127.0.0.1", "--port", str(LLAMA_PORT), "-c", str(N_CTX), "-t", str(N_THREADS), "-b", "64", "--no-webui"], cwd=str(server.parent), env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        _last_used = time.monotonic()
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if _process is not None and _process.poll() is not None:
            details = log_path.read_text(errors="replace")[-8000:] if log_path.exists() else "(no llama-server log)"
            raise RuntimeError("llama-server exited while loading the model. Last llama-server log:\n" + details)
        if server_ready():
            _last_used = time.monotonic(); schedule_unload(); return
        time.sleep(1)
    raise RuntimeError("Timed out while loading the GGUF model")

def generate(message: str):
    start_model()
    global _last_used
    payload = {"messages":[{"role":"system","content":"You are CrazeAI, a helpful coding-focused AI assistant. Give accurate, concise answers."},{"role":"user","content":message}],"temperature":0.2,"top_p":0.9,"max_tokens":MAX_TOKENS,"stream":False}
    with httpx.Client(timeout=300) as c:
        r=c.post(f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions",json=payload)
        r.raise_for_status()
        data=r.json()
    _last_used=time.monotonic(); schedule_unload()
    return data["choices"][0]["message"]["content"].strip()

@app.get("/", response_class=HTMLResponse)
async def root():
    return "<html><body style='background:#0b0b0b;color:white;font-family:Arial;text-align:center;padding:50px'><h1 style='color:#ff8a00'>CrazeAI Cloud</h1><p>API online — lazy model loading enabled.</p><p><a href='/health' style='color:#ff8a00'>Health</a></p></body></html>"

@app.get("/health")
async def health():
    running=_process is not None and _process.poll() is None
    return {"status":"ok","service":"online","model_downloaded":MODEL_PATH.exists(),"model_loaded":running and server_ready(),"lazy_loading":True,"idle_seconds":IDLE_SECONDS}

async def handle_chat(req: ChatRequest):
    try: return ChatResponse(response=await asyncio.to_thread(generate,req.message))
    except Exception as exc:
        print(f"[CrazeAI] Request failed: {exc}",flush=True)
        raise HTTPException(status_code=503,detail=f"CrazeAI unavailable: {exc}") from exc

@app.post("/chat",response_model=ChatResponse)
async def chat(req: ChatRequest): return await handle_chat(req)

@app.post("/api/chat",response_model=ChatResponse)
async def api_chat(req: ChatRequest): return await handle_chat(req)

@app.post("/api/unload")
async def unload():
    global _last_used
    _last_used=0; stop_model(); return {"status":"ok","model_loaded":False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=int(os.getenv("PORT","10000")))
