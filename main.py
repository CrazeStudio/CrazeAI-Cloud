import asyncio, os, subprocess, tarfile, threading, time
from pathlib import Path
from typing import Optional
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi import UploadFile, File
import cv2
import numpy as np
from pydantic import BaseModel, Field

MODEL_URL=os.getenv("MODEL_URL","https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf?download=true")
MODEL_PATH=Path(os.getenv("MODEL_PATH","models/Qwen3-0.6B-Q2_K.gguf")).resolve()
LLAMA_DIR=Path(os.getenv("LLAMA_DIR","llama")).resolve()
LLAMA_PORT=int(os.getenv("LLAMA_PORT","8081")); IDLE_SECONDS=int(os.getenv("MODEL_IDLE_SECONDS","300"))
MAX_TOKENS=int(os.getenv("MAX_TOKENS","192")); N_CTX=int(os.getenv("N_CTX","512")); N_THREADS=int(os.getenv("N_THREADS","2"))
app=FastAPI(title="CrazeAI Cloud",version="3.1.3")
_process:Optional[subprocess.Popen]=None; _process_lock=threading.Lock(); _download_lock=threading.Lock(); _last_used=0.0; _startup_error=None; _starting=False
class ChatRequest(BaseModel): message:str=Field(min_length=1,max_length=12000)
class ChatResponse(BaseModel): response:str

def download_file(url,destination,label):
    destination.parent.mkdir(parents=True,exist_ok=True); temp=destination.with_suffix(destination.suffix+".part"); started=time.monotonic(); last=-1
    headers={"User-Agent":"CrazeAI-Cloud"}
    if temp.exists() and temp.stat().st_size: headers["Range"]=f"bytes={temp.stat().st_size}-"
    with httpx.stream("GET",url,follow_redirects=True,timeout=None,headers=headers) as r:
        r.raise_for_status(); append=r.status_code==206 and temp.exists(); base=temp.stat().st_size if append else 0
        total=int(r.headers.get("content-length","0"))+base if append else int(r.headers.get("content-length","0")); downloaded=base; mode="ab" if append else "wb"
        with open(temp,mode) as f:
            for chunk in r.iter_bytes(1024*1024):
                if chunk:
                    f.write(chunk); downloaded+=len(chunk)
                    if total:
                        pct=min(100,int(downloaded*100/total))
                        if pct>=last+5 or pct==100: print(f"[CrazeAI] {label}: {pct}% ({downloaded/1048576:.1f} MB)",flush=True); last=pct
    temp.replace(destination); print(f"[CrazeAI] {label}: download complete in {time.monotonic()-started:.1f}s",flush=True)

def ensure_model():
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size>200_000_000: print(f"[CrazeAI] Using cached GGUF ({MODEL_PATH.stat().st_size/1048576:.1f} MB)",flush=True); return
    with _download_lock:
        if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size>200_000_000:return
        print("[CrazeAI] Downloading GGUF to disk (streaming, low RAM)...",flush=True); download_file(MODEL_URL,MODEL_PATH,"GGUF")

def _ubuntu_asset(assets):
    for a in assets:
        name=a.get("name","").lower()
        if name.endswith("-bin-ubuntu-x64.tar.gz"): return a.get("browser_download_url"),name
    return None,None

def llama_url():
    api="https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"; headers={"Accept":"application/vnd.github+json","User-Agent":"CrazeAI-Cloud"}
    with httpx.Client(timeout=30,follow_redirects=True,headers=headers) as c:
        r=c.get(api); r.raise_for_status(); release=r.json(); url,name=_ubuntu_asset(release.get("assets",[]))
        if url:return url
        tag_asset=next((a for a in release.get("assets",[]) if a.get("name")=="nightly-tag.txt"),None)
        if tag_asset:
            tr=c.get(tag_asset["browser_download_url"]); tr.raise_for_status(); nightly_tag=tr.text.strip().splitlines()[0].strip(); print(f"[CrazeAI] Resolved nightly tag: {nightly_tag}",flush=True)
            rr=c.get(f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{nightly_tag}"); rr.raise_for_status(); url,name=_ubuntu_asset(rr.json().get("assets",[]))
            if url:return url
            raise RuntimeError(f"No Ubuntu x64 llama.cpp asset found in nightly {nightly_tag}")
        raise RuntimeError(f"No Ubuntu x64 llama.cpp asset found in release {release.get('tag_name','unknown')}")

def find_server():
    xs=[p for p in LLAMA_DIR.rglob("llama-server") if p.is_file()]; return sorted(xs,key=lambda p:len(str(p)))[0].resolve() if xs else None

def ensure_server():
    x=find_server()
    if x:return x
    with _download_lock:
        x=find_server()
        if x:return x
        LLAMA_DIR.mkdir(parents=True,exist_ok=True); archive=LLAMA_DIR/"llama-server.tar.gz"; print("[CrazeAI] Downloading llama.cpp...",flush=True); download_file(llama_url(),archive,"llama.cpp")
        with tarfile.open(archive,"r:gz") as t:t.extractall(LLAMA_DIR,filter="data")
        x=find_server()
        if not x:raise RuntimeError("llama-server executable was not found after extraction")
        x.chmod(x.stat().st_mode|0o111); return x

def ready():
    try:
        with httpx.Client(timeout=2) as c:return c.get(f"http://127.0.0.1:{LLAMA_PORT}/health").status_code==200
    except:return False

def start_model():
    global _process,_last_used,_startup_error,_starting
    with _process_lock:
        if _process and _process.poll() is None:return
        if _starting:return
        _starting=True; _startup_error=None
    try:
        ensure_model(); server=ensure_server(); log=LLAMA_DIR/"llama-server.log"; lf=open(log,"a",buffering=1); env=os.environ.copy(); libs={str(server.parent),str(LLAMA_DIR)}|{str(p.parent) for p in LLAMA_DIR.rglob("*.so*")}; env["LD_LIBRARY_PATH"]=":".join(sorted(libs|({env["LD_LIBRARY_PATH"]} if env.get("LD_LIBRARY_PATH") else set())))
        print(f"[CrazeAI] Starting llama-server: {server}",flush=True)
        with _process_lock:
            _process=subprocess.Popen([str(server),"-m",str(MODEL_PATH),"--host","127.0.0.1","--port",str(LLAMA_PORT),"-c",str(N_CTX),"-t",str(N_THREADS),"-b","16","--parallel","1","--no-webui","--jinja","--reasoning","off","-ngl","0"],cwd=str(server.parent),env=env,stdout=lf,stderr=subprocess.STDOUT,text=True); _last_used=time.monotonic()
        deadline=time.monotonic()+240
        while time.monotonic()<deadline:
            if _process and _process.poll() is not None:
                details=log.read_text(errors="replace")[-10000:] if log.exists() else "(no log)"; raise RuntimeError("llama-server exited while loading the model. Last log:\n"+details)
            if ready():_last_used=time.monotonic(); print("[CrazeAI] Model loaded and llama-server is ready.",flush=True); return
            time.sleep(1)
        raise RuntimeError("Timed out while loading the GGUF model")
    except Exception as e:
        _startup_error=str(e)
        with _process_lock:
            if _process and _process.poll() is not None:_process=None
        raise
    finally:
        _starting=False

def generate(msg):
    start_model()
    payload={"messages":[{"role":"system","content":"You are CrazeAI, a helpful coding-focused AI assistant. Answer directly and naturally. Do not output token IDs, token numbers, debug fields, or repeated characters."},{"role":"user","content":msg}],"temperature":0.2,"top_p":0.9,"max_tokens":MAX_TOKENS,"stream":False}
    with httpx.Client(timeout=300) as c:
        r=c.post(f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions",json=payload)
        r.raise_for_status()
        data=r.json()
    global _last_used
    _last_used=time.monotonic()
    choices=data.get("choices") if isinstance(data,dict) else None
    if not choices or not isinstance(choices,list):
        raise RuntimeError("llama-server returned no choices")
    message=choices[0].get("message") if isinstance(choices[0],dict) else None
    if not isinstance(message,dict):
        raise RuntimeError("llama-server returned an invalid message")
    answer=message.get("content")
    if not isinstance(answer,str) or not answer.strip():
        answer=message.get("reasoning_content")
    if not isinstance(answer,str) or not answer.strip():
        raise RuntimeError("llama-server returned empty content")
    return answer.strip()

@app.on_event("startup")
async def boot():
    async def worker():
        try: print("[CrazeAI] Startup: preparing model...",flush=True); await asyncio.to_thread(start_model); print("[CrazeAI] Startup: model ready.",flush=True)
        except Exception as e: print(f"[CrazeAI] Startup failed: {e}",flush=True)
    asyncio.create_task(worker())

@app.get("/",response_class=HTMLResponse)
async def root():return "<html><body style='background:#0b0b0b;color:white;font-family:Arial;text-align:center;padding:50px'><h1 style='color:#ff8a00'>CrazeAI Cloud</h1><p>Model startup in background. Check /health.</p></body></html>"
@app.get("/health")
async def health():
    running=_process is not None and _process.poll() is None
    return {"status":"ok","service":"online","model":"Qwen3-0.6B-Q2_K","model_downloaded":MODEL_PATH.exists(),"model_loaded":running and ready(),"starting":_starting,"startup_error":_startup_error}
async def handle(req):
    try:return ChatResponse(response=await asyncio.to_thread(generate,req.message))
    except Exception as e:print(f"[CrazeAI] Request failed: {e}",flush=True);raise HTTPException(503,detail=f"CrazeAI unavailable: {e}")
@app.post("/chat",response_model=ChatResponse)
async def chat(req:ChatRequest):return await handle(req)
@app.post("/api/chat",response_model=ChatResponse)
async def api_chat(req:ChatRequest):return await handle(req)

@app.post("/enhance-document")
@app.post("/api/enhance-document")
async def enhance_document(file: UploadFile = File(...), mode: str = "clean"):
    """Restore a user-provided scan without generating or changing document content."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(415, detail="Upload an image file (PNG, JPEG, or WebP).")
    raw = await file.read()
    if not raw or len(raw) > 12 * 1024 * 1024:
        raise HTTPException(413, detail="Image must be non-empty and no larger than 12 MB.")
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, detail="The uploaded image could not be decoded.")
    h, w = image.shape[:2]
    if h * w > 20_000_000:
        raise HTTPException(413, detail="Image dimensions are too large (maximum 20 megapixels).")
    # Conservative, non-generative cleanup: denoise, improve local contrast, and upscale.
    # Never synthesize text, numbers, signatures, stamps, or account details.
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
    if mode == "grayscale":
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    else:
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
        result = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    if max(h, w) < 3000:
        result = cv2.resize(result, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    ok, encoded = cv2.imencode(".png", result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise HTTPException(500, detail="Could not encode the enhanced image.")
    return Response(content=encoded.tobytes(), media_type="image/png",
                    headers={"Content-Disposition": 'inline; filename="enhanced-document.png"',
                             "X-Enhancement-Method": "non-generative-restoration"})

@app.post("/api/unload")
async def unload():
    global _last_used; _last_used=0; return {"status":"ok","model_loaded":False}
if __name__=="__main__":
    import uvicorn; uvicorn.run(app,host="0.0.0.0",port=int(os.getenv("PORT","10000")))
