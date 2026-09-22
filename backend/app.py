import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="CrazeAI Cloud", version="1.0.0")

LLAMA_URL = f"http://127.0.0.1:{os.getenv('LLAMA_PORT', '8081')}"
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)

class ChatResponse(BaseModel):
    response: str

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{LLAMA_URL}/health")
            if r.status_code == 200:
                return {"status": "ok", "model": "ready"}
    except Exception:
        pass
    return {"status": "starting", "model": "loading"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are CrazeAI, a helpful coding-focused AI assistant. Give accurate, concise answers and put code in fenced code blocks."
            },
            {"role": "user", "content": req.message},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(f"{LLAMA_URL}/v1/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
            answer = data["choices"][0]["message"]["content"].strip()
            return ChatResponse(response=answer)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"CrazeAI model is not ready: {e}")
    except (KeyError, IndexError, TypeError) as e:
        raise HTTPException(status_code=502, detail=f"Invalid model response: {e}")
