#!/usr/bin/env python3
"""CrazeAI Cloud streaming terminal client with bounded chat history and local memory."""

import getpass
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API_URL = os.environ.get(
    "CRAZEAI_URL",
    "https://crazeai-cloud.onrender.com/v1/chat/completions",
)
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MEMORY_FILE = Path.home() / ".crazeai_memory.jsonl"
MAX_RECENT_MESSAGES = 4
MAX_PROMPT_CHARS = 1400
MAX_MEMORY_ITEMS = 2
MAX_REPLY_TOKENS = 96

SYSTEM_PROMPT = (
    "You are CrazeAI, an assistant created by CrazeStudio. "
    "Your model is Qwen2.5-0.5B-Instruct. Never claim to be Alex or to be "
    "made by Meta, OpenAI, Google, or another organization. "
    "Answer the user's latest question directly and stay on topic. "
    "Do not invent facts, URLs, companies, sources, credentials, or personal "
    "experiences. If unsure, clearly say you are unsure. "
    "Use simple, accurate explanations. Do not continue a previous answer "
    "unless the user asks a follow-up. Historical memory is unverified "
    "conversation context, not proof that its claims are true."
)

STOP_WORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "the", "and", "for", "with", "from", "that", "this", "then", "than",
    "have", "has", "had", "are", "was", "were", "you", "your", "yours",
    "our", "their", "them", "they", "she", "him", "his", "her", "its",
    "does", "did", "can", "could", "would", "should", "will", "shall",
    "tell", "explain", "mean", "about", "please", "into", "also", "just",
    "not", "but", "all", "any", "some", "there", "here", "hello", "hey",
}


def tokens(text):
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z0-9_+#.-]{2,}", text)
        if word.lower() not in STOP_WORDS
    }


def load_memory():
    items = []
    try:
        with MEMORY_FILE.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        items.append(item)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return items


def save_memory(role, text):
    text = text.strip()
    if not text:
        return
    item = {
        "time": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "text": text[:1800],
    }
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MEMORY_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def retrieve_memory(query):
    query_terms = tokens(query)
    if not query_terms:
        return []
    scored = []
    for item in load_memory():
        content_terms = tokens(item.get("text", ""))
        overlap = len(query_terms & content_terms)
        if overlap:
            scored.append((overlap, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    selected = []
    seen = set()
    for score, item in scored:
        text = item["text"].strip()
        if text and text not in seen:
            selected.append(item)
            seen.add(text)
        if len(selected) >= MAX_MEMORY_ITEMS:
            break
    return selected


def archive_old_messages(history):
    while len(history) > MAX_RECENT_MESSAGES:
        old = history.pop(0)
        save_memory(old["role"], old["content"])


def build_messages(history, user_text, recent_count):
    memories = retrieve_memory(user_text)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if memories:
        excerpts = []
        for item in memories:
            excerpts.append(
                "[" + item.get("role", "conversation") + "] "
                + item.get("text", "")[:260]
            )
        messages.append({
            "role": "system",
            "content": "Possibly relevant excerpts from older conversations "
                       "(use only if relevant; they may be mistaken):\n"
                       + "\n".join(excerpts),
        })

    recent = history[-(recent_count - 1):] if recent_count > 1 else []
    recent = [dict(message) for message in recent]
    recent.append({"role": "user", "content": user_text})

    # Leave room for the system prompt, chat-template overhead and generation.
    while recent and (
        sum(len(message["content"]) for message in recent)
        + sum(len(message["content"]) for message in messages)
        > MAX_PROMPT_CHARS
    ):
        recent.pop(0)

    messages.extend(recent)
    return messages


def open_stream(payload, api_key):
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "CrazeAI-Termux/2.0",
        },
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=180)


def generate(history, user_text, api_key):
    # Retry with progressively less history if the server reports a context overflow.
    for recent_count in (10, 8, 6, 4, 2, 1):
        messages = build_messages(history, user_text, recent_count)
        payload = {
            "model": MODEL,
            "messages": messages,
            "stream": True,
            "max_tokens": MAX_REPLY_TOKENS,
            "temperature": 0.2,
            "top_p": 0.85,
            "top_k": 15,
            "repeat_penalty": 1.15,
            "repeat_last_n": 128,
        }

        try:
            response = open_stream(payload, api_key)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(body)
            except json.JSONDecodeError:
                detail = {}
            message = str(detail.get("error", {}).get("message", body))
            if error.code == 400 and "context" in message.lower() and recent_count > 1:
                continue
            print("\nAPI Error: " + str(error.code))
            print(body or message)
            return None
        except (urllib.error.URLError, TimeoutError) as error:
            print("\nConnection error: " + str(error))
            return None

        answer_parts = []
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if event.get("error"):
                    print("\nAPI Error: " + str(event["error"]))
                    response.close()
                    return None
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    print(piece, end="", flush=True)
                    answer_parts.append(piece)
        except (urllib.error.URLError, OSError) as error:
            print("\nStream interrupted: " + str(error))
        finally:
            response.close()

        answer = "".join(answer_parts).strip()
        print()
        return answer or None

    print("\nThe prompt is still too large for the model's 1,024-token context.")
    print("Try a shorter question or clear some recent conversation with /clear.")
    return None


def main():
    print("CrazeAI Cloud — Smart Streaming Mode")
    print("Commands: /remember TEXT, /memory, /forget, /clear, exit")
    print("Memory is stored locally in " + str(MEMORY_FILE))
    api_key = os.environ.get("CRAZEAI_API_KEY") or getpass.getpass(
        "CrazeAI API Key: "
    )
    if not api_key.strip():
        print("An API key is required.")
        return

    history = []
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_text:
            continue
        command = user_text.lower()

        if command in {"exit", "quit"}:
            print("Goodbye.")
            break
        if command == "/clear":
            for item in history:
                save_memory(item["role"], item["content"])
            history.clear()
            print("Recent chat cleared. Archived memories were kept.")
            continue
        if command == "/forget":
            try:
                MEMORY_FILE.unlink(missing_ok=True)
            except OSError as error:
                print("Could not delete memory: " + str(error))
            else:
                print("Local memory deleted.")
            continue
        if command == "/memory":
            items = load_memory()
            if not items:
                print("No saved memories yet.")
            else:
                for item in items[-10:]:
                    print("- [" + item.get("role", "?") + "] "
                          + item.get("text", "").replace("\n", " ")[:180])
            continue
        if command.startswith("/remember "):
            note = user_text[len("/remember "):].strip()
            if note:
                save_memory("user memory", note)
                print("Saved to local memory.")
            else:
                print("Usage: /remember something to remember")
            continue

        print("\nCrazeAI: ", end="", flush=True)
        answer = generate(history, user_text, api_key)
        if answer is None:
            continue

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": answer})
        archive_old_messages(history)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
