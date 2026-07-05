import os
import re
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from backend.models import ChatRequest, ChatResponse, ClearMemoryRequest, SourceItem
from backend.rag_orchestrator import RagOrchestrator
from backend.local_llm import generate_local_answer
from backend.prompt import compose_language_instruction

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def get_generation_backend() -> str:
    backend = os.getenv("GENERATION_BACKEND", "groq").strip().lower()
    return backend if backend in {"groq", "openai", "local"} else "groq"


def generate_with_groq(messages: list) -> str | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        response = requests.post(
            os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 300,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]
    except Exception:
        return None


def generate_with_openai(messages: list) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        import openai

        if hasattr(openai, "OpenAI"):
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages,
            )
            return response.choices[0].message.content

        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
        )
        return response["choices"][0]["message"]["content"]
    except Exception:
        return None


def generate_answer(messages: list) -> str:
    backend = get_generation_backend()

    try:
        if backend == "groq":
            answer = generate_with_groq(messages)
            if answer:
                return answer
            return generate_local_answer(messages)

        if backend == "openai":
            answer = generate_with_openai(messages)
            if answer:
                return answer
            return generate_local_answer(messages)

        return generate_local_answer(messages)
    except Exception:
        return "I don't have enough relevant context to answer that."


def detect_query_language(text: str) -> str | None:
    hindi_chars = sum(1 for char in text if "\u0900" <= char <= "\u097f")
    latin_chars = sum(1 for char in text if char.isascii() and char.isalpha())

    if hindi_chars > 0 and hindi_chars >= latin_chars:
        return "hi"
    if latin_chars > 0:
        return "en"
    return None


app = FastAPI(title="Housing RAG API")
orch = RagOrchestrator()


@app.post("/clear-memory")
def clear_memory(req: ClearMemoryRequest):
    cleared = orch.clear_memory(req.session_id)
    return {"session_id": req.session_id, "cleared": cleared}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Retrieve and rerank
    try:
        hits = orch.retrieve_then_rerank(req.query, top_k=50)
    except Exception:
        hits = []

    # keep top N and filter out unrelated context
    top_hits = hits[: req.top_k]
    final_hits = orch.filter_relevant_hits(req.query, top_hits)
    if not final_hits:
        final_hits = top_hits

    # compose prompt and call LLM
    language = detect_query_language(req.query)
    language_instruction = compose_language_instruction(language)
    try:
        messages = orch.compose_messages(
            req.session_id,
            req.query,
            final_hits,
            system_prompt="\n\n".join(
                part for part in [req.system_prompt, language_instruction] if part
            ),
            debug=req.debug_prompt,
        )
    except Exception:
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": req.query}]

    # LLM call: use Groq by default, with local fallback available.
    llm_answer = None
    if final_hits:
        try:
            llm_answer = generate_answer(messages)
        except Exception:
            llm_answer = None

    if llm_answer and llm_answer.strip().lower() in {"null", "none", "n/a", "no answer generated."}:
        llm_answer = None

    if llm_answer:
        prompt_markers = ["Conversation history:", "Context:", "Critical Rules:", "End of Critical Rules", "You are a helpful assistant"]
        if any(marker.lower() in llm_answer.lower() for marker in prompt_markers):
            llm_answer = None

    if llm_answer is None and final_hits:
        # Provide a concise answer from the best retrieved context when the model is weak.
        best_hit = final_hits[0]
        snippet = best_hit.get("text", "") or ""
        if snippet:
            sentences = re.split(r"(?<=[.!?])\s+", snippet)
            first_sentence = next((s.strip() for s in sentences if s.strip()), "")
            if first_sentence:
                llm_answer = first_sentence[:400].strip() + ("..." if len(first_sentence) > 400 else "")
            else:
                llm_answer = snippet[:400].strip() + ("..." if len(snippet) > 400 else "")
    elif llm_answer is None and not final_hits:
        llm_answer = "I don't have enough relevant context to answer that."

    # save dialog
    try:
        orch.append_memory(req.session_id, "user", req.query)
        orch.append_memory(req.session_id, "assistant", llm_answer or "null")
    except Exception:
        pass

    sources = [
        SourceItem(
            chunk_id=h["chunk_id"],
            title=h.get("title"),
            url=h.get("url"),
            score=h.get("score"),
            snippet=(h.get("text")[:400] + "...") if h.get("text") and len(h.get("text")) > 400 else h.get("text"),
        )
        for h in final_hits
    ]

    return ChatResponse(session_id=req.session_id, answer=llm_answer, sources=sources, raw_llm=str(messages))


@app.get("/ready")
def ready():
    return {"status": "ok"}
