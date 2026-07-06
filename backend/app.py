import os
import re
from pathlib import Path
from typing import List
from urllib.parse import parse_qs

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response
from backend.models import ChatRequest, ChatResponse, ClearMemoryRequest, SourceItem
from backend.rag_orchestrator import RagOrchestrator
from backend.local_llm import generate_local_answer
from backend.prompt import compose_language_instruction
from backend.whatsapp import extract_whatsapp_message, format_whatsapp_reply, send_whatsapp_message
from backend.twilio_whatsapp import build_twilio_twiml, extract_twilio_whatsapp_message, format_twilio_whatsapp_reply

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


def build_chat_response(req: ChatRequest) -> ChatResponse:
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


def _get_whatsapp_verify_token() -> str:
    return os.getenv("WHATSAPP_VERIFY_TOKEN", "").strip()


def _get_whatsapp_reply_top_k() -> int:
    raw_value = os.getenv("WHATSAPP_TOP_K", "3").strip()
    try:
        return max(1, min(10, int(raw_value)))
    except ValueError:
        return 3


def _get_twilio_reply_top_k() -> int:
    raw_value = os.getenv("TWILIO_TOP_K", os.getenv("WHATSAPP_TOP_K", "3")).strip()
    try:
        return max(1, min(10, int(raw_value)))
    except ValueError:
        return 3


app = FastAPI(title="Housing RAG API")
orch = RagOrchestrator()


@app.post("/clear-memory")
def clear_memory(req: ClearMemoryRequest):
    cleared = orch.clear_memory(req.session_id)
    return {"session_id": req.session_id, "cleared": cleared}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return build_chat_response(req)


@app.get("/whatsapp/webhook")
def whatsapp_webhook_verify(request: Request):
    mode = request.query_params.get("hub.mode")
    challenge = request.query_params.get("hub.challenge")
    token = request.query_params.get("hub.verify_token")

    if mode == "subscribe" and token and token == _get_whatsapp_verify_token():
        return PlainTextResponse(challenge or "", status_code=200)

    raise HTTPException(status_code=403, detail="Invalid WhatsApp webhook verification token")


@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request):
    payload = await request.json()
    incoming = extract_whatsapp_message(payload)

    if not incoming:
        return {"ok": True, "ignored": True}

    query, sender_id, phone_number_id = incoming
    if not query:
        return {"ok": True, "ignored": True}

    chat_response = build_chat_response(
        ChatRequest(
            session_id=f"whatsapp:{sender_id}",
            query=query,
            top_k=_get_whatsapp_reply_top_k(),
            system_prompt=None,
            debug_prompt=False,
        )
    )

    message_text = format_whatsapp_reply(
        answer=chat_response.answer or "I don't have enough relevant context to answer that.",
        sources=[source.model_dump() for source in chat_response.sources],
    )

    send_whatsapp_message(
        recipient=sender_id,
        body=message_text,
        phone_number_id=phone_number_id,
    )

    return {"ok": True, "session_id": chat_response.session_id}


# @app.post("/twilio/whatsapp/webhook")
# async def twilio_whatsapp_webhook(request: Request):
#     raw_body = (await request.body()).decode("utf-8")
#     form_data = {key: values[-1] for key, values in parse_qs(raw_body, keep_blank_values=True).items()}
#     incoming = extract_twilio_whatsapp_message(form_data)

#     if not incoming:
#         return Response(content=build_twilio_twiml(""), media_type="application/xml")

#     query, sender_id = incoming
#     chat_response = build_chat_response(
#         ChatRequest(
#             session_id=f"twilio:{sender_id}",
#             query=query,
#             top_k=_get_twilio_reply_top_k(),
#             system_prompt=None,
#             debug_prompt=False,
#         )
#     )

#     message_text = format_twilio_whatsapp_reply(
#         answer=chat_response.answer or "I don't have enough relevant context to answer that.",
#         sources=[source.model_dump() for source in chat_response.sources],
#     )

#     return Response(content=build_twilio_twiml(message_text), media_type="application/xml")


from fastapi import BackgroundTasks
from twilio.rest import Client
import os

# --- Add this new background worker function ---
def process_twilio_whatsapp_in_background(query: str, sender_id: str):
    """Runs the heavy local embedding and LLM generation outside the 5-second Twilio limit."""
    
    # 1. Run the heavy RAG pipeline
    chat_response = build_chat_response(
        ChatRequest(
            session_id=f"twilio:{sender_id}",
            query=query,
            top_k=_get_twilio_reply_top_k(),
            system_prompt=None,
            debug_prompt=False,
        )
    )

    message_text = format_twilio_whatsapp_reply(
        answer=chat_response.answer or "I don't have enough relevant context to answer that.",
        sources=[source.model_dump() for source in chat_response.sources],
    )

    # 2. Push the generated answer back to WhatsApp asynchronously
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_number = os.getenv("TWILIO_WHATSAPP_NUMBER")
    
    if account_sid and auth_token and twilio_number:
        try:
            client = Client(account_sid, auth_token)
            client.messages.create(
                from_=twilio_number,
                body=message_text,
                to=sender_id
            )
        except Exception as e:
            print(f"Failed to send async Twilio message: {e}")
    else:
        print("Missing Twilio credentials in .env. Cannot send async reply.")


# --- Update your existing webhook route ---
@app.post("/twilio/whatsapp/webhook")
async def twilio_whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    raw_body = (await request.body()).decode("utf-8")
    form_data = {key: values[-1] for key, values in parse_qs(raw_body, keep_blank_values=True).items()}
    incoming = extract_twilio_whatsapp_message(form_data)

    # Instantly acknowledge Twilio to avoid the 5-second timeout
    empty_twiml = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'

    if incoming:
        query, sender_id = incoming
        # Offload the slow local model processing to the background
        background_tasks.add_task(process_twilio_whatsapp_in_background, query, sender_id)

    return Response(content=empty_twiml, media_type="application/xml")

@app.get("/ready")
def ready():
    return {"status": "ok"}
