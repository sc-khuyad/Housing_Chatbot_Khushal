from __future__ import annotations

from html import escape
from typing import Any


def extract_twilio_whatsapp_message(form_data: dict[str, Any]) -> tuple[str, str] | None:
    body = (form_data.get("Body") or "").strip()
    sender = (form_data.get("From") or form_data.get("WaId") or "").strip()

    if not body or not sender:
        return None

    return body, sender


def format_twilio_whatsapp_reply(answer: str, sources: list[dict[str, Any]] | None = None, max_sources: int = 3) -> str:
    reply = answer.strip() or "I don't have enough relevant context to answer that."
    if not sources:
        return reply

    source_lines: list[str] = []
    for source in sources[:max_sources]:
        chunk_id = source.get("chunk_id") or "n/a"
        url = source.get("url") or "n/a"
        source_lines.append(f"- {chunk_id} | {url}")

    return f"{reply}\n\nSources:\n" + "\n".join(source_lines)


def build_twilio_twiml(message: str) -> str:
    return f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{escape(message)}</Message></Response>'