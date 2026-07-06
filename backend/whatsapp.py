from __future__ import annotations

import os
from typing import Any

import requests


def extract_whatsapp_message(payload: dict[str, Any]) -> tuple[str, str, str | None] | None:
    entries = payload.get("entry") or []
    for entry in entries:
        for change in entry.get("changes") or []:
            value = change.get("value") or {}
            messages = value.get("messages") or []
            if not messages:
                continue

            message = messages[0] or {}
            if message.get("type") not in {None, "text"}:
                continue

            text = (message.get("text") or {}).get("body") or ""
            text = text.strip()
            sender_id = message.get("from")
            if not sender_id:
                contacts = value.get("contacts") or []
                if contacts:
                    sender_id = contacts[0].get("wa_id")

            phone_number_id = (value.get("metadata") or {}).get("phone_number_id")
            if text and sender_id:
                return text, sender_id, phone_number_id

    return None


def format_whatsapp_reply(answer: str, sources: list[dict[str, Any]] | None = None, max_sources: int = 3) -> str:
    reply = answer.strip() or "I don't have enough relevant context to answer that."
    if not sources:
        return reply

    source_lines: list[str] = []
    for source in sources[:max_sources]:
        chunk_id = source.get("chunk_id") or "n/a"
        url = source.get("url") or "n/a"
        source_lines.append(f"- {chunk_id} | {url}")

    return f"{reply}\n\nSources:\n" + "\n".join(source_lines)


def send_whatsapp_message(
    recipient: str,
    body: str,
    phone_number_id: str | None = None,
    access_token: str | None = None,
    api_version: str | None = None,
) -> None:
    token = (access_token or os.getenv("WHATSAPP_ACCESS_TOKEN") or os.getenv("WHATSAPP_TOKEN") or "").strip()
    number_id = (phone_number_id or os.getenv("WHATSAPP_PHONE_NUMBER_ID") or "").strip()
    version = (api_version or os.getenv("WHATSAPP_API_VERSION") or "v20.0").strip()

    if not token:
        raise ValueError("Missing WHATSAPP_ACCESS_TOKEN or WHATSAPP_TOKEN")
    if not number_id:
        raise ValueError("Missing WHATSAPP_PHONE_NUMBER_ID")

    response = requests.post(
        f"https://graph.facebook.com/{version}/{number_id}/messages",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "messaging_product": "whatsapp",
            "to": recipient,
            "type": "text",
            "text": {"preview_url": False, "body": body},
        },
        timeout=60,
    )
    response.raise_for_status()