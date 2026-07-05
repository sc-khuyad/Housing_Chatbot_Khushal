import os
import streamlit as st
import requests
from typing import Any

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/chat")


def normalize_backend_url(raw_url: str | None) -> str:
    if not raw_url:
        return DEFAULT_BACKEND_URL

    url = raw_url.strip()
    if not url:
        return DEFAULT_BACKEND_URL

    url = url.replace("0.0.0.0", "127.0.0.1")
    if ":8501" in url and "/chat" in url:
        url = url.replace(":8501", ":8000", 1)
    if url.endswith("/chat") and ":8000" not in url and ":8001" not in url:
        url = url.replace("/chat", ":8000/chat", 1)
    return url


st.set_page_config(page_title="Housing RAG Chat", layout="wide")
st.title("Housing RAG Chatbot")
st.markdown(
    "Use this interface to ask housing questions and get source-backed answers from your FastAPI RAG backend."
)

with st.sidebar:
    st.header("Settings")
    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_BACKEND_URL

    # The widget maintains `st.session_state['api_url']` itself; read its
    # current value and normalize it without assigning back to session_state
    api_url_raw = st.text_input("Backend API URL", value=st.session_state.api_url, key="api_url")
    api_url = normalize_backend_url(api_url_raw)
    session_id = st.text_input("Session ID", "demo")
    top_k = st.number_input("Top K hits", min_value=1, max_value=20, value=5, step=1)
    system_prompt = st.text_area(
        "Custom system prompt",
        "",
        help="Optional instructions to prepend to the RAG prompt before the default policy.",
    )
    debug_prompt = st.checkbox("Show raw prompt/debug info", value=False)

    if st.button("Reset conversation"):
        st.session_state.history = []
        st.session_state.pop("latest_sources", None)
        st.session_state.pop("latest_raw_llm", None)
        st.session_state.pop("latest_answer", None)
        st.rerun()

    if st.button("Clear chatbot memory"):
        try:
            clear_url = api_url.rstrip("/").rsplit("/chat", 1)[0] + "/clear-memory" if api_url.rstrip("/").endswith("/chat") else api_url.rstrip("/") + "/clear-memory"
            res = requests.post(clear_url, json={"session_id": session_id}, timeout=6000)
            res.raise_for_status()
            st.session_state.history = []
            st.session_state.pop("latest_sources", None)
            st.session_state.pop("latest_raw_llm", None)
            st.session_state.pop("latest_answer", None)
            st.success("Conversation memory cleared for this session.")
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Failed to clear memory: {exc}")

if "history" not in st.session_state:
    st.session_state.history = []

response_container = st.container()
source_container = st.container()
summary_container = st.container()

with st.form(key="chat_form", clear_on_submit=True):
    user_query = st.text_input("Enter your question", key="query_input")
    submit = st.form_submit_button("Send")

if submit and user_query:
    payload: dict[str, Any] = {
        "session_id": session_id,
        "query": user_query,
        "top_k": top_k,
        "system_prompt": system_prompt if system_prompt.strip() else None,
        "debug_prompt": debug_prompt,
    }

    try:
        with st.spinner("Sending query to backend..."):
            res = requests.post(api_url, json=payload, timeout=6000)
            res.raise_for_status()
            chat_data = res.json()
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
        chat_data = None

    if chat_data:
        answer = chat_data.get("answer") or "No answer generated."
        raw_llm = chat_data.get("raw_llm")
        sources = chat_data.get("sources", [])

        st.session_state.history.append({"role": "user", "text": user_query})
        st.session_state.history.append({"role": "assistant", "text": answer})
        st.session_state.latest_sources = sources
        st.session_state.latest_raw_llm = raw_llm
        st.session_state.latest_answer = answer

with response_container:
    for message in st.session_state.history:
        if message["role"] == "user":
            st.chat_message("user").write(message["text"])
        else:
            st.chat_message("assistant").write(message["text"])

if "latest_answer" in st.session_state:
    with summary_container.expander("Answer summary", expanded=True):
        st.write(st.session_state.latest_answer)

if "latest_sources" in st.session_state:
    with source_container.expander("Retrieved sources", expanded=True):
        if st.session_state.latest_sources:
            for idx, source in enumerate(st.session_state.latest_sources, start=1):
                title = source.get("title") or source.get("chunk_id") or "Source"
                chunk_id = source.get("chunk_id") or "N/A"
                url = source.get("url") or "N/A"
                st.markdown(
                    f"**{idx}. {title}**\n"
                    f"Chunk ID: {chunk_id}\n"
                    f"URL: {url}\n"
                    f"Score: {source.get('score')}"
                )
                if source.get("snippet"):
                    st.write(source.get("snippet"))
                st.markdown("---")
        else:
            st.info("No sources returned by the backend.")

if debug_prompt and "latest_raw_llm" in st.session_state:
    with st.expander("Raw prompt / debug output", expanded=False):
        st.code(st.session_state.latest_raw_llm or "(no raw prompt returned)")
