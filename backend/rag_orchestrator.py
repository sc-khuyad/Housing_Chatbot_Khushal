from typing import List, Dict, Any
from pathlib import Path
from scraper.retriever import hybrid_search
from backend.reranker import Reranker
from backend.prompt import compose_system_prompt
import json
import os
import re

ES_INDEX_NAME = os.getenv("ELASTICSEARCH_INDEX_NAME", "housing_chunks_bge_m3")

# Simple file-backed conversation memory (per-session) for now; uses LangGraph message merging if available
try:
    from langgraph.graph.message import add_messages
    HAS_LANGGRAPH = True
except Exception:
    HAS_LANGGRAPH = False


class MemoryStore:
    def __init__(self, base: Path):
        self.base = base
        os.makedirs(self.base, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.base / f"{session_id}.json"

    def load(self, session_id: str) -> List[Dict[str, Any]]:
        p = self._path(session_id)
        if not p.exists():
            return []
        return json.loads(p.read_text(encoding="utf-8"))

    def save(self, session_id: str, messages: List[Dict[str, Any]]):
        p = self._path(session_id)
        p.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")

    def _message_like(self, message: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "role": message.get("role"),
            "content": message.get("content", message.get("text")),
        }

    def _normalize_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role") or msg.get("type")
                content = msg.get("content") or msg.get("text")
            else:
                role = getattr(msg, "type", None)
                content = getattr(msg, "content", None) or getattr(msg, "text", None)

            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"

            if role and content is not None:
                normalized.append({"role": role, "text": content})

        return normalized

    def merge(self, existing: List[Dict[str, Any]], new_message: Dict[str, Any]) -> List[Dict[str, Any]]:
        if HAS_LANGGRAPH:
            try:
                existing_messages = [self._message_like(msg) for msg in existing]
                merged = add_messages(existing_messages, [self._message_like(new_message)])
                return self._normalize_messages(merged)
            except Exception:
                pass

        return existing + [new_message]


class RagOrchestrator:
    def __init__(self, memory_dir: Path | None = None):
        self.reranker = Reranker()
        base = memory_dir or (Path(__file__).resolve().parent.parent / "scraper" / "data" / "memories")
        self.memory = MemoryStore(base)

    def retrieve_then_rerank(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # get top-50 via hybrid
        hybrid = hybrid_search(query, top_k=50)
        ids = [doc_id for doc_id, _ in hybrid]

        # fetch the texts from ES
        from elasticsearch import Elasticsearch
        es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        es = Elasticsearch([es_url])
        docs = []
        for doc_id in ids:
            try:
                res = es.get(index=ES_INDEX_NAME, id=doc_id)
                src = res["_source"]
                docs.append({"id": doc_id, "text": src.get("raw_text", ""), "title": src.get("title"), "url": src.get("url")})
            except Exception:
                continue

        # rerank top-50 to top_k
        candidates = [d["text"] for d in docs]
        scores = self.reranker.score(query, candidates)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for (doc, score) in ranked:
            results.append({"chunk_id": doc["id"], "title": doc.get("title"), "url": doc.get("url"), "score": float(score), "text": doc.get("text")})
        return results

    def _build_relevance_terms(self, query: str) -> set[str]:
        raw_terms = [term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2]
        terms = set(raw_terms)

        if any(term in {"buy", "buying", "purchase", "purchasing", "acquire", "acquisition"} for term in terms):
            terms.update({"buy", "buying", "purchase", "purchasing", "seller", "buyer", "sale", "transfer", "acquire", "ownership"})

        if any(term in {"house", "home", "property", "flat", "apartment", "villa"} for term in terms):
            terms.update({"house", "home", "property", "flat", "apartment", "villa", "real", "estate"})

        if any(term in {"legal", "law", "process", "steps", "procedure", "registration"} for term in terms):
            terms.update({"legal", "process", "procedure", "steps", "step", "registration", "agreement", "stamp", "duty", "title", "deed", "possession", "document", "documents"})

        return terms

    def filter_relevant_hits(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not hits:
            return []

        query_terms = self._build_relevance_terms(query)
        if not query_terms:
            return list(hits)

        strong_terms = {
            "rent",
            "rental",
            "tenant",
            "landlord",
            "agreement",
            "agreements",
            "lease",
            "tenancy",
            "security",
            "deposit",
            "refund",
            "housing",
            "rule",
            "rules",
            "delhi",
            "ncr",
            "buy",
            "buyer",
            "seller",
            "sale",
            "purchase",
            "registration",
            "stamp",
            "duty",
            "title",
            "deed",
            "possession",
            "property",
            "house",
            "home",
        }
        process_terms = {"buy", "buyer", "seller", "sale", "purchase", "registration", "stamp", "agreement", "title", "deed", "possession"}

        relevant_hits = []
        for hit in hits:
            text_blob = " ".join([hit.get("title", ""), hit.get("text", "")]).lower()
            score = sum(1 for term in query_terms if term in text_blob)
            domain_matches = sum(1 for term in strong_terms if term in text_blob)
            has_process_signal = any(term in text_blob for term in process_terms)

            if score >= 3:
                relevant_hits.append(hit)
            elif score >= 2 and has_process_signal:
                relevant_hits.append(hit)
            elif domain_matches >= 3 and score >= 1:
                relevant_hits.append(hit)
            elif has_process_signal and score >= 1:
                relevant_hits.append(hit)

        return relevant_hits

    def append_memory(self, session_id: str, role: str, text: str):
        msgs = self.memory.load(session_id)
        message = {"role": role, "text": text}
        merged = self.memory.merge(msgs, message)
        self.memory.save(session_id, merged)

    def get_memory(self, session_id: str) -> List[Dict[str, Any]]:
        return self.memory.load(session_id)

    def clear_memory(self, session_id: str) -> bool:
        path = self.memory._path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def compose_messages(
        self,
        session_id: str,
        query: str,
        hits: List[Dict[str, Any]],
        system_prompt: str | None = None,
        debug: bool = True,
    ) -> List[Dict[str, str]]:
        # 1. Build the System Message (Instructions + Context)
        base_system = compose_system_prompt()
        system_text = f"{system_prompt}\n\n{base_system}" if system_prompt else base_system

        context_blocks = "\n\n".join([
            f"[chunk_id: {h.get('chunk_id', 'n/a')} | url: {h.get('url', 'n/a')} | title: {h.get('title', 'n/a')}]\n{h.get('text', '')}"
            for h in hits
        ])
        system_content = f"{system_text}\n\nContext Documents:\n{context_blocks}"

        messages = [{"role": "system", "content": system_content}]

        # 2. Append Conversation History
        memory = self.get_memory(session_id)
        for m in memory[-10:]:
            messages.append({"role": m["role"], "content": m["text"]})

        # 3. Append the Current User Query
        messages.append({"role": "user", "content": query})

        if debug:
            print("\n=== RAG MESSAGES ===")
            for msg in messages:
                print(f"[{msg['role'].upper()}]: {msg['content'][:150]}...")
            print("=== END RAG MESSAGES ===\n")

        return messages
