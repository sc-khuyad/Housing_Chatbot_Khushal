import sys
import types
from pathlib import Path


def test_reranker_uses_tokenizer_call_interface(monkeypatch):
    class FakeTokenizer:
        def __call__(self, text, text_pair=None, padding=None, truncation=None, return_tensors=None):
            class FakeTensor:
                def __init__(self, value):
                    self.value = value

                def to(self, device):
                    return self

            return {"input_ids": FakeTensor([1, 2])}

    class FakeModel:
        def __init__(self):
            self.called = False

        def to(self, device):
            return self

        def eval(self):
            return None

        def __call__(self, **kwargs):
            self.called = True

            class FakeTensor:
                shape = (1, 2)

                def __getitem__(self, item):
                    return types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(tolist=lambda: [0.9]))

            return types.SimpleNamespace(logits=FakeTensor())

    import backend.reranker as br

    monkeypatch.setattr(br, "AutoTokenizer", lambda *args, **kwargs: FakeTokenizer())
    monkeypatch.setattr(br, "AutoModelForSequenceClassification", lambda *args, **kwargs: FakeModel())

    reranker = br.Reranker(device="cpu")
    scores = reranker.score("query", ["candidate"])

    assert scores == [0.9]


def test_filter_relevant_hits_removes_irrelevant_context(monkeypatch, tmp_path):
    from backend import rag_orchestrator

    monkeypatch.setattr(rag_orchestrator, "Reranker", lambda *args, **kwargs: object())
    orch = rag_orchestrator.RagOrchestrator(memory_dir=tmp_path)

    hits = [
        {
            "chunk_id": "c1",
            "title": "All about Indore Master Plan",
            "url": "u1",
            "score": 0.9,
            "text": "Transferable development rights and city master plans are discussed here.",
        },
        {
            "chunk_id": "c2",
            "title": "Can a tenant charge interest if landlord delays security deposit refund?",
            "url": "u2",
            "score": 0.7,
            "text": "The rental agreement may include clauses about security deposit, landlord, tenant, and refund timing.",
        },
    ]

    filtered = orch.filter_relevant_hits("What housing rules apply in Delhi NCR for rent agreements?", hits)

    assert [hit["chunk_id"] for hit in filtered] == ["c2"]


def test_rag_orchestrator_retrieve_and_rerank(monkeypatch, tmp_path):
    # Replace backend.reranker.Reranker with a lightweight fake to avoid heavy model downloads
    class FakeReranker:
        def __init__(self, *args, **kwargs):
            pass

        def score(self, query, candidates, batch_size: int = 8):
            # return descending dummy scores based on index
            return [float(len(candidates) - i) for i in range(len(candidates))]

    import backend.reranker as br

    monkeypatch.setattr(br, "Reranker", FakeReranker)

    # Inject a fake 'elasticsearch' module so rag_orchestrator's runtime import uses it
    fake_es_mod = types.ModuleType("elasticsearch")

    class FakeESClient:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, index, id):
            # return a minimal document structure expected by RagOrchestrator
            return {
                "_source": {
                    "raw_text": f"text for {id}",
                    "title": f"title {id}",
                    "url": f"url_{id}",
                }
            }

    fake_es_mod.Elasticsearch = FakeESClient
    sys.modules["elasticsearch"] = fake_es_mod

    # Now import the orchestrator and monkeypatch hybrid_search to return deterministic ids
    from backend import rag_orchestrator

    def fake_hybrid_search(query, top_k=50):
        # return 10 fake ids
        return [(f"chunk_{i}", 1.0 / i) for i in range(1, 11)]

    monkeypatch.setattr(rag_orchestrator, "hybrid_search", fake_hybrid_search)

    # Instantiate orchestrator pointing memory at tmp_path
    orch = rag_orchestrator.RagOrchestrator(memory_dir=tmp_path)

    results = orch.retrieve_then_rerank("sample query", top_k=5)

    # Should return top_k results with expected keys
    assert isinstance(results, list)
    assert len(results) == 5
    for r in results:
        assert "chunk_id" in r and r["chunk_id"].startswith("chunk_")
        assert "score" in r


def test_compose_prompt_contains_context_and_history(monkeypatch, tmp_path):
    # Use the same FakeReranker and fake ES module
    class FakeReranker:
        def __init__(self, *args, **kwargs):
            pass

        def score(self, query, candidates, batch_size: int = 8):
            return [1.0 for _ in candidates]

    import backend.reranker as br

    monkeypatch.setattr(br, "Reranker", FakeReranker)

    import types, sys

    fake_es_mod = types.ModuleType("elasticsearch")

    class FakeESClient:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, index, id):
            return {
                "_source": {
                    "raw_text": f"text for {id}",
                    "title": f"title {id}",
                    "url": f"url_{id}",
                }
            }

    fake_es_mod.Elasticsearch = FakeESClient
    sys.modules["elasticsearch"] = fake_es_mod

    from backend import rag_orchestrator

    monkeypatch.setattr(
        rag_orchestrator,
        "hybrid_search",
        lambda q, top_k=50: [("c1", 1.0), ("c2", 0.9)],
    )

    orch = rag_orchestrator.RagOrchestrator(memory_dir=tmp_path)
    # seed memory
    orch.append_memory("sess1", "user", "Hello")
    orch.append_memory("sess1", "assistant", "Hi, how can I help?")

    hits = [
        {
            "chunk_id": "c1",
            "title": "t1",
            "url": "u1",
            "score": 1.0,
            "text": "text for c1",
        },
        {
            "chunk_id": "c2",
            "title": "t2",
            "url": "u2",
            "score": 0.9,
            "text": "text for c2",
        },
    ]

    prompt = orch.compose_prompt("sess1", "What is rent agreement?", hits)
    print(prompt)

    assert "Critical Rules" in prompt
    assert "text for c1" in prompt
    assert "user: Hello" in prompt


def test_compose_prompt_accepts_custom_system_prompt(monkeypatch, tmp_path):
    orch = None
    class FakeReranker:
        def __init__(self, *args, **kwargs):
            pass

        def score(self, query, candidates, batch_size: int = 8):
            return [1.0 for _ in candidates]

    import backend.reranker as br
    monkeypatch.setattr(br, "Reranker", FakeReranker)

    import types, sys
    fake_es_mod = types.ModuleType("elasticsearch")

    class FakeESClient:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, index, id):
            return {"_source": {"raw_text": "demo", "title": "demo", "url": "demo"}}

    fake_es_mod.Elasticsearch = FakeESClient
    sys.modules["elasticsearch"] = fake_es_mod

    from backend import rag_orchestrator
    monkeypatch.setattr(rag_orchestrator, "hybrid_search", lambda q, top_k=50: [("c1", 1.0)])

    orch = rag_orchestrator.RagOrchestrator(memory_dir=tmp_path)
    prompt = orch.compose_prompt(
        "sess2",
        "Explain the process",
        [{"chunk_id": "c1", "title": "t1", "url": "u1", "score": 1.0, "text": "demo chunk"}],
        system_prompt="You are a custom tester. Answer in one short sentence.",
    )

    assert "You are a custom tester" in prompt
    assert "demo chunk" in prompt


def test_sanitize_generated_text_rejects_prompt_echo():
    from backend.local_llm import sanitize_generated_text

    prompt_echo = "Conversation history:\nuser: hi\nContext:\nchunk:1 some text"

    assert sanitize_generated_text(prompt_echo) is None


def test_compose_messages_includes_chunk_metadata(tmp_path):
    from backend import rag_orchestrator

    orch = rag_orchestrator.RagOrchestrator(memory_dir=tmp_path)
    messages = orch.compose_messages(
        "sess-meta",
        "What does RERA cover?",
        [
            {
                "chunk_id": "housing_delhi_001_006_chunk_001",
                "title": "RERA basics",
                "url": "https://example.com/rera",
                "score": 1.0,
                "text": "RERA protects homebuyers and regulates projects.",
            }
        ],
        debug=False,
    )

    system_content = messages[0]["content"]
    assert "chunk_id: housing_delhi_001_006_chunk_001" in system_content
    assert "url: https://example.com/rera" in system_content


def test_filter_relevant_hits_keeps_property_purchase_context(monkeypatch, tmp_path):
    from backend import rag_orchestrator

    monkeypatch.setattr(rag_orchestrator, "Reranker", lambda *args, **kwargs: object())
    orch = rag_orchestrator.RagOrchestrator(memory_dir=tmp_path)

    hits = [
        {
            "chunk_id": "c1",
            "title": "States devise new methods, as property registrations drop amid COVID-19",
            "url": "u1",
            "score": 0.9,
            "text": "Stamp duty in key tier-2 cities in India. Can property registration be completed online?",
        },
        {
            "chunk_id": "c2",
            "title": "How to prioritise must-haves vs nice-to-haves in a new home?",
            "url": "u2",
            "score": 0.8,
            "text": "A property with an additional room can become a must-have feature.",
        },
    ]

    filtered = orch.filter_relevant_hits("What is the legal process of buying a house?", hits)

    assert [hit["chunk_id"] for hit in filtered] == ["c1"]


def test_clear_memory_removes_session_file(tmp_path):
    from backend import rag_orchestrator

    orch = rag_orchestrator.RagOrchestrator(memory_dir=tmp_path)
    orch.append_memory("s1", "user", "hello")

    assert orch.clear_memory("s1") is True
    assert orch.get_memory("s1") == []
    assert orch.clear_memory("s1") is False


def test_generation_backend_defaults_to_groq(monkeypatch):
    from backend import app as app_module

    monkeypatch.delenv("GENERATION_BACKEND", raising=False)
    assert app_module.get_generation_backend() == "groq"

    monkeypatch.setenv("GENERATION_BACKEND", "local")
    assert app_module.get_generation_backend() == "local"


def test_chat_returns_response_when_generation_fails(monkeypatch):
    from backend import app as app_module
    from backend.models import ChatRequest

    monkeypatch.setattr(app_module.orch, "retrieve_then_rerank", lambda query, top_k=50: [{"chunk_id": "c1", "title": "t", "url": "u", "score": 1.0, "text": "sample"}])
    monkeypatch.setattr(app_module.orch, "filter_relevant_hits", lambda query, hits: hits)
    monkeypatch.setattr(app_module, "generate_answer", lambda messages: (_ for _ in ()).throw(RuntimeError("boom")))

    response = app_module.chat(ChatRequest(session_id="demo", query="hello", top_k=1, system_prompt=None, debug_prompt=False))

    assert response.answer is not None


def test_extract_whatsapp_message_parses_text_payload():
    from backend.whatsapp import extract_whatsapp_message

    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "from": "919999999999",
                                    "type": "text",
                                    "text": {"body": "What is RERA?"},
                                }
                            ],
                            "metadata": {"phone_number_id": "123456789"},
                        }
                    }
                ]
            }
        ]
    }

    assert extract_whatsapp_message(payload) == ("What is RERA?", "919999999999", "123456789")


def test_format_whatsapp_reply_includes_chunk_sources():
    from backend.whatsapp import format_whatsapp_reply

    reply = format_whatsapp_reply(
        "RERA regulates real estate projects.",
        [
            {"chunk_id": "housing_delhi_001_006_chunk_001", "url": "https://example.com/rera"},
            {"chunk_id": "housing_delhi_001_007_chunk_002", "url": "https://example.com/second"},
        ],
    )

    assert "RERA regulates real estate projects." in reply
    assert "housing_delhi_001_006_chunk_001" in reply
    assert "https://example.com/rera" in reply


def test_extract_twilio_whatsapp_message_parses_form_payload():
    from backend.twilio_whatsapp import extract_twilio_whatsapp_message

    payload = {
        "Body": "What is RERA?",
        "From": "whatsapp:+919999999999",
        "WaId": "919999999999",
    }

    assert extract_twilio_whatsapp_message(payload) == ("What is RERA?", "whatsapp:+919999999999")


def test_build_twilio_twiml_wraps_reply_text():
    from backend.twilio_whatsapp import build_twilio_twiml, format_twilio_whatsapp_reply

    reply = format_twilio_whatsapp_reply(
        "RERA regulates real estate projects.",
        [{"chunk_id": "housing_delhi_001_006_chunk_001", "url": "https://example.com/rera"}],
    )
    twiml = build_twilio_twiml(reply)

    assert twiml.startswith("<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
    assert "<Message>" in twiml
    assert "housing_delhi_001_006_chunk_001" in twiml
