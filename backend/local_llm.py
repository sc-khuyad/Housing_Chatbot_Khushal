import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen3-0.6B")


def load_local_llm(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    return tokenizer, model


_tokenizer = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_local_llm():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    try:
        _tokenizer, _model = load_local_llm()
        _model.to(_device)
    except Exception:
        _tokenizer = None
        _model = None

    return _tokenizer, _model


def sanitize_generated_text(text: str) -> str | None:
    if not text:
        return None

    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return None

    prompt_markers = [
        "Conversation history:",
        "Context:",
        "Critical Rules:",
        "End of Critical Rules.",
        "You are a helpful assistant",
        "You are a strict",
    ]
    if any(marker.lower() in cleaned.lower() for marker in prompt_markers):
        return None

    if cleaned.lower().startswith("assistant:"):
        cleaned = cleaned.split(":", 1)[1].strip()

    if cleaned.lower().startswith("user:"):
        cleaned = cleaned.split(":", 1)[1].strip()

    words = re.findall(r"[a-z0-9]+", cleaned.lower())
    if len(words) < 6:
        return None

    if cleaned.endswith("?") and len(words) < 12:
        return None

    if cleaned.count("?") >= 1 and len(words) < 16:
        return None

    return cleaned or None


def generate_local_answer(messages: list, max_new_tokens: int = 30000) -> str:
    tokenizer, model = _ensure_local_llm()
    if tokenizer is None or model is None:
        return "No local model loaded."

    # 1. Apply the model's specific chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    
    # 2. Capture the length of the input tokens to slice later
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        )

    # 3. Slice the tensor to isolate only the new generation
    generated_tokens = outputs[0][input_length:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    
    cleaned = sanitize_generated_text(text)
    return cleaned or "No answer generated."
