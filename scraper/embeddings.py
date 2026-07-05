import os

from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel

# Multilingual embedding model for English and Hindi queries
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


def load_model(device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def embed_texts(texts: list[str], tokenizer, model, device, batch_size: int = 8):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        for k, v in enc.items():
            enc[k] = v.to(device)

        with torch.no_grad():
            out = model(**enc)
            # mean pooling over last hidden state (assuming conventional model output)
            last_hidden = out.last_hidden_state
            attention_mask = enc.get("attention_mask")
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = summed / counts
            # normalize
            normed = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            embeddings.extend(normed.cpu().numpy())

    return embeddings


def get_embeddings_for_texts(texts: list[str]):
    tokenizer, model, device = load_model()
    return embed_texts(texts, tokenizer, model, device)


if __name__ == "__main__":
    # quick smoke test
    tokenizer, model, device = load_model()
    embs = embed_texts(["Hello world", "नमस्ते दुनिया"], tokenizer, model, device)
    print("Got", len(embs), "embeddings; dim=", len(embs[0]))
