from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Multilingual reranker for English and Hindi queries
MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class Reranker:
    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    def _encode_pairs(self, pair_texts: List[tuple[str, str]]):
        if hasattr(self.tokenizer, "batch_encode_plus"):
            return self.tokenizer.batch_encode_plus(
                pair_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        queries = [query for query, _ in pair_texts]
        candidates = [candidate for _, candidate in pair_texts]
        return self.tokenizer(
            queries,
            text_pair=candidates,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def score(self, query: str, candidates: List[str], batch_size: int = 8) -> List[float]:
        """Return a list of scores aligned with candidates (higher is better)."""
        scores = []
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]
            pair_texts = [(query, c) for c in batch]
            enc = self._encode_pairs(pair_texts)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
                logits = out.logits
                # if binary, take logit for positive class; else take scalar
                if logits.shape[-1] == 1:
                    batch_scores = logits.squeeze(-1).cpu().tolist()
                else:
                    # take max logit or positive class index
                    batch_scores = logits[:, 1].cpu().tolist()
            scores.extend(batch_scores)
        return scores


if __name__ == "__main__":
    # quick smoke (will download model)
    r = Reranker()
    s = r.score("What are rent agreement rules in Delhi?", ["Article text 1", "Article text 2"]) 
    print(s)
