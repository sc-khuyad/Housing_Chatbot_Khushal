from pathlib import Path
import json
import os
from elasticsearch import Elasticsearch, helpers
from .embeddings import load_model, embed_texts

ES_HOST = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
INDEX_NAME = os.getenv("ELASTICSEARCH_INDEX_NAME", "housing_chunks_bge_m3")


def create_index(es: Elasticsearch, dim: int, force_recreate: bool = True):
    if es.indices.exists(index=INDEX_NAME):
        if force_recreate:
            es.indices.delete(index=INDEX_NAME)
            print(f"Deleted existing index {INDEX_NAME}")
        else:
            print(f"Index {INDEX_NAME} already exists")
            return

    mapping = {
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "url": {"type": "keyword"},
                "category": {"type": "keyword"},
                "region": {"type": "keyword"},
                "raw_text": {"type": "text"},
                "embedding": {"type": "dense_vector", "dims": dim}
            }
        }
    }
    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"Created index {INDEX_NAME} with dim={dim}")


def index_chunks(chunks_path: Path):
    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["raw_text"] for c in chunks]
    tokenizer, model, device = load_model()
    embeddings = embed_texts(texts, tokenizer, model, device, batch_size=8)

    es = Elasticsearch([ES_HOST])
    dim = len(embeddings[0])
    create_index(es, dim, force_recreate=os.getenv("FORCE_RECREATE_INDEX", "1") != "0")

    actions = []
    for doc, emb in zip(chunks, embeddings):
        action = {
            "_index": INDEX_NAME,
            "_id": doc.get("chunk_id"),
            "_source": {
                "title": doc.get("title"),
                "url": doc.get("url"),
                "category": doc.get("category"),
                "region": doc.get("region"),
                "raw_text": doc.get("raw_text"),
                "embedding": emb.tolist() if hasattr(emb, "tolist") else list(emb),
            },
        }
        actions.append(action)

    helpers.bulk(es, actions)
    print(f"Indexed {len(actions)} chunks into {INDEX_NAME}")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    chunks_path = base / "data" / "chunks.json"
    index_chunks(chunks_path)
