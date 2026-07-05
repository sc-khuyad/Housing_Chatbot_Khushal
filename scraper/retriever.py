from elasticsearch import Elasticsearch
from .embeddings import load_model, embed_texts
from typing import List
import numpy as np
import os

ES_HOST = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
INDEX_NAME = os.getenv("ELASTICSEARCH_INDEX_NAME", "housing_chunks_bge_m3")


def dense_search(es: Elasticsearch, query_vector: List[float], top_k: int = 10):
    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_vector},
                },
            }
        }
    }
    res = es.search(index=INDEX_NAME, body=body)
    return [(hit['_id'], hit['_score']) for hit in res['hits']['hits']]


def bm25_search(es: Elasticsearch, query: str, top_k: int = 10):
    body = {
        "size": top_k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "raw_text"],
            }
        }
    }
    res = es.search(index=INDEX_NAME, body=body)
    return [(hit['_id'], hit['_score']) for hit in res['hits']['hits']]


def rrf_merge(dense_results, bm25_results, k: int = 60):
    # dense_results and bm25_results are lists of (id, score) ordered by rank
    scores = {}
    for rank, (doc_id, _) in enumerate(dense_results, start=1):
        scores.setdefault(doc_id, 0.0)
        scores[doc_id] += 1.0 / (k + rank)
    for rank, (doc_id, _) in enumerate(bm25_results, start=1):
        scores.setdefault(doc_id, 0.0)
        scores[doc_id] += 1.0 / (k + rank)

    # return sorted list by RRF score
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged


def hybrid_search(query: str, top_k: int = 10):
    es = Elasticsearch([ES_HOST])
    tokenizer, model, device = load_model()
    q_emb = embed_texts([query], tokenizer, model, device, batch_size=1)[0].tolist()

    dense = dense_search(es, q_emb, top_k=top_k)
    bm25 = bm25_search(es, query, top_k=top_k)

    merged = rrf_merge(dense, bm25)
    return merged


if __name__ == "__main__":
    res = hybrid_search("rent agreement Delhi RERA act", top_k=10)
    print("Top results (doc_id, rrf_score):")
    for doc_id, score in res[:10]:
        print(doc_id, score)
