from __future__ import annotations

import json
import re
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_SCRAPED_PATH = DATA_DIR / "raw_scraped.json"
CHUNKS_PATH = DATA_DIR / "chunks.json"


@dataclass
class ChunkerConfig:
    raw_data_path: Path = RAW_SCRAPED_PATH
    output_path: Path = CHUNKS_PATH
    max_words: int = 250
    overlap: int = 50


class DocumentChunker:
    def __init__(self, config: ChunkerConfig | None = None):
        self.config = config or ChunkerConfig()
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def split_text_to_chunks(text: str, max_words: int = 250, overlap: int = 50) -> List[str]:
        words = text.split()
        if not words:
            return []

        if len(words) <= max_words:
            return [" ".join(words)]

        chunks = []
        start = 0
        step = max_words - overlap
        while start < len(words):
            end = min(start + max_words, len(words))
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            if end == len(words):
                break
            start += step

        return chunks

    def load_raw_documents(self) -> List[dict]:
        if not self.config.raw_data_path.exists():
            raise FileNotFoundError(f"Raw scraped file not found: {self.config.raw_data_path}")

        with self.config.raw_data_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def build_chunks(self, documents: List[dict]) -> List[dict]:
        chunks: List[dict] = []
        for doc in documents:
            raw_text = self.normalize_text(doc.get("raw_text", ""))
            if not raw_text:
                continue

            text_chunks = self.split_text_to_chunks(raw_text, max_words=self.config.max_words, overlap=self.config.overlap)
            for idx, chunk_text in enumerate(text_chunks, start=1):
                chunks.append(
                    {
                        "document_id": doc.get("document_id", f"unknown_{idx}"),
                        "chunk_id": f"{doc.get('document_id', 'unknown')}_chunk_{idx:03d}",
                        "chunk_index": idx,
                        "url": doc.get("url", ""),
                        "title": doc.get("title", ""),
                        "category": doc.get("category", ""),
                        "region": doc.get("region", ""),
                        "raw_text": chunk_text,
                        "word_count": len(chunk_text.split()),
                    }
                )
        return chunks

    def save_chunks(self, chunks: List[dict]) -> None:
        with self.config.output_path.open("w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)

    def run(self) -> None:
        print(f"Loading raw scraped documents from {self.config.raw_data_path}")
        documents = self.load_raw_documents()

        print(f"Creating chunks from {len(documents)} documents")
        chunks = self.build_chunks(documents)

        self.save_chunks(chunks)
        print(f"Saved {len(chunks)} chunks to {self.config.output_path}")


def run_chunker() -> None:
    chunker = DocumentChunker()
    chunker.run()


if __name__ == "__main__":
    run_chunker()
