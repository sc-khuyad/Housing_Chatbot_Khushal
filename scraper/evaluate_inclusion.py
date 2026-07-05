import json
from pathlib import Path

from scraper.housing_scrapper import TARGET_URLS, extract_article_links
from scraper.chunker import CHUNKS_PATH

RAW_SCRAPED_PATH = Path(__file__).resolve().parent / "data" / "raw_scraped.json"


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_raw_scraped(raw_docs: list[dict]) -> dict:
    results = {}
    doc_urls = {doc.get("url") for doc in raw_docs if doc.get("url")}

    for seed in TARGET_URLS:
        seed_url = seed["url"]
        category = seed["category"]
        expected_links = extract_article_links(seed_url, max_links=10)

        if expected_links:
            missing = [link for link in expected_links if link not in doc_urls]
            results[seed_url] = {
                "category": category,
                "mode": "deep_links",
                "scraped_count": len([link for link in expected_links if link in doc_urls]),
                "expected_count": len(expected_links),
                "missing": missing,
            }
        else:
            found = any(doc.get("url") == seed_url for doc in raw_docs)
            results[seed_url] = {
                "category": category,
                "mode": "direct_seed",
                "found": found,
                "expected_url": seed_url,
            }

    return results


def evaluate_chunks(raw_docs: list[dict], chunks: list[dict]) -> dict:
    raw_ids = {doc.get("document_id") for doc in raw_docs if doc.get("document_id")}
    chunks_by_doc = {}
    for chunk in chunks:
        doc_id = chunk.get("document_id")
        if not doc_id:
            continue
        chunks_by_doc.setdefault(doc_id, 0)
        chunks_by_doc[doc_id] += 1

    coverage = {
        doc_id: chunks_by_doc.get(doc_id, 0)
        for doc_id in sorted(raw_ids)
    }
    return {
        "total_docs": len(raw_ids),
        "total_chunks": len(chunks),
        "coverage_by_doc": coverage,
    }


def print_results(raw_results: dict, chunk_results: dict) -> None:
    print("\n=== Raw Scraped Inclusion Results ===")
    for url, data in raw_results.items():
        if data["mode"] == "deep_links":
            status = "PASS" if data["scraped_count"] == data["expected_count"] else "FAIL"
            print(
                f"{status}: {url} ({data['category']}) - {data['scraped_count']}/{data['expected_count']} found"
            )
            if data["missing"]:
                print(f"  Missing {len(data['missing'])} URLs:")
                for missing_url in data["missing"]:
                    print(f"    - {missing_url}")
        else:
            status = "PASS" if data["found"] else "FAIL"
            print(f"{status}: {url} ({data['category']}) - direct page found = {data['found']}")

    print("\n=== Chunk Coverage ===")
    print(f"Total docs: {chunk_results['total_docs']}")
    print(f"Total chunks: {chunk_results['total_chunks']}")
    uncovered = [doc_id for doc_id, count in chunk_results["coverage_by_doc"].items() if count == 0]
    if uncovered:
        print(f"Docs with no chunks: {len(uncovered)}")
        for doc_id in uncovered:
            print(f"  - {doc_id}")
    else:
        print("All scraped documents have at least one chunk.")


def main() -> None:
    raw_docs = load_json(RAW_SCRAPED_PATH)
    chunks = []
    if Path(CHUNKS_PATH).exists():
        chunks = load_json(Path(CHUNKS_PATH))
    else:
        print(f"Warning: chunks file not found at {CHUNKS_PATH}. Skipping chunk coverage evaluation.")

    raw_results = evaluate_raw_scraped(raw_docs)
    chunk_results = evaluate_chunks(raw_docs, chunks)
    print_results(raw_results, chunk_results)


if __name__ == "__main__":
    main()
