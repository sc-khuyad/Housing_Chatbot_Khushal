import argparse

from scraper.chunker import run_chunker
from scraper.housing_scrapper import run_scraper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the housing scraper pipeline: scrape listings, then chunk the results."
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip the scraping step and only run chunking.",
    )
    parser.add_argument(
        "--skip-chunk",
        action="store_true",
        help="Skip the chunking step and only run scraping.",
    )

    args = parser.parse_args()

    if not args.skip_scrape:
        run_scraper()

    if not args.skip_chunk:
        run_chunker()


if __name__ == "__main__":
    main()
