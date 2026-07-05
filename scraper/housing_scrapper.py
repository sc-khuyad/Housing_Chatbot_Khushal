from __future__ import annotations

import json
from datetime import datetime, timezone
import time
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup

# Playwright is used for robust browsing to avoid WAF blocks.
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    HAS_PLAYWRIGHT = True
except Exception:
    HAS_PLAYWRIGHT = False


DATA_DIR = Path(__file__).resolve().parent / "data"
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_TARGET_URLS = [
    {"url": "https://housing.com/news?paged=950", "category": "Buying_Guide", "doc_id": "housing_delhi_001"},
    {"url": "https://housing.com/news?paged=4", "category": "Buying_Guide", "doc_id": "housing_delhi_004"},
    {"url": "https://housing.com/news?paged=1011", "category": "Locality_Trends", "doc_id": "housing_delhi_005"},
    {"url": "https://housing.com/news?paged=942", "category": "Locality_Trends", "doc_id": "housing_delhi_006"},
    {"url": "https://housing.com/news?paged=1230", "category": "EMI_Info", "doc_id": "housing_delhi_007"},
    {"url": "https://housing.com/in/home-loans/faq/18-how-does-the-tenure-affect-the-cost-of-the-loan", "category": "EMI_Info", "doc_id": "housing_delhi_008"},
    {"url": "https://housing.com/contact-us", "category": "FAQ_Support", "doc_id": "housing_delhi_009"},
    {"url": "https://housing.com/in/home-loans/faq", "category": "FAQ_Support", "doc_id": "housing_delhi_010"},
    {"url": "https://housing.com/noida-greater-noida-expressway-noida-overview-Ppapzdour07bmxvz", "category": "Buying_Guide", "doc_id": "housing_delhi_011"},
    {"url": "https://new.housing.com/free-rent-agreement-format", "category": "Buying_Guide", "doc_id": "housing_delhi_012"},
    {"url": "https://new.housing.com/tenant-verification", "category": "Buying_Guide", "doc_id": "housing_delhi_013"},
    {"url": "https://housing.com/flatmates/shared-accomodation-for-rent-in-gurgaon-haryana-P1od1w26jrfqap1jl", "category": "Buying_Guide", "doc_id": "housing_delhi_014"},
    {"url": "https://new.housing.com/discount-pricing-tenant-verification", "category": "Buying_Guide", "doc_id": "housing_delhi_015"},
    {"url": "https://housing.com/rent/20209235-4500-sqft-8-bhk-apartment-on-rent-in-vasant-vihar-delhi", "category": "Buying_Guide", "doc_id": "housing_delhi_016"},
    {"url": "https://housing.com/sector-c-overview-Pg7tuy2aw0y1rn46", "category": "City_Guides", "doc_id": "housing_delhi_017"},
    {"url": "https://housing.com/southern-peripheral-road-gurgaon-overview-P3k8picn3z3olvut2", "category": "City_Guides", "doc_id": "housing_delhi_018"},
    {"url": "https://housing.com/in/buy/gurgaon/under-50-lakhs-fid/", "category": "City_Guides", "doc_id": "housing_delhi_019"},
    {"url": "https://housing.com/in/buy/noida/under-50-lakhs-fid/", "category": "City_Guides", "doc_id": "housing_delhi_020"},
    {"url": "https://housing.com/in/buy/new-delhi/under-50-lakhs-fid/", "category": "City_Guides", "doc_id": "housing_delhi_021"},
    {"url": "https://housing.com/paying-guests/pg-in-new-delhi-P6xfqdsey6cc3d95h", "category": "City_Guides", "doc_id": "housing_delhi_022"},
    {"url": "https://housing.com/paying-guests/pg-in-delhi-central-P73zmp9xy6s6egbw", "category": "City_Guides", "doc_id": "housing_delhi_023"},
    {"url": "https://housing.com/paying-guests/pg-with-food-in-south-delhi-AP0P49rjj9lqcgscvm36", "category": "City_Guides", "doc_id": "housing_delhi_024"},
    {"url": "https://housing.com/paying-guests/pg-in-south-delhi-P49rjj9lqcgscvm36", "category": "City_Guides", "doc_id": "housing_delhi_025"},
    {"url": "https://housing.com/in/buy/projects/page/33413-emaar-gurgaon-greens-by-emaar-india-in-sector-102", "category": "City_Guides", "doc_id": "housing_delhi_026"},
    {"url": "https://housing.com/in/buy/projects/page/231893-birla-navya-gurugram-by-birla-estates-pvt-ltd-in-kadarpur", "category": "City_Guides", "doc_id": "housing_delhi_027"},
    {"url": "https://housing.com/in/buy/projects/page/282728-real-83-avenue-by-real-town-properties-pvt-ltd-in-sector-83", "category": "City_Guides", "doc_id": "housing_delhi_028"},
    {"url": "https://housing.com/in/buy/projects/page/225858-signature-orchard-avenue-2-by-signature-global-in-sector-93", "category": "City_Guides", "doc_id": "housing_delhi_029"},
    {"url": "https://housing.com/in/buy/resale/page/20560787-residential-plot-in-sector-138-for-rs-11000000", "category": "City_Guides", "doc_id": "housing_delhi_030"},
    {"url": "https://housing.com/sector-137-noida-overview-P2lu7arqq5090hq61", "category": "City_Guides", "doc_id": "housing_delhi_031"},
    {"url": "https://housing.com/in/buy/resale/page/20598248-residential-plot-in-sector-138-for-rs-11800000", "category": "City_Guides", "doc_id": "housing_delhi_032"},
    {"url": "https://housing.com/price-trends/property-rates-for-buy-in-sector_22d_greater_noida_yeida_greater_noida-P4nqg4v0kt4v2vibx", "category": "City_Guides", "doc_id": "housing_delhi_033"},
    {"url": "https://housing.com/in/buy/gurgaon/dwarka-expressway-gid/", "category": "City_Guides", "doc_id": "housing_delhi_034"},
    {"url": "https://housing.com/in/buy/gurgaon/dwarka-expressway-gid/2bhk-flats-fid/", "category": "City_Guides", "doc_id": "housing_delhi_035"},
    {"url": "https://housing.com/rent/withoutbrokerage-flats-for-rent-in-dwarka-expressway-gurgaon-D2P3bt1uv74npfwg0rs", "category": "City_Guides", "doc_id": "housing_delhi_036"},
    {"url": "https://housing.com/gamma-sahibabad-ghaziabad-overview-P2sf9gwxea5rxr7g7", "category": "Price_Query", "doc_id": "housing_delhi_037"},
    {"url": "https://housing.com/greater-noida-uttar-pradesh-overview-P6fh9cuf4xravz1ts", "category": "Price_Query", "doc_id": "housing_delhi_038"},
    {"url": "https://housing.com/faridabad-haryana-overview-P3nlekdze1dlp2923", "category": "Price_Query", "doc_id": "housing_delhi_039"},
    {"url": "https://housing.com/terms-of-use", "category": "Price_Query", "doc_id": "housing_delhi_040"},
    {"url": "https://housing.com/news?paged=1198", "category": "Price_Query", "doc_id": "housing_delhi_041"},
    {"url": "https://housing.com/news?paged=456", "category": "Price_Query", "doc_id": "housing_delhi_042"},
    {"url": "https://new.housing.com/legal-services", "category": "Price_Query", "doc_id": "housing_delhi_043"},
    {"url": "https://housing.com/in/home-loans/faq/18-how-does-the-tenure-affect-the-cost-of-the-loan", "category": "EMI_Info", "doc_id": "housing_delhi_044"},
    {"url": "https://housing.com/in/home-loans/eligibility-check/details", "category": "EMI_Info", "doc_id": "housing_delhi_045"},
    {"url": "https://housing.com/home-affordability-calculator", "category": "EMI_Info", "doc_id": "housing_delhi_046"},
    {"url": "https://housing.com/in/home-loans/faq/26-what-are-the-documents-required-for-my-home-loan-application", "category": "EMI_Info", "doc_id": "housing_delhi_047"},
    {"url": "https://support.housing.com/support/solutions/articles/4000200515-how-does-tenure-affect-the-cost-of-a-loan-", "category": "EMI_Info", "doc_id": "housing_delhi_048"},
    {"url": "https://housing.com/home-loans/yes-bank/interest-rates", "category": "EMI_Info", "doc_id": "housing_delhi_049"},
    {"url": "https://housing.com/home-loans/kotak-mahindra", "category": "EMI_Info", "doc_id": "housing_delhi_050"},
    {"url": "https://support.housing.com/support/solutions/articles/4000172979-how-do-i-login-signup-", "category": "FAQ_Support", "doc_id": "housing_delhi_051"},
    {"url": "https://careers.housing.com/terms-of-use/", "category": "FAQ_Support", "doc_id": "housing_delhi_052"},
    {"url": "https://support.housing.com/support/solutions/articles/4000200887-main-apna-account-kaise-deactivate-kar-sakta-hoon-", "category": "FAQ_Support", "doc_id": "housing_delhi_053"},
    {"url": "https://support.housing.com/support/solutions", "category": "FAQ_Support", "doc_id": "housing_delhi_054"},
    {"url": "https://housing.com/owner-packages", "category": "FAQ_Support", "doc_id": "housing_delhi_055"},
    {"url": "https://housing.com/rent/flats-for-rent-in-new-delhi-india-P6xfqdsey6cc3d95h", "category": "FAQ_Support", "doc_id": "housing_delhi_056"},
    {"url": "https://support.housing.com/support/solutions/articles/4000200811-how-can-i-post-a-listing-on-housing-com-", "category": "FAQ_Support", "doc_id": "housing_delhi_057"},
    {"url": "https://contents.housing.com/2/65/301296/746/b0fcd7eb-5b1d-419a.pdf", "category": "FAQ_Support", "doc_id": "housing_delhi_058"},
]

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"


@dataclass
class ScraperConfig:
    target_urls: List[dict] = field(default_factory=lambda: DEFAULT_TARGET_URLS)
    max_articles_per_seed: int = 30
    region: str = "Delhi NCR"
    headless: bool = False
    timeout_ms: int = 25000
    data_dir: Path = DATA_DIR
    user_agent: str = USER_AGENT


class PlaywrightSession:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self._browser = None
        self._playwright = None

    def __enter__(self):
        if not HAS_PLAYWRIGHT:
            raise RuntimeError("Playwright is not installed. Install with: pip install playwright && playwright install")
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def new_page(self):
        return self._browser.new_page()


class ArticleScraper:
    @staticmethod
    def clean_html(soup: BeautifulSoup) -> BeautifulSoup:
        for unwanted in soup(["nav", "footer", "script", "style", "aside", "header", "form", "iframe"]):
            unwanted.decompose()
        return soup

    @staticmethod
    def normalize_url(href: str) -> str:
        href = href.strip()
        if href.startswith("//"):
            return f"https:{href}"
        if href.startswith("/"):
            return f"https://housing.com{href}"
        return href

    @classmethod
    def extract_links(cls, html: str, max_links: int = 10) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        anchors = soup.select("a.article-card-link")
        if not anchors:
            anchors = soup.find_all("a", href=True)

        discovered = []
        for anchor in anchors:
            link_text = anchor.get_text(strip=True).upper()
            if "READ FULL STORY" not in link_text:
                continue

            href = anchor.get("href")
            if not href or "/news/" not in href:
                continue

            normalized = cls.normalize_url(href)
            if normalized not in discovered:
                discovered.append(normalized)
            if len(discovered) >= max_links:
                break

        return discovered

    @classmethod
    def scrape_article(cls, page, url: str, category: str, region: str = "Delhi NCR") -> Optional[dict]:
        print(f"Scraping article: {url}")
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=cls._timeout_ms())
            page.wait_for_timeout(3000)
            html = page.content()
            soup = BeautifulSoup(html, "html.parser")
        except PlaywrightTimeoutError:
            print(f"  -> Playwright timeout loading article {url}")
            return None
        except Exception as e:
            print(f"  -> Playwright error loading article {url}: {e}")
            return None

        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else ""
        cls.clean_html(soup)

        article_body = soup.find("article") or soup.find("main") or soup.find("body")
        raw_text = article_body.get_text(separator=" ", strip=True) if article_body else ""

        if not raw_text or len(raw_text) < 50:
            print(f"  -> Warning: short content for {url}")

        return {
            "url": url,
            "title": title,
            "category": category,
            "region": region,
            "raw_text": raw_text,
        }

    @staticmethod
    def _timeout_ms() -> int:
        return 25000


class ScraperEngine:
    def __init__(self, config: ScraperConfig | None = None):
        self.config = config or ScraperConfig()
        self.data_dir = self.config.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.data_dir / "raw_scraped.json"
        self.log_file = self.data_dir / "scrape_log.txt"

    def _write_log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.log_file.open("a", encoding="utf-8") as logfile:
            logfile.write(f"[{timestamp}] {message}\n")

    def scrape_listing(self, page, listing_url: str, category: str) -> List[str]:
        print(f"Extracting article links from listing: {listing_url}")
        self._write_log(f"LISTING_START category={category} url={listing_url}")
        try:
            page.goto(listing_url, wait_until="domcontentloaded", timeout=self.config.timeout_ms)
            page.wait_for_timeout(3000)
        except PlaywrightTimeoutError:
            print(f"  -> Playwright timeout loading listing {listing_url}")
            self._write_log(f"LISTING_FAIL category={category} url={listing_url} reason=timeout")
            return []
        except Exception as e:
            print(f"  -> Playwright error fetching listing {listing_url}: {e}")
            self._write_log(f"LISTING_FAIL category={category} url={listing_url} reason={type(e).__name__}: {e}")
            return []

        html = page.content()
        links = ArticleScraper.extract_links(html, max_links=self.config.max_articles_per_seed)
        self._write_log(f"LISTING_DONE category={category} url={listing_url} links={len(links)}")
        return links

    def build_document(self, url: str, category: str, page, document_id: str) -> Optional[dict]:
        doc = ArticleScraper.scrape_article(page, url, category, region=self.config.region)
        if not doc or len(doc.get("raw_text", "")) < 100:
            self._write_log(f"ARTICLE_FAIL document_id={document_id} category={category} url={url} reason=empty_or_short_content")
            return None
        doc["document_id"] = document_id
        title = doc.get("title", "").replace("\n", " ").strip()
        self._write_log(f"ARTICLE_OK document_id={document_id} category={category} url={url} title={title}")
        return doc

    def save_documents(self, documents: List[dict]) -> None:
        with self.output_file.open("w", encoding="utf-8") as outfile:
            json.dump(documents, outfile, indent=4, ensure_ascii=False)
        print(f"Saved {len(documents)} documents to {self.output_file}")
        self._write_log(f"SAVE_DONE documents={len(documents)} output={self.output_file}")

    def run(self) -> None:
        if not HAS_PLAYWRIGHT:
            raise RuntimeError("Playwright is not installed. Install with: pip install playwright && playwright install")

        documents: List[dict] = []
        self._write_log("RUN_START")
        with PlaywrightSession(headless=self.config.headless) as session:
            page = session.new_page()
            for seed in self.config.target_urls:
                listing_url = seed["url"]
                category = seed["category"]
                doc_prefix = seed.get("doc_id", category).replace(" ", "_")

                links = self.scrape_listing(page, listing_url, category)
                if not links:
                    print(f"No links extracted from {listing_url}; falling back to seed URL")
                    links = [listing_url]

                for idx, link in enumerate(links, start=1):
                    document_id = f"{doc_prefix}_{idx:03d}"
                    document = self.build_document(link, category, page, document_id)
                    if document:
                        documents.append(document)
                    time.sleep(3.0)

        self.save_documents(documents)
        self._write_log(f"RUN_DONE documents={len(documents)}")
        print(f"\nScraping complete. Collected {len(documents)} documents.")


def run_scraper() -> None:
    engine = ScraperEngine()
    engine.run()


if __name__ == "__main__":
    run_scraper()
