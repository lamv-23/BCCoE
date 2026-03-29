"""
Document updater: scrapers for each government website + core update logic.

Each scraper returns (pdf_url, version_label) for the latest available PDF.
The DocumentUpdater class ties scrapers to the document registry, downloads
new versions when hashes differ, and triggers a callback (index rebuild).
"""

import hashlib
import logging
import os
import re
import tempfile
from datetime import datetime
from typing import Callable, Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

import document_registry as registry

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; BCCoE-DocUpdater/1.0; "
        "+https://github.com/lamv-23/bccoe)"
    )
}
DOWNLOAD_TIMEOUT = 60  # seconds


# ──────────────────────────────────────────────────────────────────────────────
# Scrapers
# ──────────────────────────────────────────────────────────────────────────────


class ATAPScraper:
    """
    Scrapes https://www.atap.gov.au/tools-techniques/cost-benefit-analysis/index
    to find the latest ATAP T2 CBA PDF link.
    Filenames follow the pattern ATAP-T2-CBA-FINAL-YYYY-MM-DD[...].pdf
    """

    source_page = "https://www.atap.gov.au/tools-techniques/cost-benefit-analysis/index"
    base_url = "https://www.atap.gov.au"

    def get_latest_pdf_url(self) -> Optional[tuple[str, str]]:
        try:
            resp = httpx.get(self.source_page, headers=HEADERS, timeout=30, follow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("ATAPScraper: failed to fetch page: %s", e)
            return None

        soup = BeautifulSoup(resp.text, "lxml")
        pattern = re.compile(r"ATAP-T2-CBA.*?(\d{4}-\d{2}-\d{2}).*?\.pdf", re.IGNORECASE)

        candidates: list[tuple[str, str]] = []  # (date_str, url)
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            m = pattern.search(href)
            if m:
                date_str = m.group(1)
                full_url = urljoin(self.base_url, href)
                candidates.append((date_str, full_url))

        if not candidates:
            logger.warning("ATAPScraper: no matching PDF links found on page")
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        date_str, url = candidates[0]
        version_label = date_str
        logger.info("ATAPScraper: latest PDF = %s (version %s)", url, version_label)
        return url, version_label


class TfNSWScraper:
    """
    Scrapes https://www.transport.nsw.gov.au/projects/project-delivery-requirements/project-cost-benefit-analysis
    to find TfNSW PDFs matching a given keyword pattern.

    PDFs are typically stored at paths like:
      /system/files/media/documents/2024/TfNSW%20CBA%20Guide%20Aug%202024.pdf
    """

    source_page = "https://www.transport.nsw.gov.au/projects/project-delivery-requirements/project-cost-benefit-analysis"
    base_url = "https://www.transport.nsw.gov.au"

    def get_latest_pdf_url(self, keyword: str) -> Optional[tuple[str, str]]:
        try:
            resp = httpx.get(self.source_page, headers=HEADERS, timeout=30, follow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("TfNSWScraper: failed to fetch page: %s", e)
            return None

        soup = BeautifulSoup(resp.text, "lxml")
        keyword_lower = keyword.lower()

        # Year pattern to extract from path or filename
        year_pattern = re.compile(r"(\d{4})")

        candidates: list[tuple[int, str, str]] = []  # (year, label, url)
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            if not href.lower().endswith(".pdf"):
                continue
            if keyword_lower not in href.lower() and keyword_lower not in tag.get_text().lower():
                continue
            full_url = urljoin(self.base_url, href)
            years = year_pattern.findall(href)
            year = int(max(years)) if years else 0
            # Use the filename as version label
            filename = href.split("/")[-1].replace("%20", " ").replace("+", " ")
            filename = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
            candidates.append((year, filename, full_url))

        if not candidates:
            logger.warning("TfNSWScraper: no PDF matching '%s' found", keyword)
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, label, url = candidates[0]
        logger.info("TfNSWScraper: latest PDF for '%s' = %s (label: %s)", keyword, url, label)
        return url, label


class NSWTreasuryScraper:
    """
    Scrapes https://www.treasury.nsw.gov.au/finance-resource/guidelines-cost-benefit-analysis
    The NSW Government CBA guide is updated in-place; version detection relies on SHA-256.
    We scrape the page for the first PDF link that contains 'tpg' or 'cost-benefit'.
    """

    source_page = "https://www.treasury.nsw.gov.au/finance-resource/guidelines-cost-benefit-analysis"
    base_url = "https://www.treasury.nsw.gov.au"
    # Fallback direct URL pattern
    fallback_url = "https://www.treasury.nsw.gov.au/sites/default/files/2023-04/tpg23-08_nsw-government-guide-to-cost-benefit-analysis_202304.pdf"

    def get_latest_pdf_url(self) -> Optional[tuple[str, str]]:
        try:
            resp = httpx.get(self.source_page, headers=HEADERS, timeout=30, follow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("NSWTreasuryScraper: failed to fetch page: %s", e)
            # Fall back to known URL
            return self.fallback_url, "latest"

        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            if not href.lower().endswith(".pdf"):
                continue
            href_lower = href.lower()
            if "tpg" in href_lower or "cost-benefit" in href_lower or "cba" in href_lower:
                full_url = urljoin(self.base_url, href)
                version_label = "latest"
                logger.info("NSWTreasuryScraper: found PDF = %s", full_url)
                return full_url, version_label

        logger.warning("NSWTreasuryScraper: no matching PDF found, using fallback")
        return self.fallback_url, "latest"


# ──────────────────────────────────────────────────────────────────────────────
# Document config
# ──────────────────────────────────────────────────────────────────────────────

_atap = ATAPScraper()
_tfnsw = TfNSWScraper()
_treasury = NSWTreasuryScraper()

DOCUMENT_CONFIG: dict[str, dict] = {
    "atap_t2": {
        "display_name": "ATAP T2 Cost-Benefit Analysis",
        "get_url": lambda: _atap.get_latest_pdf_url(),
        "filename_prefix": "atap_t2",
    },
    "tfnsw_cba": {
        "display_name": "TfNSW CBA Guide",
        "get_url": lambda: _tfnsw.get_latest_pdf_url("TfNSW CBA Guide"),
        "filename_prefix": "tfnsw_cba",
    },
    "nsw_gov_cba": {
        "display_name": "NSW Government Guide to Cost-Benefit Analysis",
        "get_url": lambda: _treasury.get_latest_pdf_url(),
        "filename_prefix": "nsw_gov_cba",
    },
    "new_cba_req": {
        "display_name": "New CBA Requirements for Transport Projects",
        "get_url": lambda: _tfnsw.get_latest_pdf_url("New CBA requirements"),
        "filename_prefix": "new_cba_req",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Core updater
# ──────────────────────────────────────────────────────────────────────────────


class DocumentUpdater:
    def __init__(
        self,
        data_dir: str,
        on_updated: Optional[Callable[[list[str]], None]] = None,
    ):
        self.data_dir = data_dir
        self.on_updated = on_updated  # async or sync callback called with list of updated doc_ids

    async def check_all(self) -> list[str]:
        """Check all documents for updates. Returns list of updated doc_ids."""
        updated: list[str] = []
        for doc_id in DOCUMENT_CONFIG:
            try:
                was_updated = await self.check_document(doc_id)
                if was_updated:
                    updated.append(doc_id)
            except Exception as e:
                logger.error("check_document(%s) failed: %s", doc_id, e)
        return updated

    async def check_document(self, doc_id: str) -> bool:
        """
        Check a single document for updates.
        Returns True if the document was updated (new version downloaded).
        """
        config = DOCUMENT_CONFIG.get(doc_id)
        if not config:
            logger.warning("Unknown doc_id: %s", doc_id)
            return False

        logger.info("Checking %s (%s)…", doc_id, config["display_name"])

        # Get the latest PDF URL from the scraper
        result = config["get_url"]()
        if result is None:
            logger.warning("%s: scraper returned no URL, skipping", doc_id)
            registry.mark_checked(doc_id)
            return False

        pdf_url, version_label = result

        # Download to a temp file
        tmp_path = None
        try:
            tmp_path = self._download_pdf(pdf_url)
            if tmp_path is None:
                registry.mark_checked(doc_id)
                return False

            new_hash = self._hash_file(tmp_path)

            # Compare with stored hash
            doc_record = registry.get(doc_id)
            stored_hash = doc_record["sha256"] if doc_record else None

            if new_hash == stored_hash:
                logger.info("%s: no change (hash match)", doc_id)
                registry.mark_checked(doc_id)
                return False

            # New version detected — save to data dir
            ext = ".pdf"
            new_filename = f"{config['filename_prefix']}_{version_label.replace(' ', '_')}{ext}"
            new_path = os.path.join(self.data_dir, new_filename)
            os.replace(tmp_path, new_path)
            tmp_path = None  # ownership transferred

            logger.info(
                "%s: NEW VERSION detected! version=%s path=%s",
                doc_id, version_label, new_path,
            )
            registry.mark_updated(doc_id, new_hash, version_label, new_path)
            return True

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _download_pdf(self, url: str) -> Optional[str]:
        """Download a PDF to a temp file. Returns temp file path or None on failure."""
        try:
            with httpx.stream("GET", url, headers=HEADERS, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                if "pdf" not in content_type and "octet-stream" not in content_type:
                    logger.warning("Unexpected content-type '%s' for %s", content_type, url)

                fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
                with os.fdopen(fd, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        f.write(chunk)
                return tmp_path
        except Exception as e:
            logger.error("Failed to download %s: %s", url, e)
            return None

    @staticmethod
    def _hash_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
