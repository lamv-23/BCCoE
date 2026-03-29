"""
SQLite-backed registry for tracking document versions and update state.
"""

import sqlite3
import hashlib
import os
from datetime import datetime
from typing import Optional


DB_PATH = os.environ.get("REGISTRY_DB_PATH", os.path.join(os.path.dirname(__file__), "document_registry.db"))

# Seed data matching the existing PDFs in data/
INITIAL_DOCUMENTS = [
    {
        "doc_id": "atap_t2",
        "display_name": "ATAP T2 Cost-Benefit Analysis",
        "source_page_url": "https://www.atap.gov.au/tools-techniques/cost-benefit-analysis/index",
        "local_filename": "ATAP-T2-CBA-FINAL-2022-04-26_20250526.pdf",
        "version_label": "2022-04-26",
    },
    {
        "doc_id": "tfnsw_cba",
        "display_name": "TfNSW CBA Guide",
        "source_page_url": "https://www.transport.nsw.gov.au/projects/project-delivery-requirements/project-cost-benefit-analysis",
        "local_filename": "TfNSW CBA Guide Aug 2024_20250526.pdf",
        "version_label": "Aug 2024",
    },
    {
        "doc_id": "nsw_gov_cba",
        "display_name": "NSW Government Guide to Cost-Benefit Analysis",
        "source_page_url": "https://www.treasury.nsw.gov.au/finance-resource/guidelines-cost-benefit-analysis",
        "local_filename": "tpg23-08_nsw-government-guide-to-cost-benefit-analysis_202304_20250526.pdf",
        "version_label": "Apr 2023",
    },
    {
        "doc_id": "new_cba_req",
        "display_name": "New CBA Requirements for Transport Projects",
        "source_page_url": "https://www.transport.nsw.gov.au/projects/project-delivery-requirements/project-cost-benefit-analysis",
        "local_filename": "New CBA requirements for Transport projects 2025_1_20250526.pdf",
        "version_label": "2025",
    },
]


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(data_dir: str) -> None:
    """Create the table and seed initial records if empty."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id          TEXT PRIMARY KEY,
                display_name    TEXT NOT NULL,
                source_page_url TEXT NOT NULL,
                local_path      TEXT NOT NULL,
                sha256          TEXT,
                version_label   TEXT,
                last_checked    TIMESTAMP,
                last_updated    TIMESTAMP
            )
        """)
        conn.commit()

        # Seed if empty
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if count == 0:
            for doc in INITIAL_DOCUMENTS:
                local_path = os.path.join(data_dir, doc["local_filename"])
                sha256 = _hash_file(local_path) if os.path.exists(local_path) else None
                now = datetime.utcnow().isoformat()
                conn.execute(
                    """INSERT INTO documents
                       (doc_id, display_name, source_page_url, local_path,
                        sha256, version_label, last_checked, last_updated)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        doc["doc_id"],
                        doc["display_name"],
                        doc["source_page_url"],
                        local_path,
                        sha256,
                        doc["version_label"],
                        now,
                        now,
                    ),
                )
            conn.commit()


def get_all() -> list[dict]:
    with _get_connection() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY doc_id").fetchall()
        return [dict(r) for r in rows]


def get(doc_id: str) -> Optional[dict]:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None


def mark_updated(
    doc_id: str,
    sha256: str,
    version_label: str,
    local_path: str,
) -> None:
    now = datetime.utcnow().isoformat()
    with _get_connection() as conn:
        conn.execute(
            """UPDATE documents
               SET sha256 = ?, version_label = ?, local_path = ?,
                   last_checked = ?, last_updated = ?
               WHERE doc_id = ?""",
            (sha256, version_label, local_path, now, now, doc_id),
        )
        conn.commit()


def mark_checked(doc_id: str) -> None:
    now = datetime.utcnow().isoformat()
    with _get_connection() as conn:
        conn.execute(
            "UPDATE documents SET last_checked = ? WHERE doc_id = ?",
            (now, doc_id),
        )
        conn.commit()


def _hash_file(path: str) -> Optional[str]:
    """Return SHA-256 hex digest of a file, or None if it doesn't exist."""
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
