"""
RAG pipeline using BM25 retrieval — fully local, no embedding API needed.

The BM25 index is persisted to disk so it survives restarts. A module-level
asyncio.Lock ensures only one rebuild runs at a time under concurrent requests.
Call invalidate_and_rebuild() after a document update to atomically swap in
a fresh index.
"""

import asyncio
import json
import logging
import os
import pickle
import re
from datetime import datetime
from typing import Optional

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
INDEX_DIR = os.environ.get("INDEX_DIR", os.path.join(os.path.dirname(__file__), "bm25_index"))
INDEX_FILE = os.path.join(INDEX_DIR, "bm25.pkl")
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.json")

# ──────────────────────────────────────────────────────────────────────────────
# Module-level shared state
# ──────────────────────────────────────────────────────────────────────────────

_bm25: Optional[BM25Okapi] = None
_chunks: list[str] = []
_index_lock = asyncio.Lock()
_last_built: Optional[datetime] = None
_chunk_count: int = 0

SYSTEM_PROMPT = """
You are an expert Cost-Benefit Analysis (CBA) consultant and teacher with deep knowledge of government CBA guidelines and best practices.

INTERACTION GUIDELINES:
- For casual greetings (like "Hi", "Hello"), respond warmly and briefly introduce what you can help with
- For general conversation, be friendly and natural
- For CBA-specific questions, provide detailed, expert guidance based on the provided documents

CBA ANALYSIS INSTRUCTIONS (when relevant):
1. **Always prioritize the provided PDF context** - Base your answers primarily on the information from the CBA guide documents
2. **Be specific and detailed** - When referencing numbers, parameters, tables, or formulas, present them clearly with proper formatting
3. **Cite your sources** - When possible, reference which document or section your information comes from
4. **Structure your responses** - Use headings, bullet points, and clear formatting to make answers easy to read
5. **Handle uncertainty honestly** - If information isn't clearly available in the guides, say so explicitly
6. **Provide practical guidance** - Add brief explanations to help users understand and apply the material
7. **Consider context** - Use the chat history to provide consistent, connected responses

RESPONSE FORMAT FOR CBA QUESTIONS:
- Use clear headings and subheadings
- Present numerical data in tables or organised lists
- Include specific page references when available
- Provide actionable guidance where appropriate

If a CBA answer cannot be found in the provided documents, respond with: "I couldn't find this specific information in the provided CBA guides. However, [provide any relevant general guidance if appropriate]."
"""


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


async def get_index() -> tuple[BM25Okapi, list[str]]:
    """Return the cached BM25 index and chunks, building if needed."""
    global _bm25, _chunks
    if _bm25 is None:
        await _build_and_cache(force_rebuild=False)
    return _bm25, _chunks


async def invalidate_and_rebuild() -> int:
    """Force a full rebuild of the index. Returns new chunk count."""
    count = await _build_and_cache(force_rebuild=True)
    logger.info("BM25 index rebuilt: %d chunks", count)
    return count


def get_chunk_count() -> int:
    return _chunk_count


def get_last_built() -> Optional[datetime]:
    return _last_built


# ──────────────────────────────────────────────────────────────────────────────
# Internal build logic
# ──────────────────────────────────────────────────────────────────────────────


async def _build_and_cache(force_rebuild: bool) -> int:
    global _bm25, _chunks, _last_built, _chunk_count

    async with _index_lock:
        if not force_rebuild and os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
            try:
                bm25, chunks = await asyncio.to_thread(_load_index)
                _bm25 = bm25
                _chunks = chunks
                _chunk_count = len(chunks)
                _last_built = datetime.utcnow()
                logger.info("Loaded BM25 index from disk: %d chunks", _chunk_count)
                return _chunk_count
            except Exception as e:
                logger.warning("Failed to load index from disk, rebuilding: %s", e)

        chunks = await asyncio.to_thread(_extract_all_chunks)
        if not chunks:
            raise RuntimeError("No text chunks extracted from PDFs — cannot build index")

        bm25 = await asyncio.to_thread(_build_bm25, chunks)

        os.makedirs(INDEX_DIR, exist_ok=True)
        await asyncio.to_thread(_save_index, bm25, chunks)

        _bm25 = bm25
        _chunks = chunks
        _chunk_count = len(chunks)
        _last_built = datetime.utcnow()
        logger.info("Built and saved BM25 index: %d chunks", _chunk_count)
        return _chunk_count


def _build_bm25(chunks: list[str]) -> BM25Okapi:
    tokenized = [_tokenize(c) for c in chunks]
    return BM25Okapi(tokenized)


def _save_index(bm25: BM25Okapi, chunks: list[str]) -> None:
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(bm25, f)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)


def _load_index() -> tuple[BM25Okapi, list[str]]:
    with open(INDEX_FILE, "rb") as f:
        bm25 = pickle.load(f)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return bm25, chunks


def _extract_all_chunks() -> list[str]:
    os.makedirs(DATA_DIR, exist_ok=True)
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        logger.error("No PDF files found in %s", DATA_DIR)
        return []

    full_text = ""
    for fn in pdf_files:
        path = os.path.join(DATA_DIR, fn)
        if not _is_extractable(path):
            logger.warning("Skipping non-extractable PDF: %s", fn)
            continue
        doc_text = _extract_pdf(path, fn)
        if doc_text:
            full_text += f"\n=== DOCUMENT: {fn} ===\n{doc_text}\n"

    if not full_text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    )
    return splitter.split_text(full_text)


def _extract_pdf(path: str, filename: str) -> str:
    doc_text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                txt = page.extract_text()
                if txt:
                    cleaned = _clean_text(txt)
                    if cleaned:
                        doc_text += f"[Source: {filename}, Page {page_num + 1}]\n{cleaned}\n\n"
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
    return doc_text


def _is_extractable(pdf_path: str) -> bool:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:3]:
                txt = page.extract_text()
                if txt and len(txt.strip()) > 50:
                    return True
    except Exception:
        pass
    return False


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────────────────────


def get_relevant_context(
    bm25: BM25Okapi,
    chunks: list[str],
    query: str,
    max_chunks: int = 20,
) -> list[str]:
    """
    BM25 retrieval with multi-query expansion.
    Returns up to max_chunks deduplicated text chunks.
    """
    all_indices: list[int] = []

    # Primary query
    all_indices.extend(_bm25_top_k(bm25, query, k=12))

    # Expanded query
    expanded = expand_query(query)
    if expanded != query:
        all_indices.extend(_bm25_top_k(bm25, expanded, k=6))

    # Key term queries
    for term in _key_terms(query)[:2]:
        all_indices.extend(_bm25_top_k(bm25, term, k=4))

    # Deduplicate preserving order
    seen: set[int] = set()
    result: list[str] = []
    for idx in all_indices:
        if idx not in seen and len(result) < max_chunks:
            seen.add(idx)
            result.append(chunks[idx])

    return result[:max_chunks]


def _bm25_top_k(bm25: BM25Okapi, query: str, k: int) -> list[int]:
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    # Only return indices with a positive score
    return [i for i in top_k if scores[i] > 0]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


def expand_query(query: str) -> str:
    synonyms = {
        "cost": ["expense", "expenditure", "financial"],
        "benefit": ["advantage", "gain", "value"],
        "analysis": ["assessment", "evaluation", "review"],
        "discount": ["present value", "npv", "discounting"],
        "risk": ["uncertainty", "sensitivity", "probability"],
        "social": ["societal", "community", "public"],
        "economic": ["financial", "monetary", "fiscal"],
    }
    extras: list[str] = []
    for word in query.lower().split():
        if word in synonyms:
            extras.extend(synonyms[word][:2])
    if extras:
        return f"{query} {' '.join(extras[:3])}"
    return query


STOP_WORDS = {"what", "how", "when", "where", "should", "would", "could", "which", "that", "this"}


def _key_terms(query: str) -> list[str]:
    return [w for w in query.lower().split() if len(w) > 4 and w not in STOP_WORDS]


def is_casual_greeting(text: str) -> bool:
    patterns = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening",
                "how are you", "what's up", "sup", "greetings", "yo"]
    text_lower = text.lower().strip()
    if len(text_lower) < 20:
        return any(p in text_lower for p in patterns)
    return False
