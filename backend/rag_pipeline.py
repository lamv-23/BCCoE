"""
RAG pipeline: vectorstore management and retrieval logic.

The FAISS index is persisted to disk so it survives restarts. A module-level
asyncio.Lock ensures only one rebuild runs at a time even under concurrent
requests.  After a document update, call invalidate_and_rebuild() to
atomically swap in a fresh index.
"""

import asyncio
import logging
import os
import re
from datetime import datetime
from typing import Optional

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
FAISS_DIR = os.environ.get("FAISS_DIR", os.path.join(os.path.dirname(__file__), "faiss_index"))

# ──────────────────────────────────────────────────────────────────────────────
# Module-level shared state
# ──────────────────────────────────────────────────────────────────────────────

_vectorstore: Optional[FAISS] = None
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


async def get_vectorstore() -> FAISS:
    """Return the cached vectorstore, building it first if needed."""
    global _vectorstore
    if _vectorstore is None:
        await _build_and_cache(force_rebuild=False)
    return _vectorstore


async def invalidate_and_rebuild() -> int:
    """Force a full rebuild of the index. Returns new chunk count."""
    count = await _build_and_cache(force_rebuild=True)
    logger.info("Index rebuilt: %d chunks", count)
    return count


def get_chunk_count() -> int:
    return _chunk_count


def get_last_built() -> Optional[datetime]:
    return _last_built


# ──────────────────────────────────────────────────────────────────────────────
# Internal build logic
# ──────────────────────────────────────────────────────────────────────────────


async def _build_and_cache(force_rebuild: bool) -> int:
    global _vectorstore, _last_built, _chunk_count

    async with _index_lock:
        # Try loading from disk first (unless forcing rebuild)
        if not force_rebuild and _faiss_index_exists():
            try:
                embeddings = _make_embeddings()
                vs = await asyncio.to_thread(
                    FAISS.load_local, FAISS_DIR, embeddings, allow_dangerous_deserialization=True
                )
                _vectorstore = vs
                _last_built = datetime.utcnow()
                # We don't know exact chunk count from disk load, set a sentinel
                _chunk_count = -1
                logger.info("Loaded FAISS index from disk (%s)", FAISS_DIR)
                return _chunk_count
            except Exception as e:
                logger.warning("Failed to load FAISS from disk, rebuilding: %s", e)

        # Build from PDFs
        chunks = await asyncio.to_thread(_extract_all_chunks)
        if not chunks:
            raise RuntimeError("No text chunks extracted from PDFs — cannot build index")

        embeddings = _make_embeddings()
        vs = await asyncio.to_thread(FAISS.from_texts, chunks, embeddings)

        # Persist to disk
        os.makedirs(FAISS_DIR, exist_ok=True)
        await asyncio.to_thread(vs.save_local, FAISS_DIR)

        _vectorstore = vs
        _chunk_count = len(chunks)
        _last_built = datetime.utcnow()
        logger.info("Built and saved FAISS index: %d chunks", _chunk_count)
        return _chunk_count


def _faiss_index_exists() -> bool:
    return os.path.exists(os.path.join(FAISS_DIR, "index.faiss"))


def _make_embeddings() -> OpenAIEmbeddings:
    api_key = os.environ.get("OPENAI_API_KEY")
    return OpenAIEmbeddings(openai_api_key=api_key)


def _extract_all_chunks() -> list[str]:
    """Scan data/ for PDFs, extract text, split into chunks."""
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


def get_relevant_context(vectorstore: FAISS, query: str, max_chunks: int = 20) -> list:
    """3-strategy hybrid retrieval with deduplication."""
    all_docs = []

    # Strategy 1: direct similarity
    primary = vectorstore.similarity_search(query, k=12)
    all_docs.extend(primary)

    # Strategy 2: expanded query if sparse results
    if len(primary) < 8:
        expanded = expand_query(query)
        if expanded != query:
            all_docs.extend(vectorstore.similarity_search(expanded, k=6))

    # Strategy 3: key term search
    if len(all_docs) < 10:
        for term in _key_terms(query)[:2]:
            all_docs.extend(vectorstore.similarity_search(term, k=4))

    # Deduplicate
    seen: set[int] = set()
    unique: list = []
    for doc in all_docs:
        h = hash(doc.page_content)
        if h not in seen and len(unique) < max_chunks:
            seen.add(h)
            unique.append(doc)

    return unique[:max_chunks]


def expand_query(query: str) -> str:
    synonyms = {
        "cost": ["expense", "expenditure", "financial"],
        "benefit": ["advantage", "gain", "value"],
        "analysis": ["assessment", "evaluation", "review"],
        "discount": ["present value", "NPV", "discounting"],
        "risk": ["uncertainty", "sensitivity", "probability"],
        "social": ["societal", "community", "public"],
        "economic": ["financial", "monetary", "fiscal"],
    }
    extras = []
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
