"""
FastAPI backend for the BCCoE CBA Guide Assistant.

Endpoints:
  POST /api/chat               — SSE streaming chat
  GET  /api/documents          — list tracked documents
  POST /api/documents/refresh  — manual document refresh trigger
  GET  /api/documents/status   — scheduler status + update flag
  GET  /api/health             — liveness check

On startup:
  - Initialises the document registry (SQLite)
  - Builds / loads the FAISS vector index
  - Starts APScheduler for weekly document update checks
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
from datetime import datetime, timedelta
from typing import AsyncGenerator

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import anthropic
from pydantic import BaseModel

import document_registry as registry
import rag_pipeline as rag
from document_updater import DocumentUpdater

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ──────────────────────────────────────────────────────────────────────────────
# Global state shared across requests
# ──────────────────────────────────────────────────────────────────────────────

_scheduler = AsyncIOScheduler()
_updater: DocumentUpdater = None  # initialised in lifespan
_update_available: bool = False
_recently_updated_docs: list[str] = []
_last_check: datetime = None
_next_check: datetime = None


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ──────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _updater, _last_check, _next_check

    logger.info("=== BCCoE CBA Assistant starting up ===")

    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — LLM calls will fail")

    # Initialise document registry
    registry.init_db(DATA_DIR)
    logger.info("Document registry initialised")

    # Build / load BM25 index
    logger.info("Building BM25 index from PDFs…")
    await rag.get_index()
    logger.info("BM25 index ready")

    # Set up document updater
    _updater = DocumentUpdater(data_dir=DATA_DIR)

    # Schedule weekly check
    _scheduler.add_job(
        _weekly_check_job,
        trigger=IntervalTrigger(weeks=1),
        id="weekly_doc_check",
        next_run_time=None,  # don't run immediately on startup
        replace_existing=True,
    )
    _scheduler.start()
    _last_check = datetime.utcnow()
    _next_check = datetime.utcnow() + timedelta(weeks=1)
    logger.info("Scheduler started. Next check: %s", _next_check.isoformat())

    yield

    _scheduler.shutdown(wait=False)
    logger.info("=== BCCoE CBA Assistant shut down ===")


async def _weekly_check_job():
    global _update_available, _recently_updated_docs, _last_check, _next_check
    logger.info("Running weekly document check…")
    _last_check = datetime.utcnow()
    _next_check = datetime.utcnow() + timedelta(weeks=1)
    updated = await _updater.check_all()
    if updated:
        logger.info("Updated documents: %s — rebuilding index…", updated)
        await rag.invalidate_and_rebuild()
        _update_available = True
        _recently_updated_docs = updated
        logger.info("Index rebuild complete")
    else:
        logger.info("No document updates found")


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="BCCoE CBA Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production (e.g. your Vercel domain)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────────────────────────────────────


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str = "claude-sonnet-4-6"
    max_chunks: int = 18


class DocumentInfo(BaseModel):
    doc_id: str
    display_name: str
    source_page_url: str
    local_path: str
    version_label: str | None
    last_updated: str | None
    last_checked: str | None


class RefreshResult(BaseModel):
    updated: list[str]
    rebuilt: bool
    message: str


class DocumentStatus(BaseModel):
    update_available: bool
    recently_updated_docs: list[str]
    last_check: str | None
    next_check: str | None
    chunk_count: int
    last_built: str | None


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/api/documents", response_model=list[DocumentInfo])
async def list_documents():
    docs = registry.get_all()
    return [
        DocumentInfo(
            doc_id=d["doc_id"],
            display_name=d["display_name"],
            source_page_url=d["source_page_url"],
            local_path=d["local_path"],
            version_label=d.get("version_label"),
            last_updated=d.get("last_updated"),
            last_checked=d.get("last_checked"),
        )
        for d in docs
    ]


@app.post("/api/documents/refresh", response_model=RefreshResult)
async def manual_refresh():
    global _update_available, _recently_updated_docs, _last_check, _next_check

    if _updater is None:
        raise HTTPException(status_code=503, detail="Updater not initialised")

    _last_check = datetime.utcnow()
    updated = await _updater.check_all()
    rebuilt = False

    if updated:
        await rag.invalidate_and_rebuild()
        rebuilt = True
        _update_available = True
        _recently_updated_docs = updated

    return RefreshResult(
        updated=updated,
        rebuilt=rebuilt,
        message=(
            f"Updated {len(updated)} document(s) and rebuilt index."
            if updated
            else "All documents are up to date."
        ),
    )


@app.get("/api/documents/status", response_model=DocumentStatus)
async def document_status():
    return DocumentStatus(
        update_available=_update_available,
        recently_updated_docs=_recently_updated_docs,
        last_check=_last_check.isoformat() if _last_check else None,
        next_check=_next_check.isoformat() if _next_check else None,
        chunk_count=rag.get_chunk_count(),
        last_built=rag.get_last_built().isoformat() if rag.get_last_built() else None,
    )


@app.post("/api/documents/status/acknowledge")
async def acknowledge_update():
    """Dismiss the update notification banner."""
    global _update_available, _recently_updated_docs
    _update_available = False
    _recently_updated_docs = []
    return {"acknowledged": True}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    user_message = request.messages[-1].content
    if not user_message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    return StreamingResponse(
        _stream_chat(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Streaming chat logic
# ──────────────────────────────────────────────────────────────────────────────


async def _stream_chat(request: ChatRequest) -> AsyncGenerator[str, None]:
    user_message = request.messages[-1].content

    # Handle casual greetings without RAG
    if rag.is_casual_greeting(user_message):
        greeting = (
            "Hi there! I'm your CBA Guide Assistant, here to help you with "
            "cost-benefit analysis questions. I have access to comprehensive CBA "
            "guidelines including ATAP T2, TfNSW, and NSW Government guides.\n\n"
            "Feel free to ask me about:\n"
            "- CBA methodology and principles\n"
            "- Calculating NPV, discount rates, and other metrics\n"
            "- Identifying and valuing costs and benefits\n"
            "- Sensitivity and risk analysis\n"
            "- Best practices for government CBAs"
        )
        yield _sse("token", greeting)
        yield _sse("done", "")
        return

    try:
        bm25, chunks = await rag.get_index()
    except Exception as e:
        logger.error("Failed to get BM25 index: %s", e)
        yield _sse("error", "Search index not available. Please try again shortly.")
        return

    # Retrieve context
    docs = rag.get_relevant_context(bm25, chunks, user_message, max_chunks=request.max_chunks)
    if not docs:
        yield _sse("token", "I couldn't find relevant information in the CBA guides for your question. Could you try rephrasing it?")
        yield _sse("done", "")
        return

    context = "\n\n".join(
        f"--- Context Chunk {i} ---\n{doc}"
        for i, doc in enumerate(docs, 1)
    )

    # Build chat history string (last 6 messages, excluding current)
    history_msgs = request.messages[-7:-1]
    chat_history = ""
    for msg in history_msgs:
        if msg.role == "user":
            chat_history += f"User: {msg.content}\n"
        else:
            chat_history += f"Assistant: {msg.content[:200]}…\n"

    # Build messages for Anthropic Claude API
    system_content = (
        f"{rag.SYSTEM_PROMPT}\n\n"
        f"RELEVANT CONTEXT FROM CBA GUIDES:\n{context}"
    )
    claude_messages = []
    if chat_history:
        claude_messages.append({"role": "user", "content": f"[Previous conversation]\n{chat_history}"})
        claude_messages.append({"role": "assistant", "content": "Understood, I have noted the previous conversation."})
    claude_messages.append({"role": "user", "content": user_message})

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    # Stream tokens
    try:
        async with client.messages.stream(
            model=request.model,
            system=system_content,
            messages=claude_messages,
            temperature=1,
            max_tokens=2000,
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    yield _sse("token", text)
    except Exception as e:
        logger.error("LLM streaming error: %s", e)
        yield _sse("error", f"Error generating response: {str(e)}")
        return

    yield _sse("done", "")


def _sse(event: str, data: str) -> str:
    # Escape newlines in data so the SSE frame stays valid
    safe_data = data.replace("\n", "\\n")
    return f"event: {event}\ndata: {safe_data}\n\n"
