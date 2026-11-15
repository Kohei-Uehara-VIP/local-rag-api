"""
Day 5: GenAI microservice that exposes the local RAG pipeline as a REST API.

Run from the project root (with the `local-rag-api` conda env activated and the Ollama app running):
    uvicorn app.main:app --port 8000
Then open http://127.0.0.1:8000/docs to try it.
"""

import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from ollama import ResponseError

from app.rag import LLM_MODEL, build_rag_chain, load_retriever
from app.schemas import ChatRequest, ChatResponse, Source

PREVIEW_CHARS = 200
state: dict = {}

# Errors meaning "the Ollama server is not reachable".
# ChatOllama streams its responses, and on that code path the underlying httpx
# ConnectError is raised as-is instead of being wrapped in Python's ConnectionError.
OLLAMA_UNREACHABLE_ERRORS = (ConnectionError, httpx.ConnectError)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the RAG chain once at startup (loading the embedding model is slow)."""
    try:
        state["chain"] = build_rag_chain(load_retriever())
    except FileNotFoundError as e:
        # Start anyway so /health can explain the problem; /chat will return 503.
        state["error"] = str(e)
    yield
    state.clear()


app = FastAPI(
    title="Local RAG API",
    version="1.0.0",
    description="Ask questions about local PDFs. Runs fully offline with Ollama + ChromaDB.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Liveness check that also reports whether the RAG pipeline is ready."""
    return {
        "status": "ok",
        "rag_ready": "chain" in state,
        "llm_model": LLM_MODEL,
        "error": state.get("error"),
    }


# A plain `def` (not `async def`): chain.invoke() is blocking, so FastAPI runs it in a
# worker thread instead of blocking the event loop.
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Retrieve relevant chunks from ChromaDB and answer with the local Ollama model."""
    chain = state.get("chain")
    if chain is None:
        raise HTTPException(status_code=503, detail=state.get("error", "RAG pipeline is not ready"))

    start = time.perf_counter()
    try:
        result = chain.invoke(request.query)
    except OLLAMA_UNREACHABLE_ERRORS:
        raise HTTPException(
            status_code=503, detail="Cannot reach the Ollama server. Is the Ollama app running?"
        ) from None
    except ResponseError as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e.error}") from None
    latency = time.perf_counter() - start

    sources = [
        Source(
            source=doc.metadata["source"],
            page=doc.metadata.get("page", 0) + 1,  # PyPDFLoader pages are 0-based
            preview=doc.page_content[:PREVIEW_CHARS].replace("\n", " "),
        )
        for doc in result["docs"]
    ]
    return ChatResponse(
        query=request.query,
        answer=result["answer"],
        sources=sources,
        model=LLM_MODEL,
        latency_seconds=round(latency, 2),
    )