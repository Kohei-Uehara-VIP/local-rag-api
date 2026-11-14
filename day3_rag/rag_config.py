"""
Shared settings for the Day 3 RAG pipeline.

Both ingest.py (indexing) and ask.py (querying) import from here, so the
embedding model, paths and chunking settings can never drift apart.
"""

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

# ---- Paths (resolved from this file, so scripts work from any working directory) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "local_rag_docs"

# ---- Models ----
# The SAME embedding model must be used for indexing and for querying;
# vectors from different models live in different spaces and cannot be compared.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"

# ---- Chunking and retrieval ----
# 1000 characters ~ 250 tokens, so TOP_K=4 chunks (~1000 tokens) fit easily
# into the model's 4096-token context window together with the question.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the local sentence-transformers embedding model."""
    # Normalized vectors make distance scores comparable across queries.
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )