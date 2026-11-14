"""
Day 3 (indexing): load PDFs, split them into chunks, embed the chunks and
store them in a local ChromaDB vector database.

Run once, and again whenever the PDFs in data/docs/ change:
    python day3_rag/ingest.py
"""

import shutil
import time

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DOCS_DIR,
    get_embeddings,
)


def load_pdfs() -> list[Document]:
    """Load every PDF in DOCS_DIR as a list of page-level Documents."""
    pdf_paths = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDF files found in {DOCS_DIR}")

    pages: list[Document] = []
    for path in pdf_paths:
        docs = PyPDFLoader(str(path)).load()  # one Document per page
        for doc in docs:
            # Keep only the file name: the full path would leak local folder names into answers.
            doc.metadata["source"] = path.name
        print(f"Loaded {len(docs):>3} pages from {path.name}")
        pages.extend(docs)
    return pages


def split_pages(pages: list[Document]) -> list[Document]:
    """Split pages into overlapping chunks small enough for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,  # records where each chunk starts within its page
    )
    return splitter.split_documents(pages)


def build_vector_store(chunks: list[Document]) -> Chroma:
    """Embed the chunks and persist them to ChromaDB, rebuilding from scratch."""
    # Delete any previous index so that re-running does not store duplicate chunks.
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )


def main() -> None:
    start = time.perf_counter()

    pages = load_pdfs()
    chunks = split_pages(pages)
    print(f"Split {len(pages)} pages into {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    store = build_vector_store(chunks)
    elapsed = time.perf_counter() - start
    print(f"Stored {len(chunks)} chunks in {CHROMA_DIR.name}/ ({elapsed:.1f} s)")

    # Sanity check: retrieval only (no LLM yet). Lower distance = more similar.
    query = "Which retriever does the RAG model use?"
    print(f"\nSanity check - top 2 chunks for: {query!r}")
    for doc, distance in store.similarity_search_with_score(query, k=2):
        page = doc.metadata.get("page", 0) + 1  # PyPDFLoader pages are 0-based
        preview = doc.page_content[:150].replace("\n", " ")
        print(f"- {doc.metadata['source']} p.{page} (distance={distance:.3f}): {preview}...")


if __name__ == "__main__":
    main()