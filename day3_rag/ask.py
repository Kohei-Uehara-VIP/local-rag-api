"""
Day 3 (querying): ask a question about the indexed PDFs.

Retrieves the most relevant chunks from ChromaDB, asks the local Ollama model to
answer using ONLY those chunks, and prints the answer together with its sources.

Usage (from the project root, after running day3_rag/ingest.py):
    python day3_rag/ask.py "What is retrieval-augmented generation?"
    python day3_rag/ask.py "What is retrieval-augmented generation?" --compare
"""

import argparse
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama

from rag_config import CHROMA_DIR, COLLECTION_NAME, LLM_MODEL, TOP_K, get_embeddings

DEFAULT_QUESTION = "What is retrieval-augmented generation (RAG)?"

# Grounding instructions: answer only from the retrieved context, and admit when it is not there.
SYSTEM_PROMPT = """You are an assistant that answers questions about research papers.
Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly:
"I don't know based on the provided documents."
Keep the answer concise (3-5 sentences).

Context:
{context}"""


def format_docs(docs: list[Document]) -> str:
    """Join retrieved chunks into one context string, each labelled with its source."""
    return "\n\n---\n\n".join(
        f"[{doc.metadata['source']} p.{doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
        for doc in docs
    )


def load_retriever() -> VectorStoreRetriever:
    """Open the persisted ChromaDB collection and expose it as a retriever."""
    if not CHROMA_DIR.exists():
        raise SystemExit("No vector database found. Run `python day3_rag/ingest.py` first.")

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),  # must match the model used in ingest.py
        persist_directory=str(CHROMA_DIR),
    )
    return store.as_retriever(search_kwargs={"k": TOP_K})


def build_rag_chain(retriever: VectorStoreRetriever):
    """Build an LCEL chain that returns {"question", "docs", "answer"}."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    generate_answer = prompt | llm | StrOutputParser()

    def to_prompt_inputs(state: dict) -> dict:
        return {"context": format_docs(state["docs"]), "question": state["question"]}

    # Step 1 (parallel): search ChromaDB with the question, and pass the question through unchanged.
    #   -> {"docs": [Document, ...], "question": "..."}
    # Step 2 (assign): add the generated answer while keeping the docs for citing sources.
    #   -> {"docs": [...], "question": "...", "answer": "..."}
    return RunnableParallel(docs=retriever, question=RunnablePassthrough()).assign(
        answer=RunnableLambda(to_prompt_inputs) | generate_answer
    )


def ask_without_rag(question: str) -> str:
    """Ask the same model with no retrieved context, for comparison."""
    prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
    chain = prompt | ChatOllama(model=LLM_MODEL, temperature=0) | StrOutputParser()
    return chain.invoke({"question": question})


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about the indexed PDFs.")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION, help="question to ask")
    parser.add_argument("--compare", action="store_true", help="also answer without RAG")
    args = parser.parse_args()

    chain = build_rag_chain(load_retriever())

    print(f"Question: {args.question}\n")

    start = time.perf_counter()
    result = chain.invoke(args.question)
    elapsed = time.perf_counter() - start

    print(f"=== Answer with RAG ({LLM_MODEL}, {elapsed:.1f} s) ===")
    print(result["answer"])

    print("\n=== Sources (retrieved chunks) ===")
    for i, doc in enumerate(result["docs"], start=1):
        page = doc.metadata.get("page", 0) + 1  # PyPDFLoader pages are 0-based
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"[{i}] {doc.metadata['source']} p.{page}: {preview}...")

    if args.compare:
        start = time.perf_counter()
        plain_answer = ask_without_rag(args.question)
        elapsed = time.perf_counter() - start
        print(f"\n=== Answer WITHOUT RAG, for comparison ({elapsed:.1f} s) ===")
        print(plain_answer)


if __name__ == "__main__":
    main()