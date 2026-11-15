"""
RAG pipeline used by the API, ported from day3_rag/ask.py.

Settings (paths, embedding model, LLM, top-k) are imported from day3_rag/rag_config.py,
so the API always queries the index with the same embedding model that built it.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama

from day3_rag.rag_config import CHROMA_DIR, COLLECTION_NAME, LLM_MODEL, TOP_K, get_embeddings

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
        raise FileNotFoundError("No vector database found. Run `python day3_rag/ingest.py` first.")

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

    # Step 1 (parallel): retrieve chunks and pass the question through unchanged.
    # Step 2 (assign): add the generated answer while keeping the docs as sources.
    return RunnableParallel(docs=retriever, question=RunnablePassthrough()).assign(
        answer=RunnableLambda(to_prompt_inputs) | generate_answer
    )