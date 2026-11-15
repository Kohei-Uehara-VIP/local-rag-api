"""Request and response models for the RAG API (validated by Pydantic)."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """A user's question about the indexed documents."""

    # Strip leading/trailing spaces first, so "   " is rejected by min_length.
    model_config = {
        "str_strip_whitespace": True,
        "json_schema_extra": {
            "examples": [
                {"query": "What is retrieval-augmented generation (RAG)?"},
                {"query": "How many attention heads does the base Transformer model use?"},
            ]
        },
    }

    query: str = Field(min_length=1, max_length=1000, description="Question to ask")


class Source(BaseModel):
    """One retrieved chunk that the answer is based on."""

    source: str = Field(description="PDF file name")
    page: int = Field(description="1-based page number in the PDF")
    preview: str = Field(description="First characters of the retrieved chunk")


class ChatResponse(BaseModel):
    """The generated answer plus the evidence it was based on."""

    query: str
    answer: str
    sources: list[Source]
    model: str = Field(description="Ollama model that generated the answer")
    latency_seconds: float = Field(description="Time spent on retrieval + generation")