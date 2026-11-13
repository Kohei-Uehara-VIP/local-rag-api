# local-rag-api

A fully local Retrieval-Augmented Generation (RAG) microservice built only with free, open-source tools — no paid APIs and no cloud services.

> Prime One Global — ML Engineer Internship, Week 11: Applied Generative AI and Open-Source LLMs

## Tech Stack

| Layer | Tool |
|---|---|
| Local LLM runtime | Ollama (`llama3.2:3b`) |
| Orchestration | LangChain (Day 2+) |
| Embeddings | sentence-transformers (Day 3+) |
| Vector store | ChromaDB (Day 3+) |
| API | FastAPI (Day 5) |

## Progress

- [x] Day 1 — Run a local LLM with Ollama (CLI + Python)
- [ ] Day 2 — LLM orchestration with LangChain
- [ ] Day 3 — Local RAG pipeline
- [ ] Day 4 — Hugging Face Transformers (vision)
- [ ] Day 5 — GenAI microservice with FastAPI

## Requirements

- macOS on Apple Silicon (tested on MacBook Pro M1, 16 GB RAM)
- [Ollama](https://ollama.com) app running locally
- Conda (Miniconda)

## Setup

```bash
ollama pull llama3.2:3b
conda create -n local-rag-api python=3.11 -y
conda activate local-rag-api
python -m pip install -r requirements.txt
```

## Day 1: Run a Local LLM

```bash
python day1_ollama/hello_ollama.py
python day1_ollama/hello_ollama.py "What is the capital of Japan? Answer in one word."
```

Measured on MacBook Pro (M1, 16 GB):

| Model | Download size | Generation speed |
|---|---|---|
| `llama3.2:3b` | 2.0 GB | ~21 tokens/s |
| `llama3.1:8b` | 4.9 GB | ~10 tokens/s |

- Verified to work fully offline (Wi-Fi turned off).
- Runs on the Apple M1 GPU via Metal (`ollama ps` shows `100% GPU`).
- `llama3.2:3b` was chosen as the main model: about 2x faster than the 8B model, which matters for an interactive API.