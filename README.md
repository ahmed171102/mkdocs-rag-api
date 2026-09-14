# MkDocs RAG API

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-6E4AFF)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A **Retrieval-Augmented Generation** service that answers natural-language questions about MkDocs documentation: chunk → embed → store in **ChromaDB** → retrieve → answer with **Google Gemini**, served over **FastAPI**.

## Overview

The MkDocs docs corpus is split into overlapping chunks, embedded with Gemini's embedding model, and stored in a local persistent Chroma collection. At query time, the top-k most relevant chunks are retrieved and passed to Gemini along with the user's question to produce a grounded answer.

## Features

- Markdown-aware chunking with cleanup (heading spacing, collapsed blank lines)
- Persistent local vector store (ChromaDB, no external DB server needed)
- FastAPI service with health check and sample-questions endpoints
- Both a **notebook** (`Embedding.ipynb`) and a **CLI script** (`ingest.py`) for building the vector store

## Tech stack

Python · FastAPI · ChromaDB · Google Generative AI (Gemini) · LangChain text splitters

## Project structure

```
mkdocs_rag/
  app.py              FastAPI app (routes)
  rag.py              Retrieval + Gemini answer generation
  ingest.py            CLI ingestion script (chunk + embed + store)
  Embedding.ipynb      Notebook version of the ingestion pipeline
  docs/                MkDocs markdown corpus
  .env.example         GOOGLE_API_KEY placeholder
requirements.txt
```

## API endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Health check |
| GET | `/sample-questions` | Example questions to try |
| POST | `/ask` | Ask a question, get a grounded answer |
| POST | `/context` | Retrieve raw context chunks for a query |

## Getting started

```bash
cd mkdocs_rag
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r ../requirements.txt
copy ..\.env.example .env
# Edit .env: set GOOGLE_API_KEY
```

Build the vector store (either approach):

```bash
python ingest.py --docs docs --db db
# or: run Embedding.ipynb
```

Start the API:

```bash
uvicorn app:app --reload
```

## Known limitations

- The vector DB (`db/`) is gitignored — must be rebuilt locally before first use.
- Requires a valid `GOOGLE_API_KEY`; the app now fails fast with a clear error if it's missing.
- No caching layer — repeated identical questions re-query the LLM every time.

## Possible next steps

- Add response caching for repeated queries
- Add a `/ask-with-context` endpoint that returns both the answer and its sources in one call
- Containerize with Docker for easier deployment

## License

MIT — see `LICENSE`.
