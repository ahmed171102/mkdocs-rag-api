# MkDocs RAG API

Retrieval-augmented Q&A over **MkDocs documentation**: chunk → embed → **ChromaDB** → answer with **Google Gemini**, served by **FastAPI**.

## Setup

From this repo root (`mkdocs-rag-api`):

```bash
cd mkdocs_rag
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r ../requirements.txt
copy ..\.env.example .env
# Edit .env and set GOOGLE_API_KEY=...
```

1. Build the local Chroma DB under `mkdocs_rag/db/` (gitignored):

   ```bash
   python ingest.py --docs docs --db db
   ```

   (or run `Embedding.ipynb` if you prefer the notebook version.)
2. Start the API:

```bash
cd mkdocs_rag
uvicorn app:app --reload
```

## API routes

- `GET /health`
- `GET /sample-questions`
- `POST /ask`
- `POST /context`
- `POST /ask-with-context` (if enabled in `app.py`)

## Layout

```
mkdocs_rag/
  app.py              FastAPI
  rag.py              retrieve + generate
  Embedding.ipynb     ingest docs into Chroma
  docs/               MkDocs corpus
requirements.txt
.env.example
```
