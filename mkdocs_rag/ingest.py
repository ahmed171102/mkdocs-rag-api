"""
Command-line ingestion for the MkDocs RAG pipeline.

Equivalent to running Embedding.ipynb, but scriptable/CI-friendly:

    python ingest.py --docs docs --db db

Chunks every markdown file under --docs, embeds with Gemini, and stores
in a local persistent ChromaDB collection ("MkDocsRAG") so app.py / rag.py
can query it.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_chunk(chunk: str) -> str:
    chunk = chunk.strip()
    chunk = re.sub(r"\n{3,}", "\n\n", chunk)
    chunk = re.sub(r"\n(#{1,6})\s*([^\n]+)", r"\n\n\1 \2\n", chunk)
    chunk = re.sub(r"\n{3,}", "\n\n", chunk)
    return chunk.strip()


def chunk_document(
    content: str,
    metadata: Dict[str, Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    raw_chunks = splitter.split_text(content)
    cleaned = [clean_chunk(c) for c in raw_chunks]
    cleaned = [c for c in cleaned if len(c) > 50]

    return [
        {
            "text": c,
            "metadata": {**metadata, "chunk_index": i},
        }
        for i, c in enumerate(cleaned)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest MkDocs markdown into ChromaDB")
    parser.add_argument("--docs", default="docs", help="Path to markdown docs folder")
    parser.add_argument("--db", default="db", help="Path to persistent Chroma DB folder")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Copy .env.example to .env and set your key."
        )

    docs_path = Path(args.docs)
    md_files = sorted(docs_path.rglob("*.md"))
    if not md_files:
        raise SystemExit(f"No markdown files found under {docs_path.resolve()}")

    client = chromadb.PersistentClient(path=args.db)
    google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=api_key)
    # Fresh collection each run for reproducibility.
    try:
        client.delete_collection("MkDocsRAG")
    except Exception:
        pass
    collection = client.get_or_create_collection(name="MkDocsRAG", embedding_function=google_ef)

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for path in md_files:
        content = path.read_text(encoding="utf-8", errors="ignore")
        rel_path = str(path.relative_to(docs_path))
        chunks = chunk_document(
            content,
            metadata={"file_path": rel_path},
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        for chunk in chunks:
            chunk_id = f"{rel_path}::{chunk['metadata']['chunk_index']}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])

    if not ids:
        raise SystemExit("No chunks produced — check docs folder contents.")

    # Chroma has a per-call batch limit on some backends; insert in batches.
    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"Ingested {len(ids)} chunks from {len(md_files)} files into '{args.db}'.")


if __name__ == "__main__":
    main()
