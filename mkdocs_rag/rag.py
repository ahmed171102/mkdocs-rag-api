from chromadb import PersistentClient
from google import genai
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
from typing import Dict
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError(
        "GOOGLE_API_KEY is missing. Copy .env.example to mkdocs_rag/.env and set your key."
    )

# Initialize ChromaDB
chroma_client = PersistentClient(path="db/")
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=api_key)

# Use get_or_create to prevent "ImportError" crashes if DB is missing
collection = chroma_client.get_or_create_collection(name="MkDocsRAG", embedding_function=google_ef)
client = genai.Client(api_key=api_key)

def get_answer(prompt: str, n_results: int = 8) -> str:
    
    # Query the vector database
    results = collection.query(
        query_texts=[prompt],
        n_results=n_results
    )
    
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0] if "distances" in results else [0] * len(documents)
    
    # Build context from retrieved documents
    context_parts = []
    for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
        source = metadata.get('file_path', 'Unknown')
        context_parts.append(f"[Source: {source}]\n{doc}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Build prompt for Gemini
    full_prompt = f"""You are a helpful assistant specialized in MkDocs documentation.

Context from MkDocs documentation:
{context}

Question: {prompt}

Provide a clear, accurate answer based on the context above:"""

    # Generate answer using Gemini
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=full_prompt
    )
    
    return response.text.strip()

def get_context_for_question(prompt: str, n_results: int = 5) -> Dict:
    
    results = collection.query(
        query_texts=[prompt],
        n_results=n_results
    )
    
    return {
        'question': prompt,
        'documents': results["documents"][0],
        'metadatas': results["metadatas"][0],
        'distances': results.get("distances", [[]])[0] if "distances" in results else []
    }
