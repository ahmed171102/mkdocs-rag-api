from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import get_answer, get_context_for_question
from typing import Optional, List

# 1. Initialize the app FIRST
app = FastAPI(
    title="MkDocs RAG API", 
    version="1.0.0",
    description="RAG system for MkDocs documentation with sample questions endpoint"
)

# 2. Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Define Models (Must come before endpoints)
class Query(BaseModel):
    question: str
    n_results: Optional[int] = 8

class QueryResponse(BaseModel):
    answer: str
    question: str

# 4. Define Endpoints
@app.get("/")
def root():
    return {
        "message": "MkDocs RAG API",
        "endpoints": {
            "/ask": "POST - Ask a question about MkDocs",
            "/context": "POST - Get retrieved context for a question",
            "/health": "GET - Health check"
        }
    }

@app.post("/ask", response_model=QueryResponse)
def ask(query: Query):
    """Ask a question about MkDocs documentation"""
    try:
        answer = get_answer(query.question, query.n_results)
        return QueryResponse(answer=answer, question=query.question)
    except Exception as e:
        # This returns a 500 error with the specific message if something fails
        raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")

@app.post("/context")
def get_context(query: Query):
    """Get retrieved context for a question (for debugging/display)"""
    try:
        context_data = get_context_for_question(query.question, query.n_results)
        return context_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context Error: {str(e)}")

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "MkDocs RAG API"}

@app.get("/sample-questions")
def get_sample_questions():
    sample_questions = [
        "How do I install MkDocs?",
        "What is a configuration file?",
        "How to customize themes?",
        "What are plugins in MkDocs?",
        "How to deploy documentation?",
        "What is the structure of mkdocs.yml?",
        "How do I create a new MkDocs project?",
        "What themes are available in MkDocs?",
        "How to add navigation to my documentation?",
        "What is the difference between mkdocs.yml and mkdocs.yaml?",
        "How to use markdown extensions?",
        "How to build and serve documentation locally?",
        "What are the requirements for running MkDocs?",
        "How to configure search functionality?",
        "How to add custom CSS to my documentation?",
    ]
    return {
        "sample_questions": sample_questions,
        "count": len(sample_questions),
        "usage": "Use these questions with POST /ask endpoint"
    }

@app.post("/ask-with-context")
def ask_with_context(query: Query):
    try:
        answer = get_answer(query.question, query.n_results)
        context_data = get_context_for_question(query.question, query.n_results)
        
        return {
            "question": query.question,
            "answer": answer,
            "retrieved_context": {
                "num_chunks": len(context_data['documents']),
                "chunks": [
                    {
                        "source": metadata.get('file_path', 'Unknown'),
                        "content": doc[:500] + "..." if len(doc) > 500 else doc,
                        "distance": dist
                    }
                    for doc, metadata, dist in zip(
                        context_data['documents'],
                        context_data['metadatas'],
                        context_data.get('distances', [])
                    )
                ]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")