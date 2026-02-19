"""
FastAPI Server for Multimodal RAG System.
Provides REST API endpoints for document processing and Q&A.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import tempfile
import shutil
from pathlib import Path
import asyncio

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing import PDFParser, TextChunker
from src.embeddings import CustomEmbedder
from src.retrieval import FAISSVectorStore, Document, HybridRetriever, RAGPipeline, DenseRetriever, SparseRetriever
from src.utils import get_logger

logger = get_logger(__name__)

# FastAPI app
app = FastAPI(
    title="Multimodal RAG API",
    description="REST API for document intelligence and Q&A",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
vector_store: Optional[FAISSVectorStore] = None
rag_pipeline: Optional[RAGPipeline] = None
embedder: Optional[CustomEmbedder] = None


# Request/Response models
class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask")
    top_k: int = Field(5, description="Number of sources to retrieve")
    model: str = Field("qwen2", description="LLM model to use")


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    latency_ms: float


class IngestRequest(BaseModel):
    index_path: str = Field("artifacts/index", description="Path to save/load index")


class StatusResponse(BaseModel):
    status: str
    documents_count: int
    model_name: Optional[str]


class HealthResponse(BaseModel):
    status: str
    version: str


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status."""
    global vector_store, rag_pipeline
    
    return StatusResponse(
        status="ready" if rag_pipeline else "not_initialized",
        documents_count=vector_store.count if vector_store else 0,
        model_name=rag_pipeline.model_name if rag_pipeline else None
    )


@app.post("/initialize")
async def initialize(model: str = "qwen2"):
    """Initialize the RAG system."""
    global vector_store, rag_pipeline, embedder
    
    try:
        embedder = CustomEmbedder()
        vector_store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)
        
        # Use DenseRetriever wrapper and SparseRetriever
        dense_retriever = DenseRetriever(vector_store=vector_store, embedder=embedder)
        sparse_retriever = SparseRetriever()
        
        retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever
        )
        
        rag_pipeline = RAGPipeline(
            retriever=retriever,
            model_name=model
        )
        
        return {"status": "success", "message": f"Initialized with model: {model}"}
    
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/upload")
async def ingest_upload(files: List[UploadFile] = File(...)):
    """Upload and process documents."""
    global vector_store, embedder, rag_pipeline
    
    # Auto-initialize if needed
    if vector_store is None or embedder is None:
        embedder = CustomEmbedder()
        vector_store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)
    
    try:
        pdf_parser = PDFParser()
        chunker = TextChunker(chunk_size=512, chunk_overlap=50)
        all_chunks = []
        processed_files = []
        
        for file in files:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = Path(tmp.name)
            
            try:
                doc = pdf_parser.parse(tmp_path)
                for page in doc.pages:
                    chunks = chunker.chunk(page.text)
                    for chunk in chunks:
                        chunk.metadata["source_file"] = file.filename
                        chunk.metadata["page_number"] = page.page_number
                        all_chunks.append(chunk)
                processed_files.append(file.filename)
            finally:
                tmp_path.unlink()
        
        if not all_chunks:
            raise HTTPException(status_code=400, detail="No text extracted")
        
        # Generate embeddings
        texts = [c.text for c in all_chunks]
        embeddings = embedder.encode(texts, show_progress=True)
        
        # Create documents
        documents = [
            Document(
                id=c.chunk_id,
                text=c.text,
                embedding=embeddings[i],
                metadata=c.metadata
            )
            for i, c in enumerate(all_chunks)
        ]
        
        vector_store.add_documents(documents)
        
        # Auto-initialize RAG pipeline
        dense_retriever = DenseRetriever(vector_store=vector_store, embedder=embedder)
        sparse_retriever = SparseRetriever()
        sparse_retriever.index_documents(documents)
        
        retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever
        )
        
        rag_pipeline = RAGPipeline(
            retriever=retriever,
            model_name="qwen2"
        )
        
        return {
            "status": "success",
            "chunks_created": len(documents),
            "files_processed": processed_files
        }
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/load")
async def ingest_load(request: IngestRequest):
    """Load existing index from disk."""
    global vector_store, rag_pipeline, embedder
    
    try:
        path = Path(request.index_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Index not found: {path}")
        
        embedder = CustomEmbedder()
        vector_store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)
        vector_store.load(request.index_path)
        
        # Use correct retriever setup
        dense_retriever = DenseRetriever(vector_store=vector_store, embedder=embedder)
        
        docs = vector_store.get_all_documents()
        sparse_retriever = SparseRetriever()
        sparse_retriever.index_documents(docs)
        
        retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever
        )
        
        rag_pipeline = RAGPipeline(
            retriever=retriever,
            model_name="qwen2"
        )
        
        return {
            "status": "success",
            "documents_count": vector_store.count,
            "index_path": str(path)
        }
        
    except Exception as e:
        logger.error(f"Load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system."""
    global rag_pipeline
    
    if rag_pipeline is None:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    try:
        import time
        start = time.time()
        
        response = rag_pipeline.query(request.question, top_k=request.top_k)
        
        latency = (time.time() - start) * 1000
        
        # Use citations (correct attribute) instead of sources
        sources = []
        for citation in response.citations[:5]:
            sources.append({
                "text": citation.text_snippet[:200] if citation.text_snippet else "",
                "score": citation.relevance_score,
                "source_file": citation.source_file,
                "page": citation.page
            })
        
        return QueryResponse(
            answer=response.answer,
            sources=sources,
            latency_ms=latency
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the index."""
    global vector_store, embedder
    
    if vector_store is None:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    # Reinitialize empty store
    vector_store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)
    
    return {"status": "success", "message": "All documents cleared"}


@app.post("/save")
async def save_index(path: str = "artifacts/index"):
    """Save the current index to disk."""
    global vector_store
    
    if vector_store is None:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    try:
        vector_store.save(path)
        return {"status": "success", "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
