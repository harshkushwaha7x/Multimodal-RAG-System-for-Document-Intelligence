"""
Gradio Web UI for Multimodal RAG System.
Provides a visual interface for document upload and Q&A.
"""

import gradio as gr
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Generator

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing import PDFParser, TextChunker
from src.embeddings import CustomEmbedder
from src.retrieval import FAISSVectorStore, Document, HybridRetriever, RAGPipeline, SparseRetriever, DenseRetriever
from src.utils import get_logger

logger = get_logger(__name__)

# Global state
vector_store = None
rag_pipeline = None
embedder = None
sparse_retriever = None
dense_retriever = None
current_model = "flan-t5"  # Default to open model


def get_current_model() -> str:
    """Get the current model name."""
    global current_model
    return current_model


def set_model(model_name: str) -> str:
    """Set the model for the RAG pipeline."""
    global current_model, rag_pipeline
    
    current_model = model_name
    
    # If pipeline exists, update it with new model
    if rag_pipeline is not None:
        try:
            # Clear existing LLM to force reload with new model
            rag_pipeline.llm = None
            rag_pipeline.model_name = rag_pipeline._resolve_model_name(model_name)
            return f"[OK] Model set to: {model_name}"
        except Exception as e:
            logger.error(f"Error setting model: {e}")
            return f"[ERROR] Failed to set model: {str(e)}"
    else:
        return f"[OK] Model will be: {model_name} (load documents first)"


def ingest_documents(files: List[tempfile.SpooledTemporaryFile]) -> str:
    """Process uploaded documents."""
    global vector_store, embedder, rag_pipeline, sparse_retriever, dense_retriever, current_model
    
    if not files:
        return "[ERROR] No files uploaded!"
    
    try:
        # Initialize embedder if needed
        if embedder is None:
            embedder = CustomEmbedder()
        
        # Create new vector store
        vector_store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)
        
        pdf_parser = PDFParser()
        chunker = TextChunker(chunk_size=512, chunk_overlap=50)
        all_chunks = []
        
        for file in files:
            file_path = Path(file.name)
            
            if file_path.suffix.lower() == ".pdf":
                doc = pdf_parser.parse(file_path)
                for page in doc.pages:
                    chunks = chunker.chunk(page.text)
                    for chunk in chunks:
                        chunk.metadata["source_file"] = file_path.name
                        chunk.metadata["page_number"] = page.page_number
                        all_chunks.append(chunk)
        
        if not all_chunks:
            return "[ERROR] No text extracted from documents!"
        
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
        
        # Add to vector store
        vector_store.add_documents(documents)
        
        # Create retrievers
        dense_retriever = DenseRetriever(vector_store=vector_store, embedder=embedder)
        sparse_retriever = SparseRetriever()
        sparse_retriever.index_documents(documents)
        
        # Create hybrid retriever
        retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever
        )
        
        # Create RAG pipeline with current model (NOT hardcoded!)
        rag_pipeline = RAGPipeline(
            retriever=retriever,
            model_name=current_model
        )
        
        return f"[OK] Ingested {len(documents)} chunks from {len(files)} file(s) using model: {current_model}"
        
    except Exception as e:
        import traceback
        logger.error(f"Ingestion error: {e}")
        logger.error(traceback.format_exc())
        return f"[ERROR] Error: {str(e)}"


def query_rag(
    message: str,
    history: List[Tuple[str, str]],
    top_k: int = 5
) -> str:
    """Query the RAG system."""
    global rag_pipeline
    
    if rag_pipeline is None:
        return "[ERROR] Please upload documents first (Documents tab)!"
    
    if not message.strip():
        return "[ERROR] Please enter a question!"
    
    try:
        logger.info(f"Processing query: {message}")
        
        # Query RAG pipeline
        response = rag_pipeline.query(message, top_k=top_k)
        
        # Format answer with sources
        answer = response.answer
        
        # Add source citations (RAGResponse uses 'citations')
        if response.citations:
            answer += "\n\n---\n**Sources:**\n"
            for i, citation in enumerate(response.citations[:3], 1):
                text_preview = citation.text_snippet[:150].replace("\n", " ") if citation.text_snippet else ""
                source = citation.source_file
                if citation.page:
                    source += f" (p.{citation.page})"
                answer += f"\n[{i}] **{source}**: {text_preview}..."
        
        return answer
        
    except Exception as e:
        import traceback
        logger.error(f"Query error: {e}")
        logger.error(traceback.format_exc())
        return f"[ERROR] Error: {str(e)}"


def query_rag_streaming(
    message: str,
    history: List[Tuple[str, str]],
    top_k: int = 5
) -> Generator[str, None, None]:
    """Query the RAG system with streaming response."""
    global rag_pipeline
    
    if rag_pipeline is None:
        yield "[ERROR] Please upload documents first (Documents tab)!"
        return
    
    if not message.strip():
        yield "[ERROR] Please enter a question!"
        return
    
    try:
        logger.info(f"Processing streaming query: {message}")
        
        # Show thinking indicator
        yield " Searching documents..."
        
        # Get response (we simulate streaming by yielding partial content)
        response = rag_pipeline.query(message, top_k=top_k)
        
        # Stream the answer word by word for effect
        answer = response.answer
        words = answer.split()
        partial = ""
        
        for i, word in enumerate(words):
            partial += word + " "
            if i % 5 == 0:  # Update every 5 words
                yield partial
        
        # Add sources at the end
        if response.citations:
            sources = "\n\n---\n**Sources:**\n"
            for i, citation in enumerate(response.citations[:3], 1):
                text_preview = citation.text_snippet[:150].replace("\n", " ") if citation.text_snippet else ""
                source = citation.source_file
                if citation.page:
                    source += f" (p.{citation.page})"
                sources += f"\n[{i}] **{source}**: {text_preview}..."
            
            yield partial + sources
        else:
            yield partial
            
    except Exception as e:
        logger.error(f"Streaming query error: {e}")
        yield f"[ERROR] Error: {str(e)}"


def export_conversation(history: List[dict]) -> str:
    """Export conversation history to markdown."""
    if not history:
        return "No conversation to export."
    
    markdown = "# RAG Conversation Export\n\n"
    markdown += f"*Exported on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
    markdown += "---\n\n"
    
    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if role == "user":
            markdown += f"## Question\n\n{content}\n\n"
        else:
            markdown += f"## Answer\n\n{content}\n\n"
        
        markdown += "---\n\n"
    
    return markdown


def save_export(history: List[dict]) -> str:
    """Save conversation export to file."""
    import tempfile
    from datetime import datetime
    
    markdown = export_conversation(history)
    
    filename = f"rag_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    filepath = Path(tempfile.gettempdir()) / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    return str(filepath)


def load_existing_index(index_path: str) -> str:
    """Load an existing FAISS index."""
    global vector_store, rag_pipeline, embedder, sparse_retriever, dense_retriever, current_model
    
    try:
        path = Path(index_path)
        if not path.exists():
            return f"[ERROR] Index path not found: {index_path}"
        
        embedder = CustomEmbedder()
        vector_store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)
        vector_store.load(index_path)
        
        # Wrap vector_store with DenseRetriever
        dense_retriever = DenseRetriever(vector_store=vector_store, embedder=embedder)
        
        # Get documents for sparse indexing
        docs = vector_store.get_all_documents()
        sparse_retriever = SparseRetriever()
        sparse_retriever.index_documents(docs)
        
        retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever
        )
        
        # Use current_model (NOT hardcoded llama3!)
        rag_pipeline = RAGPipeline(
            retriever=retriever,
            model_name=current_model
        )
        
        # PRELOAD the LLM to avoid threading issues during query
        logger.info("Preloading LLM (this takes 30-60 seconds)...")
        rag_pipeline._load_llm()
        logger.info("LLM preloaded successfully!")
        
        return f"[OK] Loaded index from {index_path} ({vector_store.count} documents) with model: {current_model}"
        
    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        return f"[ERROR] Error loading index: {str(e)}"


# Create Gradio interface
def create_ui():
    """Create the Gradio interface."""
    
    with gr.Blocks(
        title="Multimodal RAG System"
    ) as demo:
        gr.Markdown("""
        #  Multimodal RAG System
        ### Intelligent Document Q&A with Citations
        """)
        
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500
                    )
                    msg = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a question about your documents...",
                        lines=2
                    )
                    with gr.Row():
                        submit_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Settings")
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Number of Sources"
                    )
                    model_select = gr.Dropdown(
                        choices=["flan-t5", "flan-t5-base", "qwen2", "gemma2", "phi3", "opt"],
                        value="flan-t5",
                        label="LLM Model"
                    )
                    model_status = gr.Textbox(label="Model Status", interactive=False, value="Model: flan-t5")
            
            # Chat handlers (Gradio 4.x uses tuple format)
            def respond(message, chat_history, top_k):
                chat_history = chat_history or []
                response = query_rag(message, chat_history, top_k)
                # Gradio 4.x uses tuple format: [(user_msg, bot_msg), ...]
                chat_history.append((message, response))
                return "", chat_history
            
            submit_btn.click(respond, [msg, chatbot, top_k], [msg, chatbot])
            msg.submit(respond, [msg, chatbot, top_k], [msg, chatbot])
            clear_btn.click(lambda: [], None, chatbot)
            model_select.change(set_model, [model_select], [model_status])
        
        with gr.Tab("Documents"):
            gr.Markdown("### Upload Documents")
            gr.Markdown("*Select your LLM model in the Chat tab before uploading*")
            
            file_upload = gr.File(
                label="Upload PDFs",
                file_types=[".pdf"],
                file_count="multiple"
            )
            upload_btn = gr.Button("Process Documents", variant="primary")
            upload_status = gr.Textbox(label="Upload Status", interactive=False)
            
            upload_btn.click(ingest_documents, [file_upload], [upload_status])
            
            gr.Markdown("---")
            gr.Markdown("### Or Load Existing Index")
            
            index_path = gr.Textbox(
                label="Index Path",
                value="artifacts/index",
                placeholder="Path to saved FAISS index"
            )
            load_btn = gr.Button("Load Index")
            load_status = gr.Textbox(label="Load Status", interactive=False)
            
            load_btn.click(load_existing_index, [index_path], [load_status])
        
        with gr.Tab("About"):
            gr.Markdown("""
            ## About This System
            
            This is a **Multimodal RAG (Retrieval-Augmented Generation)** system for document intelligence.
            
            ### Features
            - **PDF Document Processing** - Extract text from PDFs
            -  **Hybrid Search** - Combines dense vectors + BM25
            - **LLM-Powered Answers** - Generates responses with citations
            - **GPU Accelerated** - Fast inference with CUDA
            
            ### How to Use
            1. Select your **LLM Model** in the Chat tab
            2. Go to **Documents** tab and upload PDFs
            3. Ask questions about your documents!
            
            ### Models (Open/No Auth Required)
            - **Flan-T5** (default) - Lightweight, fast
            - **Flan-T5-Base** - Even smaller
            - **Qwen2** (1.5B) - Good quality
            - **Gemma2** (2B) - Google's model
            - **Phi-3** (3.8B) - Microsoft, high quality
            - **OPT** (1.3B) - Facebook/Meta
            """)
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    # Enable queue for long-running tasks (LLM loading takes ~30s)
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
