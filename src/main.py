"""
Main Entry Point for Multimodal RAG System.
CLI interface for document processing and RAG queries.
"""

import argparse
import sys
from pathlib import Path

from src.utils import get_logger, get_config
from src import __version__

logger = get_logger(__name__)
config = get_config()


def cmd_ingest(args):
    """Ingest documents into the system."""
    from src.preprocessing import PDFParser, CVPipeline, TextChunker
    from src.embeddings import CustomEmbedder
    from src.retrieval import FAISSVectorStore, Document
    
    logger.info(f"Ingesting documents from: {args.input}")
    
    input_path = Path(args.input)
    
    # Initialize components
    pdf_parser = PDFParser()
    cv_pipeline = CVPipeline()
    chunker = TextChunker(chunk_size=args.chunk_size, chunk_overlap=args.overlap)
    embedder = CustomEmbedder(device=args.device)
    vector_store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)
    
    # Process files
    all_chunks = []
    
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.glob("**/*.pdf")) + list(input_path.glob("**/*.png")) + list(input_path.glob("**/*.jpg"))
    
    for file_path in files:
        logger.info(f"Processing: {file_path.name}")
        
        if file_path.suffix.lower() == ".pdf":
            doc = pdf_parser.parse(file_path)
            for page in doc.pages:
                chunks = chunker.chunk(page.text)
                for chunk in chunks:
                    chunk.metadata["source_file"] = str(file_path)
                    chunk.metadata["page_number"] = page.page_number
                all_chunks.extend(chunks)
        else:
            # Image file - OCR
            result = cv_pipeline.process_image(file_path)
            text = result.get("ocr_text", "")
            if text:
                chunks = chunker.chunk(text)
                for chunk in chunks:
                    chunk.metadata["source_file"] = str(file_path)
                all_chunks.extend(chunks)
    
    logger.info(f"Created {len(all_chunks)} chunks")
    
    # Generate embeddings
    texts = [c.text for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress=True)
    
    # Store in vector database
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
    
    # Save index
    output_path = Path(args.output) if args.output else Path("artifacts/index")
    vector_store.save(output_path)
    
    logger.info(f"Saved index to {output_path}")
    print(f"\nIngestion complete: {len(documents)} documents indexed")


def cmd_query(args):
    """Query the RAG system."""
    from src.embeddings import CustomEmbedder
    from src.retrieval import FAISSVectorStore, DenseRetriever, SparseRetriever, HybridRetriever, RAGPipeline
    
    # Load index
    index_path = Path(args.index)
    if not index_path.exists():
        print(f"Error: Index not found at {index_path}")
        sys.exit(1)
    
    logger.info(f"Loading index from: {index_path}")
    
    # Initialize components
    embedder = CustomEmbedder(device=args.device)
    vector_store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)
    vector_store.load(index_path)
    
    dense_retriever = DenseRetriever(vector_store, embedder)
    sparse_retriever = SparseRetriever(list(vector_store.documents.values()))
    hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)
    
    # Initialize RAG pipeline
    rag = RAGPipeline(
        retriever=hybrid_retriever,
        model_name=args.model
    )
    
    # Interactive mode or single query
    if args.interactive:
        print("\n=== Multimodal RAG System ===")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                query = input("Query: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if not query:
                    continue
                
                response = rag.query(query, top_k=args.top_k)
                print(f"\nAnswer: {response.answer}")
                
                if args.show_sources:
                    print("\nSources:")
                    for i, citation in enumerate(response.citations, 1):
                        print(f"  {i}. {citation.source_file} (score: {citation.relevance_score:.3f})")
                
                print(f"\n[Latency: {response.latency_ms:.0f}ms]\n")
                
            except KeyboardInterrupt:
                break
    else:
        response = rag.query(args.query, top_k=args.top_k)
        print(response.format_answer(include_sources=args.show_sources))


def cmd_evaluate(args):
    """Run evaluation benchmark."""
    from src.evaluation import RAGBenchmark, EvaluationSample
    
    logger.info("Running evaluation benchmark...")
    
    # Load evaluation data
    eval_path = Path(args.eval_data)
    if not eval_path.exists():
        print(f"Error: Evaluation data not found at {eval_path}")
        sys.exit(1)
    
    # Initialize RAG (similar to query command)
    from src.embeddings import CustomEmbedder
    from src.retrieval import FAISSVectorStore, DenseRetriever, SparseRetriever, HybridRetriever, RAGPipeline
    
    embedder = CustomEmbedder(device=args.device)
    vector_store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)
    vector_store.load(Path(args.index))
    
    dense_retriever = DenseRetriever(vector_store, embedder)
    sparse_retriever = SparseRetriever(list(vector_store.documents.values()))
    hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)
    rag = RAGPipeline(retriever=hybrid_retriever, model_name=args.model)
    
    # Run benchmark
    benchmark = RAGBenchmark(rag)
    samples = benchmark.load_evaluation_data(eval_path)
    results = benchmark.run(samples, name=args.name, verbose=True)
    
    # Print and save results
    print("\n" + results.summary())
    
    output_path = Path(args.output) if args.output else Path(f"artifacts/results/{args.name}.json")
    results.save(output_path)


def cmd_serve(args):
    """Start API server (placeholder)."""
    print("API server not yet implemented.")
    print(f"Would start on port {args.port}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="multimodal-rag",
        description="Multimodal RAG System for Document Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Ingest documents:
    python -m src.main ingest --input data/documents/ --output artifacts/index

  Query the system:
    python -m src.main query --index artifacts/index --query "What is machine learning?"

  Interactive mode:
    python -m src.main query --index artifacts/index --interactive

  Run evaluation:
    python -m src.main evaluate --index artifacts/index --eval-data data/eval.json
        """
    )
    
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--input", "-i", required=True, help="Input file or directory")
    ingest_parser.add_argument("--output", "-o", help="Output index directory")
    ingest_parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in characters")
    ingest_parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap")
    ingest_parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    ingest_parser.set_defaults(func=cmd_ingest)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("--index", "-i", required=True, help="Index directory")
    query_parser.add_argument("--query", "-q", help="Query text")
    query_parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    query_parser.add_argument("--show-sources", action="store_true", help="Show sources")
    query_parser.add_argument("--model", default="flan-t5", help="LLM model")
    query_parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    query_parser.set_defaults(func=cmd_query)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation benchmark")
    eval_parser.add_argument("--index", "-i", required=True, help="Index directory")
    eval_parser.add_argument("--eval-data", "-e", required=True, help="Evaluation data JSON")
    eval_parser.add_argument("--name", "-n", default="benchmark", help="Benchmark name")
    eval_parser.add_argument("--output", "-o", help="Results output path")
    eval_parser.add_argument("--model", default="flan-t5", help="LLM model")
    eval_parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port")
    serve_parser.set_defaults(func=cmd_serve)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)


if __name__ == "__main__":
    main()
