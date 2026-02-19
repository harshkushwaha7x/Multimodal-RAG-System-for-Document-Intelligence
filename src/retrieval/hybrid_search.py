"""
Hybrid Search Module.
Combines dense and sparse retrieval with Reciprocal Rank Fusion.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .vector_db import Document, SearchResult, VectorStore
from ..embeddings import CustomEmbedder
from ..utils import get_logger, get_config, LoggerMixin

logger = get_logger(__name__)
config = get_config()


class DenseRetriever(LoggerMixin):
    """
    Dense retrieval using vector similarity.
    
    Uses embedding-based similarity search for
    semantic matching.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Optional[CustomEmbedder] = None
    ):
        """
        Initialize dense retriever.
        
        Args:
            vector_store: Vector store for search
            embedder: Embedding model (optional, uses default if not provided)
        """
        self.vector_store = vector_store
        self.embedder = embedder or CustomEmbedder()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filter_metadata: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Retrieve documents using dense search.
        
        Args:
            query: Query string
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of SearchResult ordered by similarity
        """
        self.logger.debug(f"Dense retrieval for: {query[:50]}...")
        
        # Encode query
        query_embedding = self.embedder.encode(query)
        
        # Search
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )
        
        # Normalize scores to [0, 1]
        if results:
            max_score = max(r.score for r in results)
            min_score = min(r.score for r in results)
            score_range = max_score - min_score if max_score != min_score else 1
            
            for i, result in enumerate(results):
                result.score = (result.score - min_score) / score_range
                result.rank = i
        
        self.logger.debug(f"Retrieved {len(results)} documents")
        return results


class SparseRetriever(LoggerMixin):
    """
    Sparse retrieval using BM25.
    
    Uses keyword-based matching for
    lexical recall.
    """
    
    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        k1: float = None,
        b: float = None
    ):
        """
        Initialize sparse retriever.
        
        Args:
            documents: Documents to index
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self.k1 = k1 or config.retrieval.bm25_k1
        self.b = b or config.retrieval.bm25_b
        self.bm25 = None
        self.documents: List[Document] = []
        self.tokenized_corpus = []
        
        if documents:
            self.index_documents(documents)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def index_documents(self, documents: List[Document]):
        """
        Index documents for BM25 search.
        
        Args:
            documents: Documents to index
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            self.logger.error("rank-bm25 not installed")
            raise ImportError("Install rank-bm25: pip install rank-bm25")
        
        self.documents = documents
        self.tokenized_corpus = [self._tokenize(doc.text) for doc in documents]
        
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b
        )
        
        self.logger.info(f"Indexed {len(documents)} documents for BM25")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 20
    ) -> List[SearchResult]:
        """
        Retrieve documents using BM25.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of SearchResult ordered by BM25 score
        """
        if self.bm25 is None:
            self.logger.warning("No documents indexed")
            return []
        
        self.logger.debug(f"Sparse retrieval for: {query[:50]}...")
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Normalize scores
        max_score = max(scores) if max(scores) > 0 else 1
        
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                results.append(SearchResult(
                    document=self.documents[idx],
                    score=scores[idx] / max_score,  # Normalize to [0, 1]
                    rank=rank
                ))
        
        self.logger.debug(f"Retrieved {len(results)} documents")
        return results


class HybridRetriever(LoggerMixin):
    """
    Hybrid retrieval combining dense and sparse methods.
    
    Uses Reciprocal Rank Fusion (RRF) to combine
    results from multiple retrieval methods.
    """
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        dense_weight: float = None,
        rrf_k: int = None
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            dense_retriever: Dense retrieval component
            sparse_retriever: Sparse retrieval component
            dense_weight: Weight for dense results (0-1)
            rrf_k: RRF constant (typically 60)
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight or config.retrieval.dense_weight
        self.rrf_k = rrf_k or config.retrieval.rrf_k
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        initial_k: int = None,
        method: str = "rrf"
    ) -> List[SearchResult]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: Query string
            top_k: Final number of results
            initial_k: Initial results from each method
            method: Fusion method ("rrf" or "weighted")
            
        Returns:
            List of SearchResult with fused scores
        """
        top_k = top_k or config.retrieval.final_k
        initial_k = initial_k or config.retrieval.top_k
        
        self.logger.info(f"Hybrid retrieval: {query[:50]}...")
        
        # Get results from both methods
        dense_results = self.dense_retriever.retrieve(query, top_k=initial_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=initial_k)
        
        self.logger.debug(
            f"Dense: {len(dense_results)} results, "
            f"Sparse: {len(sparse_results)} results"
        )
        
        # Fusion
        if method == "rrf":
            fused_results = self._reciprocal_rank_fusion(
                dense_results, sparse_results
            )
        elif method == "weighted":
            fused_results = self._weighted_fusion(
                dense_results, sparse_results
            )
        else:
            raise ValueError(f"Unknown fusion method: {method}")
        
        # Return top-k
        return fused_results[:top_k]
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF Score = Σ 1 / (k + rank)
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            
        Returns:
            Fused results sorted by RRF score
        """
        doc_scores: Dict[str, float] = {}
        doc_objects: Dict[str, Document] = {}
        
        # Process dense results
        for result in dense_results:
            doc_id = result.document.id
            rrf_score = 1 / (self.rrf_k + result.rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_objects[doc_id] = result.document
        
        # Process sparse results
        for result in sparse_results:
            doc_id = result.document.id
            rrf_score = 1 / (self.rrf_k + result.rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_objects[doc_id] = result.document
        
        # Sort by RRF score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create results
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs):
            results.append(SearchResult(
                document=doc_objects[doc_id],
                score=score,
                rank=rank
            ))
        
        self.logger.debug(f"RRF fusion produced {len(results)} unique results")
        return results
    
    def _weighted_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Combine results using weighted averaging.
        
        Score = α × dense_score + (1-α) × sparse_score
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            
        Returns:
            Fused results sorted by weighted score
        """
        doc_scores: Dict[str, float] = {}
        doc_objects: Dict[str, Document] = {}
        
        # Process dense results
        for result in dense_results:
            doc_id = result.document.id
            weighted_score = self.dense_weight * result.score
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + weighted_score
            doc_objects[doc_id] = result.document
        
        # Process sparse results
        sparse_weight = 1 - self.dense_weight
        for result in sparse_results:
            doc_id = result.document.id
            weighted_score = sparse_weight * result.score
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + weighted_score
            doc_objects[doc_id] = result.document
        
        # Sort by weighted score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create results
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs):
            results.append(SearchResult(
                document=doc_objects[doc_id],
                score=score,
                rank=rank
            ))
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid Search Test")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    args = parser.parse_args()
    
    if args.test:
        print("Hybrid Search Test\n" + "=" * 50)
        
        # Create sample documents
        sample_docs = [
            Document(id="1", text="Machine learning is a subset of artificial intelligence."),
            Document(id="2", text="Deep learning uses neural networks with many layers."),
            Document(id="3", text="Natural language processing enables text understanding."),
            Document(id="4", text="Computer vision allows machines to interpret images."),
            Document(id="5", text="Reinforcement learning trains agents using rewards."),
        ]
        
        # Initialize sparse retriever
        sparse = SparseRetriever(documents=sample_docs)
        
        # Test BM25 search
        query = "machine learning neural networks"
        results = sparse.retrieve(query, top_k=3)
        
        print(f"Query: {query}")
        print(f"\nBM25 Results:")
        for r in results:
            print(f"  [{r.rank}] {r.document.id}: {r.score:.4f}")
            print(f"      {r.document.text[:50]}...")
