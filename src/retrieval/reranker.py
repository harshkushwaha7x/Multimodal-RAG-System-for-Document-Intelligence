"""
Cross-Encoder Reranker for improved retrieval quality.
Uses MS-MARCO trained model to rerank initial results.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from ..utils import get_logger, LoggerMixin

logger = get_logger(__name__)


@dataclass
class RerankResult:
    """Result from reranking."""
    text: str
    original_score: float
    rerank_score: float
    metadata: dict


class CrossEncoderReranker(LoggerMixin):
    """
    Cross-encoder reranker using sentence-transformers.
    Reranks initial retrieval results for better precision.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize reranker.
        
        Args:
            model_name: Cross-encoder model name
            device: Device to run on (cuda/cpu)
        """
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.device = device
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self.model is not None:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            import torch
            
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.logger.info(f"Loading reranker: {self.model_name}")
            self.model = CrossEncoder(self.model_name, device=self.device)
            self.logger.info(f"Reranker loaded on {self.device}")
            
        except ImportError:
            self.logger.error("sentence-transformers not installed")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: int = 5
    ) -> List[RerankResult]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: The search query
            documents: List of documents with 'text' and optional 'score', 'metadata'
            top_k: Number of top results to return
            
        Returns:
            Reranked results with scores
        """
        if not documents:
            return []
        
        self._load_model()
        
        # Prepare query-document pairs
        pairs = [(query, doc.get("text", doc.get("content", ""))) for doc in documents]
        
        # Get rerank scores
        scores = self.model.predict(pairs)
        
        # Combine with original data
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            results.append(RerankResult(
                text=doc.get("text", doc.get("content", "")),
                original_score=doc.get("score", 0.0),
                rerank_score=float(score),
                metadata=doc.get("metadata", {})
            ))
        
        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return results[:top_k]
    
    def rerank_search_results(
        self,
        query: str,
        search_results: List,
        top_k: int = 5
    ) -> List:
        """
        Rerank SearchResult objects from vector store.
        
        Args:
            query: The search query
            search_results: List of SearchResult objects
            top_k: Number of top results to return
            
        Returns:
            Reranked SearchResult objects
        """
        if not search_results:
            return []
        
        self._load_model()
        
        # Prepare pairs
        pairs = [(query, r.document.text) for r in search_results]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Attach scores and sort
        scored_results = list(zip(search_results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Update ranks
        reranked = []
        for i, (result, score) in enumerate(scored_results[:top_k]):
            result.rank = i
            result.score = float(score)
            reranked.append(result)
        
        return reranked


# Singleton instance
_reranker: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    """Get or create the global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker
