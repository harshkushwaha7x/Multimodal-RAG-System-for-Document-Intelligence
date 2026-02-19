"""
Tests for RAG pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRAGPipeline:
    """Tests for RAG pipeline."""
    
    def test_import(self):
        """Test that RAG pipeline can be imported."""
        from src.retrieval import RAGPipeline
        assert RAGPipeline is not None
    
    def test_prompt_formatting(self):
        """Test prompt formatting."""
        from src.retrieval.rag_pipeline import RAGPipeline
        
        # Just test the class exists and has expected attributes
        assert hasattr(RAGPipeline, 'SUPPORTED_MODELS')
        assert 'qwen2' in RAGPipeline.SUPPORTED_MODELS


class TestHybridRetriever:
    """Tests for Hybrid retriever."""
    
    def test_import(self):
        """Test that hybrid retriever can be imported."""
        from src.retrieval import HybridRetriever, DenseRetriever, SparseRetriever
        assert HybridRetriever is not None
        assert DenseRetriever is not None
        assert SparseRetriever is not None


class TestReranker:
    """Tests for reranker."""
    
    def test_import(self):
        """Test that reranker can be imported."""
        from src.retrieval.reranker import CrossEncoderReranker
        assert CrossEncoderReranker is not None
    
    def test_reranker_creation(self):
        """Test reranker instantiation."""
        from src.retrieval.reranker import CrossEncoderReranker
        
        reranker = CrossEncoderReranker()
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
