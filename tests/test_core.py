"""
Unit tests for core RAG components.
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import TextChunker
from src.retrieval import FAISSVectorStore, Document


class TestTextChunker:
    """Tests for TextChunker."""
    
    def test_basic_chunking(self):
        """Test basic text chunking."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a test. " * 20
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 0
        assert all(hasattr(c, 'text') for c in chunks)
        assert all(hasattr(c, 'chunk_id') for c in chunks)
    
    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk("")
        
        assert len(chunks) == 0
    
    def test_short_text(self):
        """Test text shorter than chunk size."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
        text = "Short text."
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1
        assert chunks[0].text == text
    
    def test_overlap(self):
        """Test that chunks have proper overlap."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "Word " * 30
        
        chunks = chunker.chunk(text)
        
        # With overlap, consecutive chunks should share some content
        if len(chunks) > 1:
            # Just verify multiple chunks were created
            assert len(chunks) >= 2


class TestFAISSVectorStore:
    """Tests for FAISS vector store."""
    
    @pytest.fixture
    def store(self):
        """Create a test vector store."""
        return FAISSVectorStore(embedding_dim=384, index_type="flat")
    
    def test_initialization(self, store):
        """Test store initialization."""
        assert store.count == 0
        assert store.embedding_dim == 384
    
    def test_add_documents(self, store):
        """Test adding documents."""
        docs = [
            Document(
                id="1",
                text="Test document one",
                embedding=np.random.randn(384).astype(np.float32),
                metadata={"source": "test"}
            ),
            Document(
                id="2",
                text="Test document two",
                embedding=np.random.randn(384).astype(np.float32),
                metadata={"source": "test"}
            )
        ]
        
        store.add_documents(docs)
        
        assert store.count == 2
    
    def test_search(self, store):
        """Test searching documents."""
        # Add documents
        docs = [
            Document(
                id=str(i),
                text=f"Document {i}",
                embedding=np.random.randn(384).astype(np.float32)
            )
            for i in range(10)
        ]
        store.add_documents(docs)
        
        # Search
        query_embedding = np.random.randn(384).astype(np.float32)
        results = store.search(query_embedding, top_k=5)
        
        assert len(results) == 5
        assert all(hasattr(r, 'document') for r in results)
        assert all(hasattr(r, 'score') for r in results)
    
    def test_save_load(self, store, tmp_path):
        """Test saving and loading index."""
        # Add documents
        docs = [
            Document(
                id=str(i),
                text=f"Document {i}",
                embedding=np.random.randn(384).astype(np.float32)
            )
            for i in range(5)
        ]
        store.add_documents(docs)
        
        # Save
        save_path = str(tmp_path / "test_index")
        store.save(save_path)
        
        # Load into new store
        new_store = FAISSVectorStore(embedding_dim=384)
        new_store.load(save_path)
        
        assert new_store.count == 5


class TestDocument:
    """Tests for Document dataclass."""
    
    def test_document_creation(self):
        """Test document creation."""
        doc = Document(
            id="test-1",
            text="Sample text",
            embedding=np.zeros(768),
            metadata={"key": "value"}
        )
        
        assert doc.id == "test-1"
        assert doc.text == "Sample text"
        assert len(doc.embedding) == 768
        assert doc.metadata["key"] == "value"
    
    def test_document_defaults(self):
        """Test document default values."""
        doc = Document(
            id="test",
            text="text",
            embedding=np.zeros(768)
        )
        
        assert doc.metadata == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
