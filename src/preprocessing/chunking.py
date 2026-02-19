"""
Text Chunking Module for Document Processing.
Implements sentence-aware and semantic chunking strategies.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from hashlib import md5

from ..utils import get_logger, get_config, LoggerMixin

logger = get_logger(__name__)
config = get_config()


@dataclass
class Chunk:
    """A text chunk with metadata."""
    
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    metadata: Dict = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.text)
    
    @property
    def token_estimate(self) -> int:
        """Estimate token count (rough: 1 token ≈ 4 chars)."""
        return len(self.text) // 4
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_estimate": self.token_estimate,
            "metadata": self.metadata
        }


class TextChunker(LoggerMixin):
    """
    Sentence-aware text chunker with configurable overlap.
    
    Creates chunks that preserve sentence boundaries while
    maintaining target chunk sizes for optimal retrieval.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_size: int = None,
        max_chunk_size: int = None,
        length_function: Optional[Callable[[str], int]] = None
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Target chunk size (tokens/chars)
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
            length_function: Custom function to measure text length
        """
        self.chunk_size = chunk_size or config.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunking.chunk_overlap
        self.min_chunk_size = min_chunk_size or config.chunking.min_chunk_size
        self.max_chunk_size = max_chunk_size or config.chunking.max_chunk_size
        self.length_function = length_function or self._default_length
        
        # Initialize sentence tokenizer
        self._init_tokenizer()
    
    def _init_tokenizer(self):
        """Initialize NLTK or spaCy for sentence tokenization."""
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                self.logger.info("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
            
            from nltk.tokenize import sent_tokenize
            self.sent_tokenize = sent_tokenize
            self.tokenizer_type = "nltk"
            self.logger.debug("Using NLTK for sentence tokenization")
            
        except ImportError:
            self.logger.warning("NLTK not available, using regex fallback")
            self.sent_tokenize = self._regex_sent_tokenize
            self.tokenizer_type = "regex"
    
    def _regex_sent_tokenize(self, text: str) -> List[str]:
        """Fallback regex sentence tokenizer."""
        # Simple pattern for sentence boundaries
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _default_length(self, text: str) -> int:
        """Default length function using character count."""
        return len(text)
    
    def _generate_chunk_id(self, text: str, index: int) -> str:
        """Generate unique chunk ID."""
        hash_input = f"{text[:50]}_{index}"
        return md5(hash_input.encode()).hexdigest()[:12]
    
    def chunk(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        self.logger.debug(f"Chunking text of length {len(text)}")
        metadata = metadata or {}
        
        # Tokenize into sentences
        sentences = self.sent_tokenize(text)
        self.logger.debug(f"Split into {len(sentences)} sentences")
        
        chunks = []
        current_chunk_sentences = []
        current_length = 0
        char_position = 0
        
        for sentence in sentences:
            sentence_length = self.length_function(sentence)
            
            # If single sentence exceeds max size, split it
            if sentence_length > self.max_chunk_size:
                # Save current chunk first
                if current_chunk_sentences:
                    chunk = self._create_chunk(
                        current_chunk_sentences,
                        len(chunks),
                        char_position - current_length,
                        metadata
                    )
                    chunks.append(chunk)
                    current_chunk_sentences = []
                    current_length = 0
                
                # Split long sentence
                sub_chunks = self._split_long_text(sentence, char_position, metadata, len(chunks))
                chunks.extend(sub_chunks)
                char_position += sentence_length + 1
                continue
            
            # Check if adding sentence exceeds chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk_sentences:
                # Create chunk from current sentences
                chunk = self._create_chunk(
                    current_chunk_sentences,
                    len(chunks),
                    char_position - current_length,
                    metadata
                )
                chunks.append(chunk)
                
                # Calculate overlap - keep some sentences for next chunk
                overlap_sentences = []
                overlap_length = 0
                for sent in reversed(current_chunk_sentences):
                    sent_len = self.length_function(sent)
                    if overlap_length + sent_len <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += sent_len
                    else:
                        break
                
                current_chunk_sentences = overlap_sentences
                current_length = overlap_length
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_length += sentence_length
            char_position += sentence_length + 1  # +1 for space
        
        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk = self._create_chunk(
                current_chunk_sentences,
                len(chunks),
                char_position - current_length,
                metadata
            )
            # Only apply min_chunk_size filter if there are other chunks to merge with
            if self.length_function(chunk.text) >= self.min_chunk_size:
                chunks.append(chunk)
            elif chunks:  # Merge with previous chunk if too small
                chunks[-1].text += " " + chunk.text
                chunks[-1].end_char = chunk.end_char
            else:  # Keep the chunk even if small (it's the only content)
                chunks.append(chunk)
        
        self.logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def _create_chunk(
        self,
        sentences: List[str],
        index: int,
        start_char: int,
        metadata: Dict
    ) -> Chunk:
        """Create a Chunk object from sentences."""
        text = " ".join(sentences)
        chunk_id = self._generate_chunk_id(text, index)
        
        return Chunk(
            chunk_id=chunk_id,
            text=text,
            start_char=start_char,
            end_char=start_char + len(text),
            metadata={
                **metadata,
                "chunk_index": index,
                "sentence_count": len(sentences)
            }
        )
    
    def _split_long_text(
        self,
        text: str,
        start_char: int,
        metadata: Dict,
        start_index: int
    ) -> List[Chunk]:
        """Split a long piece of text that exceeds max chunk size."""
        chunks = []
        words = text.split()
        current_words = []
        current_length = 0
        local_char_pos = start_char
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > self.chunk_size and current_words:
                chunk_text = " ".join(current_words)
                chunk = Chunk(
                    chunk_id=self._generate_chunk_id(chunk_text, start_index + len(chunks)),
                    text=chunk_text,
                    start_char=local_char_pos,
                    end_char=local_char_pos + len(chunk_text),
                    metadata={
                        **metadata,
                        "chunk_index": start_index + len(chunks),
                        "is_split": True
                    }
                )
                chunks.append(chunk)
                local_char_pos += len(chunk_text) + 1
                current_words = []
                current_length = 0
            
            current_words.append(word)
            current_length += word_length
        
        if current_words:
            chunk_text = " ".join(current_words)
            chunk = Chunk(
                chunk_id=self._generate_chunk_id(chunk_text, start_index + len(chunks)),
                text=chunk_text,
                start_char=local_char_pos,
                end_char=local_char_pos + len(chunk_text),
                metadata={
                    **metadata,
                    "chunk_index": start_index + len(chunks),
                    "is_split": True
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_documents(
        self,
        documents: List[Dict],
        text_key: str = "text",
        metadata_keys: Optional[List[str]] = None
    ) -> List[Chunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dicts
            text_key: Key for text content in document dict
            metadata_keys: Keys to include in chunk metadata
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        metadata_keys = metadata_keys or []
        
        for doc in documents:
            text = doc.get(text_key, "")
            
            # Extract metadata
            metadata = {key: doc.get(key) for key in metadata_keys if key in doc}
            
            # Chunk document
            chunks = self.chunk(text, metadata)
            all_chunks.extend(chunks)
        
        return all_chunks


class SemanticChunker(LoggerMixin):
    """
    Semantic-aware chunker that uses embeddings for boundary detection.
    
    Creates chunks based on semantic similarity rather than
    just sentence boundaries, for better retrieval quality.
    """
    
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 800
    ):
        """
        Initialize semantic chunker.
        
        Args:
            embedding_model: Sentence transformer model name
            similarity_threshold: Threshold for splitting chunks
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
        """
        self.embedding_model_name = embedding_model or config.embedding.model_name
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.model = None
    
    def _load_model(self):
        """Lazy load embedding model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.embedding_model_name)
                self.logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except ImportError:
                self.logger.error("sentence-transformers not installed")
                raise
    
    def chunk(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Split text into semantically coherent chunks.
        
        Args:
            text: Input text
            metadata: Optional metadata
            
        Returns:
            List of Chunk objects
        """
        self._load_model()
        
        if not text or not text.strip():
            return []
        
        self.logger.debug(f"Semantic chunking text of length {len(text)}")
        metadata = metadata or {}
        
        # First, use regular sentence splitting
        base_chunker = TextChunker(
            chunk_size=150,  # Smaller initial chunks
            chunk_overlap=0,
            min_chunk_size=50,
            max_chunk_size=300
        )
        initial_chunks = base_chunker.chunk(text)
        
        if len(initial_chunks) <= 1:
            return initial_chunks
        
        # Get embeddings for each chunk
        import numpy as np
        chunk_texts = [c.text for c in initial_chunks]
        embeddings = self.model.encode(chunk_texts, show_progress_bar=False)
        
        # Calculate cosine similarities between adjacent chunks
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            similarities.append(sim)
        
        # Merge chunks based on similarity
        final_chunks = []
        current_texts = [initial_chunks[0].text]
        current_start = initial_chunks[0].start_char
        
        for i, sim in enumerate(similarities):
            next_chunk = initial_chunks[i + 1]
            current_length = sum(len(t) for t in current_texts)
            
            # Merge if similar and not too large
            if sim >= self.similarity_threshold and current_length + len(next_chunk.text) <= self.max_chunk_size:
                current_texts.append(next_chunk.text)
            else:
                # Create chunk from accumulated texts
                merged_text = " ".join(current_texts)
                chunk = Chunk(
                    chunk_id=md5(f"{merged_text[:50]}_{len(final_chunks)}".encode()).hexdigest()[:12],
                    text=merged_text,
                    start_char=current_start,
                    end_char=current_start + len(merged_text),
                    metadata={
                        **metadata,
                        "chunk_index": len(final_chunks),
                        "chunking_method": "semantic"
                    }
                )
                final_chunks.append(chunk)
                
                # Start new chunk
                current_texts = [next_chunk.text]
                current_start = next_chunk.start_char
        
        # Don't forget last chunk
        if current_texts:
            merged_text = " ".join(current_texts)
            chunk = Chunk(
                chunk_id=md5(f"{merged_text[:50]}_{len(final_chunks)}".encode()).hexdigest()[:12],
                text=merged_text,
                start_char=current_start,
                end_char=current_start + len(merged_text),
                metadata={
                    **metadata,
                    "chunk_index": len(final_chunks),
                    "chunking_method": "semantic"
                }
            )
            final_chunks.append(chunk)
        
        self.logger.info(f"Created {len(final_chunks)} semantic chunks from {len(initial_chunks)} initial chunks")
        return final_chunks


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Text Chunking Test")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    args = parser.parse_args()
    
    if args.test:
        print("Text Chunker Test\n" + "=" * 50)
        
        sample_text = """
        Machine learning is a subset of artificial intelligence that enables systems 
        to learn and improve from experience. Deep learning, a specialized form of 
        machine learning, uses neural networks with multiple layers.
        
        Natural language processing allows computers to understand human language. 
        This technology powers chatbots, translation services, and sentiment analysis.
        
        Computer vision enables machines to interpret visual information from the world. 
        Applications include facial recognition, autonomous vehicles, and medical imaging.
        """
        
        chunker = TextChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(sample_text.strip())
        
        print(f"\nCreated {len(chunks)} chunks:\n")
        for chunk in chunks:
            print(f"Chunk {chunk.metadata['chunk_index']}:")
            print(f"  ID: {chunk.chunk_id}")
            print(f"  Length: {len(chunk)} chars, ~{chunk.token_estimate} tokens")
            print(f"  Text: {chunk.text[:100]}...")
            print()
