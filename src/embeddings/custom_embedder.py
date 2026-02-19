"""
Custom Embedding Module for Vector Generation.
Generates high-quality embeddings using sentence transformers.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from ..utils import get_logger, get_config, LoggerMixin

logger = get_logger(__name__)
config = get_config()


@dataclass
class EmbeddingResult:
    """Container for embedding results."""
    
    embeddings: np.ndarray
    texts: List[str]
    metadata: Dict = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    @property
    def dimension(self) -> int:
        return self.embeddings.shape[1] if len(self.embeddings.shape) > 1 else 0
    
    def to_dict(self) -> Dict:
        return {
            "embeddings": self.embeddings.tolist(),
            "texts": self.texts,
            "dimension": self.dimension,
            "count": len(self),
            "metadata": self.metadata
        }


class CustomEmbedder(LoggerMixin):
    """
    Custom embedding pipeline using sentence transformers.
    
    Supports:
    - Multiple embedding models (all-mpnet, bge, e5)
    - Batch processing with GPU acceleration
    - Mixed precision (FP16) for efficiency
    - Embedding normalization
    - Caching for repeated texts
    """
    
    # Available embedding models with their dimensions
    AVAILABLE_MODELS = {
        "sentence-transformers/all-mpnet-base-v2": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "BAAI/bge-large-en-v1.5": 1024,
        "BAAI/bge-base-en-v1.5": 768,
        "intfloat/e5-large-v2": 1024,
        "intfloat/e5-base-v2": 768
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = None,
        max_seq_length: int = None,
        use_fp16: bool = None,
        normalize_embeddings: bool = None,
        cache_embeddings: bool = True
    ):
        """
        Initialize embedding model.
        
        Args:
            model_name: Sentence transformer model name
            device: Device to use ("cuda" or "cpu")
            batch_size: Batch size for encoding
            max_seq_length: Maximum sequence length
            use_fp16: Whether to use FP16 precision
            normalize_embeddings: Whether to L2-normalize embeddings
            cache_embeddings: Whether to cache computed embeddings
        """
        self.model_name = model_name or config.embedding.model_name
        self.device = device or config.embedding.device
        self.batch_size = batch_size or config.embedding.batch_size
        self.max_seq_length = max_seq_length or config.embedding.max_seq_length
        self.use_fp16 = use_fp16 if use_fp16 is not None else config.embedding.use_fp16
        self.normalize_embeddings = (
            normalize_embeddings if normalize_embeddings is not None 
            else config.embedding.normalize_embeddings
        )
        self.cache_embeddings = cache_embeddings
        
        self.model = None
        self._cache: Dict[str, np.ndarray] = {}
        
        # Get embedding dimension
        self.embedding_dim = self.AVAILABLE_MODELS.get(
            self.model_name, 
            config.embedding.embedding_dim
        )
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is not None:
            return
        
        self.logger.info(f"Loading embedding model: {self.model_name}")
        
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Check device availability
            if self.device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
            
            # Load model
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Set max sequence length
            self.model.max_seq_length = self.max_seq_length
            
            # Enable FP16 if supported
            if self.use_fp16 and self.device == "cuda":
                self.model.half()
                self.logger.debug("Using FP16 precision")
            
            self.logger.info(
                f"Model loaded: {self.model_name} "
                f"(dim={self.embedding_dim}, device={self.device})"
            )
            
        except ImportError:
            self.logger.error("sentence-transformers not installed")
            raise ImportError(
                "Please install sentence-transformers: pip install sentence-transformers"
            )
    
    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        convert_to_numpy: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts
            show_progress: Whether to show progress bar
            convert_to_numpy: Whether to convert to numpy array
            
        Returns:
            Embedding array of shape (n_texts, embedding_dim)
        """
        self._load_model()
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache
        if self.cache_embeddings:
            cached_results = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                if text in self._cache:
                    cached_results.append((i, self._cache[text]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            if not uncached_texts:
                # All texts were cached
                embeddings = np.array([emb for _, emb in sorted(cached_results)])
                return embeddings
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        self.logger.debug(f"Encoding {len(uncached_texts)} texts")
        
        # Encode uncached texts
        embeddings = self.model.encode(
            uncached_texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=self.normalize_embeddings
        )
        
        # Update cache
        if self.cache_embeddings:
            for text, emb in zip(uncached_texts, embeddings):
                self._cache[text] = emb
            
            # Combine cached and new embeddings
            all_embeddings = np.zeros((len(texts), embeddings.shape[1]))
            for idx, emb in cached_results:
                all_embeddings[idx] = emb
            for i, idx in enumerate(uncached_indices):
                all_embeddings[idx] = embeddings[i]
            
            embeddings = all_embeddings
        
        return embeddings
    
    def encode_with_metadata(
        self,
        texts: List[str],
        metadata: Optional[Dict] = None,
        show_progress: bool = False
    ) -> EmbeddingResult:
        """
        Encode texts and return with metadata.
        
        Args:
            texts: List of texts to encode
            metadata: Optional metadata to include
            show_progress: Whether to show progress bar
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        embeddings = self.encode(texts, show_progress=show_progress)
        
        result_metadata = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "normalized": self.normalize_embeddings,
            "device": self.device,
            **(metadata or {})
        }
        
        return EmbeddingResult(
            embeddings=embeddings,
            texts=texts,
            metadata=result_metadata
        )
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts in batches for memory efficiency.
        
        Args:
            texts: List of texts to encode
            batch_size: Override default batch size
            show_progress: Whether to show progress bar
            
        Returns:
            Embedding array
        """
        batch_size = batch_size or self.batch_size
        return self.encode(texts, show_progress=show_progress)
    
    def similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Calculate similarity between embedding sets.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            metric: Similarity metric ("cosine" or "dot")
            
        Returns:
            Similarity matrix
        """
        if metric == "cosine":
            # Normalize if not already
            if not self.normalize_embeddings:
                embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
                embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
            return np.dot(embeddings1, embeddings2.T)
        
        elif metric == "dot":
            return np.dot(embeddings1, embeddings2.T)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        self.logger.debug("Embedding cache cleared")
    
    @property
    def cache_size(self) -> int:
        """Return number of cached embeddings."""
        return len(self._cache)
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        path: Union[str, Path],
        texts: Optional[List[str]] = None
    ):
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Embedding array
            path: Save path
            texts: Optional texts to save alongside
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(path, embeddings)
        self.logger.info(f"Saved embeddings to {path}")
        
        if texts:
            texts_path = path.with_suffix('.texts.npy')
            np.save(texts_path, np.array(texts, dtype=object))
    
    def load_embeddings(
        self,
        path: Union[str, Path],
        load_texts: bool = False
    ) -> Union[np.ndarray, tuple]:
        """
        Load embeddings from disk.
        
        Args:
            path: Load path
            load_texts: Whether to also load texts
            
        Returns:
            Embeddings array, or (embeddings, texts) tuple
        """
        path = Path(path)
        embeddings = np.load(path)
        self.logger.info(f"Loaded embeddings from {path}")
        
        if load_texts:
            texts_path = path.with_suffix('.texts.npy')
            if texts_path.exists():
                texts = np.load(texts_path, allow_pickle=True).tolist()
                return embeddings, texts
        
        return embeddings


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Custom Embedder Test")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    args = parser.parse_args()
    
    if args.test:
        print("Custom Embedder Test\n" + "=" * 50)
        
        # Initialize embedder (will use CPU if CUDA not available)
        embedder = CustomEmbedder(device="cpu")
        
        # Test texts
        texts = [
            "Machine learning is transforming industries.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing enables text understanding.",
            "The weather is sunny today."  # Unrelated text
        ]
        
        # Encode
        embeddings = embedder.encode(texts)
        
        print(f"Model: {embedder.model_name}")
        print(f"Embedding dimension: {embedder.embedding_dim}")
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Calculate similarities
        similarities = embedder.similarity(embeddings, embeddings)
        
        print("\nSimilarity Matrix:")
        print(np.round(similarities, 2))
        
        print("\nCache size:", embedder.cache_size)
