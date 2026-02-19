"""
Dimensionality Reduction Module.
Reduce embedding dimensions for storage and speed optimization.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import pickle

import numpy as np

from ..utils import get_logger, get_config, LoggerMixin

logger = get_logger(__name__)
config = get_config()


class DimensionalityReducer(LoggerMixin):
    """Base class for dimensionality reduction."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_fitted = False
    
    def fit(self, embeddings: np.ndarray) -> 'DimensionalityReducer':
        """Fit the reducer on embeddings."""
        raise NotImplementedError
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to lower dimension."""
        raise NotImplementedError
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(embeddings)
        return self.transform(embeddings)
    
    def save(self, path: Union[str, Path]):
        """Save the fitted reducer."""
        raise NotImplementedError
    
    def load(self, path: Union[str, Path]) -> 'DimensionalityReducer':
        """Load a fitted reducer."""
        raise NotImplementedError


class PCAReducer(DimensionalityReducer):
    """
    PCA-based dimensionality reduction.
    
    Reduces embedding dimensions while preserving variance.
    Recommended for:
    - Storage optimization (3-6x reduction)
    - Faster similarity search
    - Lower memory usage
    """
    
    def __init__(
        self,
        input_dim: int = None,
        output_dim: int = None,
        variance_threshold: float = 0.9,
        whiten: bool = False
    ):
        """
        Initialize PCA reducer.
        
        Args:
            input_dim: Original embedding dimension
            output_dim: Target dimension (if None, determined by variance)
            variance_threshold: Min explained variance ratio
            whiten: Whether to whiten the output
        """
        input_dim = input_dim or config.embedding.embedding_dim
        output_dim = output_dim or config.embedding.reduced_dim
        
        super().__init__(input_dim, output_dim)
        
        self.variance_threshold = variance_threshold
        self.whiten = whiten
        self.pca = None
        self.explained_variance_ratio = None
        self.cumulative_variance = None
    
    def fit(self, embeddings: np.ndarray) -> 'PCAReducer':
        """
        Fit PCA on embeddings.
        
        Args:
            embeddings: Training embeddings (n_samples, n_features)
            
        Returns:
            Self
        """
        from sklearn.decomposition import PCA
        
        self.logger.info(f"Fitting PCA: {embeddings.shape[1]} -> {self.output_dim} dimensions")
        
        # Initialize PCA
        self.pca = PCA(
            n_components=self.output_dim,
            whiten=self.whiten,
            random_state=config.seed
        )
        
        # Fit
        self.pca.fit(embeddings)
        
        # Store variance information
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.cumulative_variance = np.cumsum(self.explained_variance_ratio)
        
        total_variance = self.cumulative_variance[-1]
        self.logger.info(f"Explained variance: {total_variance:.2%}")
        
        # Check if variance threshold is met
        if total_variance < self.variance_threshold:
            n_components = np.argmax(
                np.cumsum(self.pca.explained_variance_ratio_) >= self.variance_threshold
            ) + 1
            self.logger.warning(
                f"Variance threshold {self.variance_threshold:.0%} not met. "
                f"Need {n_components} components for {self.variance_threshold:.0%} variance."
            )
        
        self.is_fitted = True
        return self
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings to lower dimension.
        
        Args:
            embeddings: Input embeddings (n_samples, n_features)
            
        Returns:
            Reduced embeddings (n_samples, output_dim)
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        return self.pca.transform(embeddings)
    
    def inverse_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reconstruct original embeddings from reduced form.
        
        Args:
            embeddings: Reduced embeddings
            
        Returns:
            Reconstructed embeddings
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        return self.pca.inverse_transform(embeddings)
    
    def get_statistics(self) -> Dict:
        """Get PCA statistics."""
        if not self.is_fitted:
            return {}
        
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "total_variance": float(self.cumulative_variance[-1]),
            "variance_per_component": self.explained_variance_ratio.tolist(),
            "cumulative_variance": self.cumulative_variance.tolist(),
            "compression_ratio": self.input_dim / self.output_dim
        }
    
    def optimal_dimensions(
        self,
        variance_targets: list = [0.8, 0.85, 0.9, 0.95, 0.99]
    ) -> Dict[float, int]:
        """
        Find optimal dimensions for different variance targets.
        
        Args:
            variance_targets: List of variance ratio targets
            
        Returns:
            Dict mapping variance target to required dimensions
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        cumsum = np.cumsum(self.explained_variance_ratio)
        
        result = {}
        for target in variance_targets:
            # Find smallest n where cumulative sum >= target
            n_components = np.argmax(cumsum >= target) + 1
            if cumsum[n_components - 1] < target:
                n_components = len(cumsum)  # Need all components
            result[target] = int(n_components)
        
        return result
    
    def save(self, path: Union[str, Path]):
        """Save fitted PCA model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'pca': self.pca,
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'variance_threshold': self.variance_threshold,
                'explained_variance_ratio': self.explained_variance_ratio,
                'cumulative_variance': self.cumulative_variance,
                'is_fitted': self.is_fitted
            }, f)
        
        self.logger.info(f"Saved PCA model to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'PCAReducer':
        """Load saved PCA model."""
        path = Path(path)
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        reducer = cls(
            input_dim=data['input_dim'],
            output_dim=data['output_dim'],
            variance_threshold=data['variance_threshold']
        )
        reducer.pca = data['pca']
        reducer.explained_variance_ratio = data['explained_variance_ratio']
        reducer.cumulative_variance = data['cumulative_variance']
        reducer.is_fitted = data['is_fitted']
        
        logger.info(f"Loaded PCA model from {path}")
        return reducer


class UMAPReducer(DimensionalityReducer):
    """
    UMAP-based dimensionality reduction.
    
    Better at preserving local structure than PCA.
    Recommended for visualization and clustering.
    """
    
    def __init__(
        self,
        input_dim: int = None,
        output_dim: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine"
    ):
        """
        Initialize UMAP reducer.
        
        Args:
            input_dim: Original dimension
            output_dim: Target dimension (typically 2 for viz)
            n_neighbors: Number of neighbors for manifold
            min_dist: Minimum distance in embedded space
            metric: Distance metric
        """
        input_dim = input_dim or config.embedding.embedding_dim
        super().__init__(input_dim, output_dim)
        
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.reducer = None
    
    def fit(self, embeddings: np.ndarray) -> 'UMAPReducer':
        """Fit UMAP on embeddings."""
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn not installed. Install with: pip install umap-learn")
        
        self.logger.info(f"Fitting UMAP: {embeddings.shape[1]} -> {self.output_dim} dimensions")
        
        self.reducer = umap.UMAP(
            n_components=self.output_dim,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=config.seed
        )
        
        self.reducer.fit(embeddings)
        self.is_fitted = True
        
        return self
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings using UMAP."""
        if not self.is_fitted:
            raise ValueError("UMAP not fitted. Call fit() first.")
        
        return self.reducer.transform(embeddings)
    
    def save(self, path: Union[str, Path]):
        """Save fitted UMAP model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'reducer': self.reducer,
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'is_fitted': self.is_fitted
            }, f)
        
        self.logger.info(f"Saved UMAP model to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'UMAPReducer':
        """Load saved UMAP model."""
        path = Path(path)
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        reducer = cls(
            input_dim=data['input_dim'],
            output_dim=data['output_dim']
        )
        reducer.reducer = data['reducer']
        reducer.is_fitted = data['is_fitted']
        
        logger.info(f"Loaded UMAP model from {path}")
        return reducer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dimensionality Reduction Test")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    args = parser.parse_args()
    
    if args.test:
        print("Dimensionality Reduction Test\n" + "=" * 50)
        
        # Create sample embeddings
        np.random.seed(42)
        n_samples = 1000
        input_dim = 768
        
        embeddings = np.random.randn(n_samples, input_dim)
        
        # Test PCA
        print("\nPCA Reduction:")
        pca_reducer = PCAReducer(input_dim=768, output_dim=256)
        reduced = pca_reducer.fit_transform(embeddings)
        
        print(f"Original shape: {embeddings.shape}")
        print(f"Reduced shape: {reduced.shape}")
        print(f"Explained variance: {pca_reducer.cumulative_variance[-1]:.2%}")
        print(f"Compression ratio: {input_dim / 256:.1f}x")
        
        # Optimal dimensions
        optimal = pca_reducer.optimal_dimensions()
        print(f"\nOptimal dimensions for variance targets:")
        for var, dim in optimal.items():
            print(f"  {var:.0%} variance: {dim} dimensions")
