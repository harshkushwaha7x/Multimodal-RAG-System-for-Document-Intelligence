"""
Configuration module for Multimodal RAG System.
Centralized settings using Pydantic for validation.
"""

from pathlib import Path
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class PathConfig(BaseSettings):
    """File and directory paths."""
    
    base_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "artifacts" / "models")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "logs")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""
    
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dim: int = 768
    reduced_dim: int = 256
    batch_size: int = 32
    max_seq_length: int = 512
    use_fp16: bool = True
    normalize_embeddings: bool = True
    device: str = "cuda"  # "cuda" or "cpu"


class ChunkingConfig(BaseSettings):
    """Text chunking configuration."""
    
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 800
    separator: str = "\n\n"


class RetrievalConfig(BaseSettings):
    """Retrieval configuration."""
    
    top_k: int = 20
    final_k: int = 5
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rrf_k: int = 60
    similarity_threshold: float = 0.5
    
    # BM25 parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


class LLMConfig(BaseSettings):
    """LLM configuration."""
    
    model_name: str = "google/flan-t5-base"
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    do_sample: bool = True
    context_window: int = 4096
    
    # Alternative models (open models that don't require auth)
    available_models: list = [
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "MBZUAI/LaMini-Flan-T5-248M",
        "facebook/opt-1.3b"
    ]


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    # PostgreSQL
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "rag_db"
    pg_user: str = "postgres"
    pg_password: str = "postgres"
    
    # pgvector
    index_type: Literal["ivfflat", "hnsw"] = "ivfflat"
    num_lists: int = 100
    probes: int = 10
    
    # FAISS
    use_faiss: bool = False
    faiss_index_type: Literal["flat", "ivf", "hnsw"] = "ivf"
    faiss_nlist: int = 100
    faiss_nprobe: int = 10


class EvaluationConfig(BaseSettings):
    """Evaluation configuration."""
    
    # Retrieval metrics
    retrieval_k_values: list = [1, 3, 5, 10, 20]
    
    # Generation metrics
    rouge_types: list = ["rouge1", "rouge2", "rougeL"]
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"
    
    # Hallucination detection
    hallucination_threshold: float = 0.7
    entailment_model: str = "microsoft/deberta-large-mnli"


class MLflowConfig(BaseSettings):
    """MLflow configuration."""
    
    tracking_uri: str = "mlruns"
    experiment_name: str = "multimodal-rag"
    artifact_location: Optional[str] = None


class Config(BaseSettings):
    """Main configuration class."""
    
    paths: PathConfig = Field(default_factory=PathConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    
    # General
    seed: int = 42
    debug: bool = False


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config
