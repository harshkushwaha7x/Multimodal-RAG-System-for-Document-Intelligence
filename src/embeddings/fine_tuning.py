"""
Embedding Fine-Tuning Module.
Fine-tune embedding models on domain-specific data.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import random

import numpy as np

from ..utils import get_logger, get_config, LoggerMixin

logger = get_logger(__name__)
config = get_config()


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    output_dir: str = "artifacts/models/fine_tuned"
    
    # Training parameters
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Loss function
    loss_type: str = "multiple_negatives_ranking"  # or "triplet", "contrastive"
    
    # Data parameters
    max_seq_length: int = 512
    num_negatives: int = 5
    
    # Evaluation
    eval_steps: int = 100
    save_steps: int = 500
    
    # Mixed precision
    use_fp16: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "loss_type": self.loss_type,
            "max_seq_length": self.max_seq_length
        }


@dataclass
class TrainingPair:
    """A training pair for fine-tuning."""
    
    query: str
    positive: str
    negatives: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "positive": self.positive,
            "negatives": self.negatives
        }


class EmbeddingFineTuner(LoggerMixin):
    """
    Fine-tune embedding models for domain adaptation.
    
    Supports:
    - Multiple Negatives Ranking Loss
    - Triplet Loss
    - Contrastive Loss
    - Hard negative mining
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize fine-tuner.
        
        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.model = None
        self.train_loss = None
    
    def _load_model(self):
        """Load the base model."""
        if self.model is not None:
            return
        
        self.logger.info(f"Loading base model: {self.config.model_name}")
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.config.model_name)
            self.model.max_seq_length = self.config.max_seq_length
            self.logger.info("Base model loaded successfully")
            
        except ImportError:
            self.logger.error("sentence-transformers not installed")
            raise
    
    def _setup_loss(self):
        """Setup the loss function."""
        from sentence_transformers import losses
        
        if self.config.loss_type == "multiple_negatives_ranking":
            self.train_loss = losses.MultipleNegativesRankingLoss(self.model)
            self.logger.info("Using Multiple Negatives Ranking Loss")
            
        elif self.config.loss_type == "triplet":
            self.train_loss = losses.TripletLoss(self.model)
            self.logger.info("Using Triplet Loss")
            
        elif self.config.loss_type == "contrastive":
            self.train_loss = losses.ContrastiveLoss(self.model)
            self.logger.info("Using Contrastive Loss")
            
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
    
    def create_training_data(
        self,
        documents: List[str],
        queries: Optional[List[str]] = None,
        relevance_scores: Optional[List[List[int]]] = None
    ) -> List[TrainingPair]:
        """
        Create training pairs from documents and queries.
        
        Args:
            documents: List of document texts
            queries: Optional list of queries
            relevance_scores: Optional relevance matrix (queries x documents)
            
        Returns:
            List of TrainingPair objects
        """
        self.logger.info("Creating training data...")
        
        if queries and relevance_scores:
            # Use provided relevance scores
            return self._create_pairs_from_relevance(
                queries, documents, relevance_scores
            )
        else:
            # Create synthetic pairs
            return self._create_synthetic_pairs(documents)
    
    def _create_pairs_from_relevance(
        self,
        queries: List[str],
        documents: List[str],
        relevance: List[List[int]]
    ) -> List[TrainingPair]:
        """Create pairs from relevance matrix."""
        pairs = []
        
        for i, query in enumerate(queries):
            positives = [documents[j] for j, rel in enumerate(relevance[i]) if rel > 0]
            negatives = [documents[j] for j, rel in enumerate(relevance[i]) if rel == 0]
            
            for pos in positives:
                # Sample negatives
                neg_sample = random.sample(
                    negatives, 
                    min(self.config.num_negatives, len(negatives))
                )
                pairs.append(TrainingPair(
                    query=query,
                    positive=pos,
                    negatives=neg_sample
                ))
        
        self.logger.info(f"Created {len(pairs)} training pairs")
        return pairs
    
    def _create_synthetic_pairs(
        self,
        documents: List[str]
    ) -> List[TrainingPair]:
        """
        Create synthetic training pairs using document structure.
        
        Uses sentences from the same document as positives and
        sentences from different documents as negatives.
        """
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            from nltk.tokenize import sent_tokenize
        except ImportError:
            self.logger.warning("NLTK not available, using simple splitting")
            sent_tokenize = lambda x: x.split('. ')
        
        pairs = []
        
        # Group sentences by document
        doc_sentences = []
        for doc in documents:
            sentences = sent_tokenize(doc)
            if len(sentences) >= 2:
                doc_sentences.append(sentences)
        
        # Create pairs
        for doc_idx, sentences in enumerate(doc_sentences):
            for i in range(len(sentences) - 1):
                query = sentences[i]
                positive = sentences[i + 1]
                
                # Get negatives from other documents
                other_docs = [s for j, sents in enumerate(doc_sentences) 
                             for s in sents if j != doc_idx]
                negatives = random.sample(
                    other_docs,
                    min(self.config.num_negatives, len(other_docs))
                )
                
                pairs.append(TrainingPair(
                    query=query,
                    positive=positive,
                    negatives=negatives
                ))
        
        self.logger.info(f"Created {len(pairs)} synthetic training pairs")
        return pairs
    
    def mine_hard_negatives(
        self,
        queries: List[str],
        documents: List[str],
        positives_idx: List[int],
        top_k: int = 10
    ) -> List[List[str]]:
        """
        Mine hard negatives using embedding similarity.
        
        Hard negatives are documents that are similar to the query
        but are not the positive document.
        
        Args:
            queries: List of query texts
            documents: List of document texts
            positives_idx: Index of positive document for each query
            top_k: Number of hard negatives to mine per query
            
        Returns:
            List of hard negatives for each query
        """
        self._load_model()
        
        self.logger.info("Mining hard negatives...")
        
        # Encode all texts
        query_embeddings = self.model.encode(queries, show_progress_bar=True)
        doc_embeddings = self.model.encode(documents, show_progress_bar=True)
        
        # Calculate similarities
        similarities = np.dot(query_embeddings, doc_embeddings.T)
        
        hard_negatives = []
        for i, pos_idx in enumerate(positives_idx):
            # Get top-k most similar documents
            top_indices = np.argsort(similarities[i])[::-1]
            
            # Filter out the positive and get hard negatives
            neg_indices = [idx for idx in top_indices 
                          if idx != pos_idx][:top_k]
            hard_negatives.append([documents[idx] for idx in neg_indices])
        
        self.logger.info(f"Mined {top_k} hard negatives per query")
        return hard_negatives
    
    def prepare_dataloader(
        self,
        training_pairs: List[TrainingPair]
    ):
        """
        Prepare data loader for training.
        
        Args:
            training_pairs: List of training pairs
            
        Returns:
            DataLoader for training
        """
        from sentence_transformers import InputExample
        from torch.utils.data import DataLoader
        
        examples = []
        for pair in training_pairs:
            if self.config.loss_type == "multiple_negatives_ranking":
                # For MNR loss, we just need (query, positive) pairs
                examples.append(InputExample(
                    texts=[pair.query, pair.positive]
                ))
            elif self.config.loss_type == "triplet":
                # For triplet loss, we need (anchor, positive, negative)
                for neg in pair.negatives:
                    examples.append(InputExample(
                        texts=[pair.query, pair.positive, neg]
                    ))
            elif self.config.loss_type == "contrastive":
                # For contrastive loss, we need (text1, text2, label)
                examples.append(InputExample(
                    texts=[pair.query, pair.positive],
                    label=1.0
                ))
                for neg in pair.negatives:
                    examples.append(InputExample(
                        texts=[pair.query, neg],
                        label=0.0
                    ))
        
        dataloader = DataLoader(
            examples,
            shuffle=True,
            batch_size=self.config.batch_size
        )
        
        self.logger.info(f"Created dataloader with {len(examples)} examples")
        return dataloader
    
    def train(
        self,
        training_pairs: List[TrainingPair],
        eval_pairs: Optional[List[TrainingPair]] = None,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Fine-tune the embedding model.
        
        Args:
            training_pairs: Training data
            eval_pairs: Optional evaluation data
            output_dir: Directory to save the model
            
        Returns:
            Training metrics
        """
        self._load_model()
        self._setup_loss()
        
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        train_dataloader = self.prepare_dataloader(training_pairs)
        
        # Calculate training steps
        num_training_steps = len(train_dataloader) * self.config.epochs
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        self.logger.info(f"Starting training for {self.config.epochs} epochs")
        self.logger.info(f"Total steps: {num_training_steps}, Warmup: {warmup_steps}")
        
        # Train
        self.model.fit(
            train_objectives=[(train_dataloader, self.train_loss)],
            epochs=self.config.epochs,
            warmup_steps=warmup_steps,
            output_path=str(output_dir),
            show_progress_bar=True,
            use_amp=self.config.use_fp16
        )
        
        # Save training config
        config_path = output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        self.logger.info(f"Model saved to {output_dir}")
        
        return {
            "output_dir": str(output_dir),
            "epochs": self.config.epochs,
            "training_samples": len(training_pairs),
            "total_steps": num_training_steps
        }
    
    def evaluate(
        self,
        eval_pairs: List[TrainingPair],
        model_path: Optional[str] = None
    ) -> Dict:
        """
        Evaluate fine-tuned model on evaluation pairs.
        
        Args:
            eval_pairs: Evaluation data
            model_path: Path to fine-tuned model
            
        Returns:
            Evaluation metrics
        """
        from sentence_transformers import SentenceTransformer
        
        if model_path:
            model = SentenceTransformer(model_path)
        else:
            self._load_model()
            model = self.model
        
        # Encode queries and positives
        queries = [p.query for p in eval_pairs]
        positives = [p.positive for p in eval_pairs]
        
        query_embeddings = model.encode(queries)
        positive_embeddings = model.encode(positives)
        
        # Calculate cosine similarities
        similarities = np.diag(
            np.dot(query_embeddings, positive_embeddings.T) /
            (np.linalg.norm(query_embeddings, axis=1, keepdims=True) *
             np.linalg.norm(positive_embeddings, axis=1, keepdims=True).T)
        )
        
        metrics = {
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities))
        }
        
        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Embedding Fine-Tuning Test")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    args = parser.parse_args()
    
    if args.test:
        print("Embedding Fine-Tuner Test\n" + "=" * 50)
        
        # Create sample training data
        documents = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            "Deep learning uses neural networks with multiple layers to model complex patterns.",
            "Natural language processing enables computers to understand human language.",
            "Computer vision allows machines to interpret visual information from images and videos.",
            "Reinforcement learning trains agents through trial and error using rewards."
        ]
        
        # Initialize fine-tuner
        config = TrainingConfig(
            epochs=1,
            batch_size=2,
            loss_type="multiple_negatives_ranking"
        )
        fine_tuner = EmbeddingFineTuner(config)
        
        # Create synthetic training pairs
        pairs = fine_tuner.create_training_data(documents)
        
        print(f"Created {len(pairs)} training pairs")
        print(f"\nSample pair:")
        print(f"  Query: {pairs[0].query[:50]}...")
        print(f"  Positive: {pairs[0].positive[:50]}...")
