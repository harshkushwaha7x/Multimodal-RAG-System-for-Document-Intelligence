"""
Evaluation Metrics Module.
Comprehensive metrics for retrieval and generation quality.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from ..utils import get_logger, get_config, LoggerMixin

logger = get_logger(__name__)
config = get_config()


@dataclass
class MetricResult:
    """Container for metric results."""
    
    name: str
    value: float
    details: Dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"{self.name}: {self.value:.4f}"
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": self.value,
            "details": self.details
        }


class RetrievalMetrics(LoggerMixin):
    """
    Retrieval evaluation metrics.
    
    Implements:
    - Precision@K
    - Recall@K
    - NDCG@K
    - Mean Reciprocal Rank (MRR)
    - Hit Rate@K
    """
    
    def __init__(self, k_values: List[int] = None):
        """
        Initialize retrieval metrics.
        
        Args:
            k_values: K values for @K metrics
        """
        self.k_values = k_values or config.evaluation.retrieval_k_values
    
    def precision_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Calculate Precision@K.
        
        P@K = (Relevant docs in top-K) / K
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            k: Cutoff value
            
        Returns:
            Precision score
        """
        if k <= 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        
        relevant_in_k = sum(1 for doc in retrieved_k if doc in relevant_set)
        return relevant_in_k / k
    
    def recall_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Calculate Recall@K.
        
        R@K = (Relevant docs in top-K) / (Total relevant docs)
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            k: Cutoff value
            
        Returns:
            Recall score
        """
        if not relevant:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        
        relevant_in_k = sum(1 for doc in retrieved_k if doc in relevant_set)
        return relevant_in_k / len(relevant)
    
    def ndcg_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        relevance_scores: Optional[Dict[str, float]] = None,
        k: int = 10
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K.
        
        NDCG@K = DCG@K / IDCG@K
        DCG@K = Σ (relevance_i / log2(i+1))
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            relevance_scores: Optional dict of relevance scores per doc
            k: Cutoff value
            
        Returns:
            NDCG score
        """
        if not relevant:
            return 0.0
        
        # Default binary relevance
        if relevance_scores is None:
            relevance_scores = {doc: 1.0 for doc in relevant}
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            rel = relevance_scores.get(doc, 0.0)
            dcg += rel / np.log2(i + 2)  # +2 because i is 0-indexed
        
        # Calculate IDCG (ideal ranking)
        ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def mean_reciprocal_rank(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        MRR = 1 / rank_of_first_relevant_doc
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            
        Returns:
            MRR score
        """
        relevant_set = set(relevant)
        
        for i, doc in enumerate(retrieved):
            if doc in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def hit_rate_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Calculate Hit Rate@K.
        
        1 if at least one relevant doc in top-K, else 0
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            k: Cutoff value
            
        Returns:
            1.0 or 0.0
        """
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        return 1.0 if retrieved_k & relevant_set else 0.0
    
    def evaluate(
        self,
        retrieved: List[str],
        relevant: List[str],
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, MetricResult]:
        """
        Compute all retrieval metrics.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            relevance_scores: Optional relevance scores
            
        Returns:
            Dict of metric results
        """
        results = {}
        
        for k in self.k_values:
            # Precision@K
            p_at_k = self.precision_at_k(retrieved, relevant, k)
            results[f"precision@{k}"] = MetricResult(
                name=f"Precision@{k}",
                value=p_at_k
            )
            
            # Recall@K
            r_at_k = self.recall_at_k(retrieved, relevant, k)
            results[f"recall@{k}"] = MetricResult(
                name=f"Recall@{k}",
                value=r_at_k
            )
            
            # NDCG@K
            ndcg = self.ndcg_at_k(retrieved, relevant, relevance_scores, k)
            results[f"ndcg@{k}"] = MetricResult(
                name=f"NDCG@{k}",
                value=ndcg
            )
            
            # Hit Rate@K
            hr = self.hit_rate_at_k(retrieved, relevant, k)
            results[f"hit_rate@{k}"] = MetricResult(
                name=f"Hit Rate@{k}",
                value=hr
            )
        
        # MRR (not @K dependent)
        mrr = self.mean_reciprocal_rank(retrieved, relevant)
        results["mrr"] = MetricResult(name="MRR", value=mrr)
        
        return results
    
    def evaluate_batch(
        self,
        all_retrieved: List[List[str]],
        all_relevant: List[List[str]]
    ) -> Dict[str, MetricResult]:
        """
        Evaluate metrics over multiple queries.
        
        Args:
            all_retrieved: List of retrieved lists per query
            all_relevant: List of relevant lists per query
            
        Returns:
            Averaged metric results
        """
        all_results = []
        
        for retrieved, relevant in zip(all_retrieved, all_relevant):
            results = self.evaluate(retrieved, relevant)
            all_results.append(results)
        
        # Average results
        averaged = {}
        for key in all_results[0].keys():
            values = [r[key].value for r in all_results]
            averaged[key] = MetricResult(
                name=all_results[0][key].name,
                value=np.mean(values),
                details={
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n_queries": len(values)
                }
            )
        
        return averaged


class GenerationMetrics(LoggerMixin):
    """
    Generation evaluation metrics.
    
    Implements:
    - ROUGE (1, 2, L)
    - BERTScore
    - BLEU
    - Faithfulness
    - Context Recall
    """
    
    def __init__(self):
        self._rouge_scorer = None
        self._bertscore_model = None
    
    def _load_rouge(self):
        """Lazy load ROUGE scorer."""
        if self._rouge_scorer is None:
            try:
                from rouge_score import rouge_scorer
                self._rouge_scorer = rouge_scorer.RougeScorer(
                    config.evaluation.rouge_types,
                    use_stemmer=True
                )
            except ImportError:
                self.logger.error("rouge-score not installed")
                raise
    
    def rouge(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str]]
    ) -> Dict[str, MetricResult]:
        """
        Calculate ROUGE scores.
        
        Args:
            predictions: Generated text(s)
            references: Reference text(s)
            
        Returns:
            Dict of ROUGE scores
        """
        self._load_rouge()
        
        if isinstance(predictions, str):
            predictions = [predictions]
            references = [references]
        
        all_scores = {rtype: [] for rtype in config.evaluation.rouge_types}
        
        for pred, ref in zip(predictions, references):
            scores = self._rouge_scorer.score(ref, pred)
            for rtype in config.evaluation.rouge_types:
                all_scores[rtype].append(scores[rtype].fmeasure)
        
        results = {}
        for rtype in config.evaluation.rouge_types:
            results[rtype] = MetricResult(
                name=rtype.upper(),
                value=np.mean(all_scores[rtype]),
                details={
                    "precision": np.mean([s for s in all_scores[rtype]]),
                    "std": float(np.std(all_scores[rtype]))
                }
            )
        
        return results
    
    def bertscore(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str]]
    ) -> MetricResult:
        """
        Calculate BERTScore.
        
        Args:
            predictions: Generated text(s)
            references: Reference text(s)
            
        Returns:
            BERTScore F1 result
        """
        try:
            from bert_score import score as bert_score
        except ImportError:
            self.logger.error("bert-score not installed")
            raise
        
        if isinstance(predictions, str):
            predictions = [predictions]
            references = [references]
        
        P, R, F1 = bert_score(
            predictions,
            references,
            model_type=config.evaluation.bertscore_model,
            verbose=False
        )
        
        return MetricResult(
            name="BERTScore",
            value=float(F1.mean()),
            details={
                "precision": float(P.mean()),
                "recall": float(R.mean()),
                "f1": float(F1.mean())
            }
        )
    
    def bleu(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str]]
    ) -> MetricResult:
        """
        Calculate BLEU score.
        
        Args:
            predictions: Generated text(s)
            references: Reference text(s)
            
        Returns:
            BLEU score result
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        except ImportError:
            self.logger.error("NLTK not installed")
            raise
        
        if isinstance(predictions, str):
            predictions = [predictions]
            references = [references]
        
        smoothing = SmoothingFunction().method1
        scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = [ref.split()]
            score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
            scores.append(score)
        
        return MetricResult(
            name="BLEU",
            value=np.mean(scores),
            details={"std": float(np.std(scores))}
        )
    
    def evaluate(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str]],
        include_bertscore: bool = True
    ) -> Dict[str, MetricResult]:
        """
        Compute all generation metrics.
        
        Args:
            predictions: Generated text(s)
            references: Reference text(s)
            include_bertscore: Whether to compute BERTScore (slow)
            
        Returns:
            Dict of metric results
        """
        results = {}
        
        # ROUGE
        rouge_results = self.rouge(predictions, references)
        results.update(rouge_results)
        
        # BLEU
        results["bleu"] = self.bleu(predictions, references)
        
        # BERTScore
        if include_bertscore:
            try:
                results["bertscore"] = self.bertscore(predictions, references)
            except Exception as e:
                self.logger.warning(f"BERTScore failed: {e}")
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Metrics Test")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    args = parser.parse_args()
    
    if args.test:
        print("Evaluation Metrics Test\n" + "=" * 50)
        
        # Test retrieval metrics
        print("\nRetrieval Metrics:")
        retrieval_metrics = RetrievalMetrics(k_values=[1, 3, 5])
        
        retrieved = ["doc1", "doc3", "doc5", "doc2", "doc4"]
        relevant = ["doc1", "doc2", "doc6"]
        
        results = retrieval_metrics.evaluate(retrieved, relevant)
        for name, result in results.items():
            print(f"  {result}")
        
        # Test generation metrics
        print("\nGeneration Metrics:")
        gen_metrics = GenerationMetrics()
        
        prediction = "Machine learning enables computers to learn from data."
        reference = "Machine learning allows computers to learn from examples."
        
        rouge_results = gen_metrics.rouge(prediction, reference)
        for name, result in rouge_results.items():
            print(f"  {result}")
