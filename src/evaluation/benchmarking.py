"""
Benchmarking Module.
End-to-end RAG evaluation and benchmarking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time

import numpy as np

from .metrics import RetrievalMetrics, GenerationMetrics, MetricResult
from .hallucination_detector import HallucinationDetector, HallucinationResult
from ..utils import get_logger, get_config, LoggerMixin

logger = get_logger(__name__)
config = get_config()


@dataclass
class EvaluationSample:
    """Single evaluation sample."""
    
    query: str
    ground_truth: str
    relevant_docs: List[str]
    metadata: Dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    
    name: str
    timestamp: str
    retrieval_metrics: Dict[str, MetricResult]
    generation_metrics: Dict[str, MetricResult]
    hallucination_rate: float
    latency_stats: Dict[str, float]
    config: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "retrieval_metrics": {k: v.to_dict() for k, v in self.retrieval_metrics.items()},
            "generation_metrics": {k: v.to_dict() for k, v in self.generation_metrics.items()},
            "hallucination_rate": self.hallucination_rate,
            "latency_stats": self.latency_stats,
            "config": self.config
        }
    
    def summary(self) -> str:
        """Generate text summary of results."""
        lines = [
            f"=== Benchmark: {self.name} ===",
            f"Timestamp: {self.timestamp}",
            "",
            "Retrieval Metrics:",
        ]
        
        for name, result in self.retrieval_metrics.items():
            lines.append(f"  {result}")
        
        lines.extend(["", "Generation Metrics:"])
        for name, result in self.generation_metrics.items():
            lines.append(f"  {result}")
        
        lines.extend([
            "",
            f"Hallucination Rate: {self.hallucination_rate:.2%}",
            "",
            "Latency (ms):",
            f"  P50: {self.latency_stats.get('p50', 0):.0f}",
            f"  P95: {self.latency_stats.get('p95', 0):.0f}",
            f"  P99: {self.latency_stats.get('p99', 0):.0f}"
        ])
        
        return "\n".join(lines)
    
    def save(self, path: Path):
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved benchmark results to {path}")


class RAGBenchmark(LoggerMixin):
    """
    Comprehensive RAG system benchmarking.
    
    Evaluates:
    - Retrieval quality (P@K, R@K, NDCG, MRR)
    - Generation quality (ROUGE, BERTScore)
    - Hallucination rate
    - Latency metrics
    """
    
    def __init__(
        self,
        rag_pipeline,
        retrieval_metrics: Optional[RetrievalMetrics] = None,
        generation_metrics: Optional[GenerationMetrics] = None,
        hallucination_detector: Optional[HallucinationDetector] = None
    ):
        """
        Initialize benchmark.
        
        Args:
            rag_pipeline: RAG pipeline to evaluate
            retrieval_metrics: Custom retrieval metrics
            generation_metrics: Custom generation metrics
            hallucination_detector: Custom hallucination detector
        """
        self.rag_pipeline = rag_pipeline
        self.retrieval_metrics = retrieval_metrics or RetrievalMetrics()
        self.generation_metrics = generation_metrics or GenerationMetrics()
        self.hallucination_detector = hallucination_detector or HallucinationDetector()
    
    def load_evaluation_data(
        self,
        path: Path
    ) -> List[EvaluationSample]:
        """
        Load evaluation dataset from file.
        
        Expected format (JSON):
        [
            {
                "query": "...",
                "ground_truth": "...",
                "relevant_docs": ["doc1", "doc2"],
                "metadata": {}
            }
        ]
        
        Args:
            path: Path to evaluation data file
            
        Returns:
            List of EvaluationSample objects
        """
        path = Path(path)
        
        with open(path) as f:
            data = json.load(f)
        
        samples = [
            EvaluationSample(
                query=item["query"],
                ground_truth=item["ground_truth"],
                relevant_docs=item.get("relevant_docs", []),
                metadata=item.get("metadata", {})
            )
            for item in data
        ]
        
        self.logger.info(f"Loaded {len(samples)} evaluation samples")
        return samples
    
    def run(
        self,
        samples: List[EvaluationSample],
        name: str = "benchmark",
        include_bertscore: bool = False,
        verbose: bool = True
    ) -> BenchmarkResult:
        """
        Run complete benchmark.
        
        Args:
            samples: Evaluation samples
            name: Benchmark name
            include_bertscore: Whether to compute BERTScore
            verbose: Print progress
            
        Returns:
            BenchmarkResult
        """
        self.logger.info(f"Starting benchmark: {name}")
        
        # Collections for metrics
        all_retrieved = []
        all_relevant = []
        all_predictions = []
        all_references = []
        latencies = []
        hallucination_results = []
        
        # Process each sample
        for i, sample in enumerate(samples):
            if verbose and i % 10 == 0:
                self.logger.info(f"Processing sample {i+1}/{len(samples)}")
            
            # Run RAG pipeline
            start_time = time.time()
            response = self.rag_pipeline.query(sample.query)
            latency = (time.time() - start_time) * 1000
            
            latencies.append(latency)
            
            # Collect retrieval results
            retrieved_ids = [c.source_id for c in response.citations]
            all_retrieved.append(retrieved_ids)
            all_relevant.append(sample.relevant_docs)
            
            # Collect generation results
            all_predictions.append(response.answer)
            all_references.append(sample.ground_truth)
            
            # Hallucination detection
            sources = [c.text_snippet for c in response.citations]
            hall_result = self.hallucination_detector.detect_ngram_overlap(
                response.answer, sources
            )
            hallucination_results.append(hall_result)
        
        # Calculate retrieval metrics
        retrieval_results = self.retrieval_metrics.evaluate_batch(
            all_retrieved, all_relevant
        )
        
        # Calculate generation metrics
        generation_results = self.generation_metrics.evaluate(
            all_predictions,
            all_references,
            include_bertscore=include_bertscore
        )
        
        # Calculate hallucination rate
        hallucination_rate = sum(
            1 for r in hallucination_results if r.is_hallucinated
        ) / len(hallucination_results) if hallucination_results else 0
        
        # Calculate latency statistics
        latency_stats = {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies))
        }
        
        result = BenchmarkResult(
            name=name,
            timestamp=datetime.now().isoformat(),
            retrieval_metrics=retrieval_results,
            generation_metrics=generation_results,
            hallucination_rate=hallucination_rate,
            latency_stats=latency_stats,
            config={
                "num_samples": len(samples),
                "model": getattr(self.rag_pipeline, 'model_name', 'unknown'),
                "include_bertscore": include_bertscore
            }
        )
        
        self.logger.info(f"Benchmark complete. Results:\n{result.summary()}")
        return result
    
    def compare_configs(
        self,
        configs: List[Dict],
        samples: List[EvaluationSample],
        metric_key: str = "ndcg@5"
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare multiple configurations.
        
        Args:
            configs: List of config dicts with 'name' and parameters
            samples: Evaluation samples
            metric_key: Primary metric for comparison
            
        Returns:
            Dict of results by config name
        """
        results = {}
        
        for cfg in configs:
            name = cfg.pop('name', f"config_{len(results)}")
            
            # Apply config to pipeline (implementation specific)
            # This is a placeholder - actual implementation depends on pipeline
            
            result = self.run(samples, name=name, verbose=False)
            results[name] = result
            
            self.logger.info(
                f"{name}: {metric_key} = "
                f"{result.retrieval_metrics.get(metric_key, MetricResult('N/A', 0)).value:.4f}"
            )
        
        return results
    
    def statistical_significance(
        self,
        results_a: List[float],
        results_b: List[float],
        alpha: float = 0.05
    ) -> Dict:
        """
        Test statistical significance between two result sets.
        
        Uses paired t-test for comparison.
        
        Args:
            results_a: Metric values for config A
            results_b: Metric values for config B
            alpha: Significance level
            
        Returns:
            Dict with test results
        """
        from scipy import stats
        
        t_stat, p_value = stats.ttest_rel(results_a, results_b)
        
        mean_diff = np.mean(results_a) - np.mean(results_b)
        ci_low, ci_high = stats.t.interval(
            1 - alpha,
            len(results_a) - 1,
            loc=mean_diff,
            scale=stats.sem(np.array(results_a) - np.array(results_b))
        )
        
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "mean_difference": float(mean_diff),
            "confidence_interval": (float(ci_low), float(ci_high)),
            "alpha": alpha
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmarking Test")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    parser.add_argument("--full", action="store_true", help="Run full benchmark")
    args = parser.parse_args()
    
    if args.test:
        print("Benchmarking Module Test\n" + "=" * 50)
        
        # Create mock evaluation samples
        samples = [
            EvaluationSample(
                query="What is machine learning?",
                ground_truth="Machine learning is a subset of AI that enables computers to learn from data.",
                relevant_docs=["doc1", "doc2"]
            ),
            EvaluationSample(
                query="Explain deep learning",
                ground_truth="Deep learning uses neural networks with multiple layers.",
                relevant_docs=["doc3", "doc4"]
            )
        ]
        
        print(f"Created {len(samples)} evaluation samples")
        print("\nSample 1:")
        print(f"  Query: {samples[0].query}")
        print(f"  Ground truth: {samples[0].ground_truth[:50]}...")
        print(f"  Relevant docs: {samples[0].relevant_docs}")
        
        print("\nNote: Full benchmark requires a configured RAG pipeline.")
