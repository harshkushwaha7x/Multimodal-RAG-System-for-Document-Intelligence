"""
Hallucination Detection Module.
Detect and measure hallucinations in RAG responses.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re
import numpy as np

from ..utils import get_logger, get_config, LoggerMixin

logger = get_logger(__name__)
config = get_config()


@dataclass
class HallucinationResult:
    """Result of hallucination detection."""
    
    is_hallucinated: bool
    confidence: float
    method: str
    details: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "is_hallucinated": self.is_hallucinated,
            "confidence": self.confidence,
            "method": self.method,
            "details": self.details
        }


class HallucinationDetector(LoggerMixin):
    """
    Detect hallucinations in generated answers.
    
    Methods:
    - Embedding similarity: Compare answer to sources
    - N-gram overlap: Check lexical coverage
    - NLI entailment: Verify logical consistency
    """
    
    def __init__(
        self,
        similarity_threshold: float = None,
        overlap_threshold: float = 0.5,
        entailment_threshold: float = 0.7
    ):
        """
        Initialize hallucination detector.
        
        Args:
            similarity_threshold: Min similarity for non-hallucination
            overlap_threshold: Min n-gram overlap ratio
            entailment_threshold: Min entailment probability
        """
        self.similarity_threshold = similarity_threshold or config.evaluation.hallucination_threshold
        self.overlap_threshold = overlap_threshold
        self.entailment_threshold = entailment_threshold
        
        self._embedder = None
        self._nli_model = None
    
    def _load_embedder(self):
        """Lazy load embedding model."""
        if self._embedder is None:
            from ..embeddings import CustomEmbedder
            self._embedder = CustomEmbedder(device="cpu")
    
    def _load_nli_model(self):
        """Lazy load NLI model."""
        if self._nli_model is None:
            try:
                from transformers import pipeline
                self._nli_model = pipeline(
                    "text-classification",
                    model=config.evaluation.entailment_model,
                    device=-1  # CPU
                )
                self.logger.info("NLI model loaded")
            except Exception as e:
                self.logger.warning(f"Failed to load NLI model: {e}")
                self._nli_model = "unavailable"
    
    def detect_embedding_similarity(
        self,
        answer: str,
        sources: List[str]
    ) -> HallucinationResult:
        """
        Detect hallucination using embedding similarity.
        
        Compares answer embedding to source embeddings.
        Low similarity indicates potential hallucination.
        
        Args:
            answer: Generated answer
            sources: Source documents
            
        Returns:
            HallucinationResult
        """
        self._load_embedder()
        
        if not sources:
            return HallucinationResult(
                is_hallucinated=True,
                confidence=1.0,
                method="embedding_similarity",
                details={"reason": "No sources provided"}
            )
        
        # Encode answer and sources
        answer_embedding = self._embedder.encode(answer)
        source_embeddings = self._embedder.encode(sources)
        
        # Calculate max cosine similarity
        similarities = self._embedder.similarity(
            answer_embedding.reshape(1, -1),
            source_embeddings
        )[0]
        
        max_similarity = float(np.max(similarities))
        avg_similarity = float(np.mean(similarities))
        
        is_hallucinated = max_similarity < self.similarity_threshold
        
        # Confidence: how far from threshold
        if is_hallucinated:
            confidence = 1 - (max_similarity / self.similarity_threshold)
        else:
            confidence = (max_similarity - self.similarity_threshold) / (1 - self.similarity_threshold)
        
        return HallucinationResult(
            is_hallucinated=is_hallucinated,
            confidence=min(max(confidence, 0), 1),
            method="embedding_similarity",
            details={
                "max_similarity": max_similarity,
                "avg_similarity": avg_similarity,
                "threshold": self.similarity_threshold,
                "num_sources": len(sources)
            }
        )
    
    def detect_ngram_overlap(
        self,
        answer: str,
        sources: List[str],
        n: int = 3
    ) -> HallucinationResult:
        """
        Detect hallucination using n-gram overlap.
        
        Checks what fraction of answer n-grams appear in sources.
        Low overlap indicates potential hallucination.
        
        Args:
            answer: Generated answer
            sources: Source documents
            n: N-gram size
            
        Returns:
            HallucinationResult
        """
        def get_ngrams(text: str, n: int) -> set:
            """Extract n-grams from text."""
            words = text.lower().split()
            return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))
        
        if not sources:
            return HallucinationResult(
                is_hallucinated=True,
                confidence=1.0,
                method="ngram_overlap",
                details={"reason": "No sources provided"}
            )
        
        # Get answer n-grams
        answer_ngrams = get_ngrams(answer, n)
        
        if not answer_ngrams:
            return HallucinationResult(
                is_hallucinated=False,
                confidence=0.5,
                method="ngram_overlap",
                details={"reason": "Answer too short for n-gram analysis"}
            )
        
        # Get source n-grams
        source_ngrams = set()
        for source in sources:
            source_ngrams.update(get_ngrams(source, n))
        
        # Calculate overlap
        overlap = answer_ngrams & source_ngrams
        overlap_ratio = len(overlap) / len(answer_ngrams)
        
        is_hallucinated = overlap_ratio < self.overlap_threshold
        
        return HallucinationResult(
            is_hallucinated=is_hallucinated,
            confidence=1 - overlap_ratio if is_hallucinated else overlap_ratio,
            method="ngram_overlap",
            details={
                "overlap_ratio": overlap_ratio,
                "answer_ngrams": len(answer_ngrams),
                "matching_ngrams": len(overlap),
                "threshold": self.overlap_threshold,
                "n": n
            }
        )
    
    def detect_nli_entailment(
        self,
        answer: str,
        sources: List[str],
        chunk_size: int = 500
    ) -> HallucinationResult:
        """
        Detect hallucination using NLI entailment.
        
        Checks if sources entail the answer.
        Low entailment indicates potential hallucination.
        
        Args:
            answer: Generated answer
            sources: Source documents
            chunk_size: Max chars per source chunk
            
        Returns:
            HallucinationResult
        """
        self._load_nli_model()
        
        if self._nli_model == "unavailable":
            return HallucinationResult(
                is_hallucinated=False,
                confidence=0.0,
                method="nli_entailment",
                details={"reason": "NLI model not available"}
            )
        
        if not sources:
            return HallucinationResult(
                is_hallucinated=True,
                confidence=1.0,
                method="nli_entailment",
                details={"reason": "No sources provided"}
            )
        
        # Break answer into sentences
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        entailment_scores = []
        
        for sentence in sentences:
            best_score = 0.0
            
            for source in sources:
                # Chunk source if too long
                for i in range(0, len(source), chunk_size):
                    chunk = source[i:i+chunk_size]
                    
                    # NLI format: premise (source) -> hypothesis (answer sentence)
                    try:
                        result = self._nli_model(
                            f"{chunk} [SEP] {sentence}",
                            truncation=True
                        )[0]
                        
                        # Extract entailment score
                        if result['label'].lower() == 'entailment':
                            score = result['score']
                        else:
                            score = 1 - result['score']
                        
                        best_score = max(best_score, score)
                    except Exception as e:
                        self.logger.debug(f"NLI inference error: {e}")
                        continue
            
            entailment_scores.append(best_score)
        
        if not entailment_scores:
            return HallucinationResult(
                is_hallucinated=False,
                confidence=0.0,
                method="nli_entailment",
                details={"reason": "Could not compute entailment"}
            )
        
        avg_entailment = np.mean(entailment_scores)
        is_hallucinated = avg_entailment < self.entailment_threshold
        
        return HallucinationResult(
            is_hallucinated=is_hallucinated,
            confidence=1 - avg_entailment if is_hallucinated else avg_entailment,
            method="nli_entailment",
            details={
                "avg_entailment": float(avg_entailment),
                "min_entailment": float(min(entailment_scores)),
                "max_entailment": float(max(entailment_scores)),
                "num_sentences": len(sentences),
                "threshold": self.entailment_threshold
            }
        )
    
    def detect(
        self,
        answer: str,
        sources: List[str],
        methods: Optional[List[str]] = None
    ) -> Dict[str, HallucinationResult]:
        """
        Run multiple hallucination detection methods.
        
        Args:
            answer: Generated answer
            sources: Source documents
            methods: Methods to use (default: all)
            
        Returns:
            Dict of results by method
        """
        available_methods = {
            "embedding": self.detect_embedding_similarity,
            "ngram": self.detect_ngram_overlap,
            "nli": self.detect_nli_entailment
        }
        
        methods = methods or list(available_methods.keys())
        
        results = {}
        for method in methods:
            if method in available_methods:
                try:
                    results[method] = available_methods[method](answer, sources)
                except Exception as e:
                    self.logger.error(f"{method} detection failed: {e}")
                    results[method] = HallucinationResult(
                        is_hallucinated=False,
                        confidence=0.0,
                        method=method,
                        details={"error": str(e)}
                    )
        
        return results
    
    def aggregate_results(
        self,
        results: Dict[str, HallucinationResult],
        weights: Optional[Dict[str, float]] = None
    ) -> HallucinationResult:
        """
        Aggregate multiple detection results.
        
        Args:
            results: Results from multiple methods
            weights: Optional weights per method
            
        Returns:
            Aggregated result
        """
        if not results:
            return HallucinationResult(
                is_hallucinated=False,
                confidence=0.0,
                method="aggregate",
                details={"reason": "No results to aggregate"}
            )
        
        # Default equal weights
        if weights is None:
            weights = {m: 1.0 for m in results}
        
        # Weighted average of hallucination confidence
        total_weight = sum(weights.get(m, 0) for m in results)
        
        if total_weight == 0:
            return HallucinationResult(
                is_hallucinated=False,
                confidence=0.0,
                method="aggregate"
            )
        
        weighted_conf = sum(
            r.confidence * weights.get(m, 0) * (1 if r.is_hallucinated else -1)
            for m, r in results.items()
        ) / total_weight
        
        # Majority vote for is_hallucinated
        hallucination_votes = sum(1 for r in results.values() if r.is_hallucinated)
        is_hallucinated = hallucination_votes > len(results) / 2
        
        return HallucinationResult(
            is_hallucinated=is_hallucinated,
            confidence=abs(weighted_conf),
            method="aggregate",
            details={
                "methods_used": list(results.keys()),
                "hallucination_votes": hallucination_votes,
                "total_methods": len(results)
            }
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hallucination Detection Test")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    args = parser.parse_args()
    
    if args.test:
        print("Hallucination Detection Test\n" + "=" * 50)
        
        detector = HallucinationDetector()
        
        # Grounded answer
        sources = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            "Deep learning uses neural networks with multiple layers."
        ]
        
        grounded_answer = "Machine learning allows computers to learn from data and is part of AI."
        hallucinated_answer = "Machine learning was invented by Albert Einstein in 1920."
        
        print("\nTesting grounded answer:")
        result = detector.detect_ngram_overlap(grounded_answer, sources)
        print(f"  N-gram: {result}")
        
        print("\nTesting hallucinated answer:")
        result = detector.detect_ngram_overlap(hallucinated_answer, sources)
        print(f"  N-gram: {result}")
