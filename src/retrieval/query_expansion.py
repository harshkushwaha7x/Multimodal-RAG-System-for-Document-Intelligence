"""
Query Expansion Module for improved retrieval.
Expands queries with synonyms and related terms.
"""

from typing import List, Optional, Dict
from ..utils import get_logger, LoggerMixin

logger = get_logger(__name__)


class QueryExpander(LoggerMixin):
    """
    Expands queries using multiple strategies:
    - LLM-based expansion
    - Synonym expansion
    - Keyword extraction
    """
    
    def __init__(
        self,
        use_llm: bool = True,
        use_synonyms: bool = True
    ):
        """
        Initialize query expander.
        
        Args:
            use_llm: Use LLM for query expansion
            use_synonyms: Use WordNet synonyms
        """
        self.use_llm = use_llm
        self.use_synonyms = use_synonyms
        self.wordnet = None
        
        if use_synonyms:
            self._load_wordnet()
    
    def _load_wordnet(self):
        """Load WordNet for synonym expansion."""
        try:
            import nltk
            try:
                from nltk.corpus import wordnet
                self.wordnet = wordnet
                self.logger.debug("WordNet loaded")
            except LookupError:
                nltk.download('wordnet', quiet=True)
                from nltk.corpus import wordnet
                self.wordnet = wordnet
        except ImportError:
            self.logger.warning("NLTK not available for synonyms")
    
    def expand(
        self,
        query: str,
        num_expansions: int = 3
    ) -> List[str]:
        """
        Expand a query into multiple variations.
        
        Args:
            query: Original query
            num_expansions: Max number of expanded queries
            
        Returns:
            List of expanded queries (including original)
        """
        expanded = [query]
        
        # Add synonym-based expansions
        if self.use_synonyms and self.wordnet:
            synonym_expansion = self._expand_with_synonyms(query)
            if synonym_expansion and synonym_expansion != query:
                expanded.append(synonym_expansion)
        
        # Add keyword extraction
        keywords = self._extract_keywords(query)
        if keywords:
            expanded.append(" ".join(keywords))
        
        # Add LLM-based expansion via Ollama if available
        if self.use_llm:
            llm_expansions = self._expand_with_llm(query)
            expanded.extend(llm_expansions)
        
        # Remove duplicates and limit
        seen = set()
        unique = []
        for q in expanded:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique.append(q)
        
        return unique[:num_expansions + 1]
    
    def _expand_with_synonyms(self, query: str) -> str:
        """Expand query using WordNet synonyms."""
        if not self.wordnet:
            return query
        
        words = query.lower().split()
        expanded_words = []
        
        for word in words:
            synsets = self.wordnet.synsets(word)
            if synsets:
                # Get first synonym that's different
                for syn in synsets[0].lemmas()[:2]:
                    synonym = syn.name().replace('_', ' ')
                    if synonym.lower() != word.lower():
                        expanded_words.append(synonym)
                        break
                else:
                    expanded_words.append(word)
            else:
                expanded_words.append(word)
        
        return " ".join(expanded_words)
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract key terms from query."""
        # Simple keyword extraction - remove stopwords
        stopwords = {
            'what', 'how', 'why', 'when', 'where', 'which', 'who',
            'is', 'are', 'was', 'were', 'the', 'a', 'an', 'in', 'on',
            'at', 'to', 'for', 'of', 'and', 'or', 'but', 'with', 'by'
        }
        
        words = query.lower().split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords
    
    def _expand_with_llm(self, query: str) -> List[str]:
        """Expand query using Ollama LLM."""
        try:
            import requests
            
            # Check if Ollama is running
            try:
                resp = requests.get("http://localhost:11434/api/tags", timeout=1)
                if resp.status_code != 200:
                    return []
            except:
                return []
            
            # Generate expansion via Ollama
            prompt = f"""Rewrite this search query in a different way to find the same information:
Query: {query}
Rewritten query (just output the new query, nothing else):"""
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2:1.5b",  # Use small model for speed
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 50}
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                # Clean up response
                result = result.split('\n')[0].strip()
                if result and len(result) > 5 and len(result) < 200:
                    return [result]
            
            return []
            
        except Exception as e:
            self.logger.debug(f"LLM expansion failed: {e}")
            return []


class HyDEExpander(LoggerMixin):
    """
    Hypothetical Document Embedding (HyDE) expander.
    Generates a hypothetical answer to use for retrieval.
    """
    
    def __init__(self):
        """Initialize HyDE expander."""
        pass
    
    def generate_hypothetical(self, query: str) -> Optional[str]:
        """
        Generate a hypothetical document/answer for the query.
        
        Args:
            query: User query
            
        Returns:
            Hypothetical document text
        """
        try:
            import requests
            
            prompt = f"""Write a short paragraph that directly answers this question:
Question: {query}
Answer:"""
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2:1.5b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 100}
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                if result and len(result) > 20:
                    return result
            
            return None
            
        except Exception as e:
            self.logger.debug(f"HyDE generation failed: {e}")
            return None


class ContextualCompressor(LoggerMixin):
    """
    Compresses retrieved contexts to remove irrelevant portions.
    """
    
    def __init__(self, max_tokens: int = 300):
        """
        Initialize contextual compressor.
        
        Args:
            max_tokens: Maximum tokens per compressed context
        """
        self.max_tokens = max_tokens
    
    def compress(
        self,
        query: str,
        contexts: List[str]
    ) -> List[str]:
        """
        Compress contexts to relevant portions only.
        
        Args:
            query: User query
            contexts: List of retrieved contexts
            
        Returns:
            Compressed contexts
        """
        compressed = []
        
        for context in contexts:
            # Split into sentences
            sentences = self._split_sentences(context)
            
            # Score each sentence for relevance
            query_words = set(query.lower().split())
            scored = []
            
            for sent in sentences:
                sent_words = set(sent.lower().split())
                overlap = len(query_words & sent_words) / max(len(query_words), 1)
                scored.append((sent, overlap))
            
            # Keep top sentences up to token limit
            scored.sort(key=lambda x: x[1], reverse=True)
            
            selected = []
            total_tokens = 0
            
            for sent, score in scored:
                sent_tokens = len(sent.split())
                if total_tokens + sent_tokens <= self.max_tokens:
                    selected.append(sent)
                    total_tokens += sent_tokens
            
            if selected:
                # Reorder by original position
                ordered = [s for s in sentences if s in selected]
                compressed.append(" ".join(ordered))
        
        return compressed
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


# Convenience functions
def get_query_expander() -> QueryExpander:
    """Get default query expander."""
    return QueryExpander(use_llm=True, use_synonyms=True)


def get_compressor() -> ContextualCompressor:
    """Get default contextual compressor."""
    return ContextualCompressor(max_tokens=300)
