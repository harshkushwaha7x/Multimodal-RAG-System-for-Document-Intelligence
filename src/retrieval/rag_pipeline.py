"""
RAG Pipeline Module.
End-to-end Retrieval-Augmented Generation with LangChain.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import time

from .hybrid_search import HybridRetriever, SearchResult
from ..utils import get_logger, get_config, LoggerMixin

logger = get_logger(__name__)
config = get_config()


@dataclass
class Citation:
    """Source citation for generated answer."""
    
    source_id: str
    source_file: str
    page: Optional[int] = None
    relevance_score: float = 0.0
    text_snippet: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "source_id": self.source_id,
            "source_file": self.source_file,
            "page": self.page,
            "relevance_score": self.relevance_score,
            "text_snippet": self.text_snippet[:200]
        }


@dataclass
class RAGResponse:
    """Complete RAG response with answer and metadata."""
    
    query: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata
        }
    
    def format_answer(self, include_sources: bool = True) -> str:
        """Format answer with sources."""
        formatted = self.answer
        
        if include_sources and self.citations:
            formatted += "\n\n**Sources:**\n"
            for i, citation in enumerate(self.citations, 1):
                source_info = citation.source_file
                if citation.page:
                    source_info += f", Page {citation.page}"
                formatted += f"{i}. {source_info}\n"
        
        return formatted


class PromptTemplates:
    """Prompt templates for RAG."""
    
    SYSTEM_PROMPT = """You are a document intelligence assistant. Your role is to answer questions based ONLY on the provided context. Follow these rules strictly:

1. Answer ONLY from the provided context
2. If the context doesn't contain enough information, say "I don't have enough information to answer this question."
3. Cite sources using [Source: filename, Page: X] format
4. Be concise and accurate
5. Do not make up information not present in the context"""
    
    CONTEXT_TEMPLATE = """Context:
{context}

---
Based on the above context, answer the following question:

Question: {question}

Answer:"""
    
    CITATION_TEMPLATE = """Answer the question based on the context below. Include citations in your answer using [Source X] format.

Context:
{context}

Question: {question}

Instructions:
- Only use information from the context
- Include [Source X] citations for each fact
- If unsure, say so

Answer:"""


class RAGPipeline(LoggerMixin):
    """
    End-to-end RAG pipeline.
    
    Combines retrieval, context assembly, and generation
    with support for multiple LLM backends.
    """
    
    # Supported models (open models that don't require authentication)
    SUPPORTED_MODELS = {
        "phi3": "microsoft/Phi-3-mini-4k-instruct",  # Best for 6GB GPU (~4GB VRAM)
        "gemma2": "google/gemma-2-2b-it",  # Lightweight option (~3GB VRAM)
        "qwen2": "Qwen/Qwen2-1.5B-Instruct",  # Very lightweight (~2GB VRAM)
        "flan-t5": "google/flan-t5-large",  # Seq2seq, works great (~2GB VRAM)
        "flan-t5-base": "google/flan-t5-base",  # Smaller seq2seq (~1GB VRAM)
        "opt": "facebook/opt-1.3b"  # Open model (~3GB VRAM)
    }
    
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        model_name: Optional[str] = None,
        max_context_tokens: int = None,
        temperature: float = None,
        use_langchain: bool = True
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: Hybrid retriever for document search
            model_name: LLM model name or shortcut
            max_context_tokens: Maximum tokens for context
            temperature: Generation temperature
            use_langchain: Whether to use LangChain
        """
        self.retriever = retriever
        self.model_name = self._resolve_model_name(model_name)
        self.max_context_tokens = max_context_tokens or config.llm.context_window
        self.temperature = temperature or config.llm.temperature
        self.use_langchain = use_langchain
        
        self.llm = None
        self.prompt_template = PromptTemplates.CONTEXT_TEMPLATE
    
    def _resolve_model_name(self, model_name: Optional[str]) -> str:
        """Resolve model shortcut to full name."""
        if model_name is None:
            # Use flan-t5-base as safe default (open model, works everywhere)
            return "google/flan-t5-base"
        
        # Check if it's a shortcut name
        if model_name.lower() in self.SUPPORTED_MODELS:
            return self.SUPPORTED_MODELS[model_name.lower()]
        
        # Check if it looks like a valid HuggingFace model path (contains /)
        if "/" in model_name:
            return model_name
        
        # Unknown shortcut - fallback to flan-t5-base
        self.logger.warning(f"Unknown model '{model_name}', falling back to google/flan-t5-base")
        return "google/flan-t5-base"
    
    def _load_llm(self):
        """Load the LLM - tries Ollama first (GPU-only), then HuggingFace."""
        if self.llm is not None:
            return
        
        self.logger.info(f"Loading LLM: {self.model_name}")
        
        # Try Ollama first (GPU-only, no CPU RAM needed)
        if self._try_load_ollama():
            return
        
        # Fallback to LangChain/HuggingFace
        if self.use_langchain:
            self._load_langchain_llm()
        else:
            self._load_hf_llm()
    
    def _try_load_ollama(self) -> bool:
        """Try to load LLM via Ollama (GPU-only). Returns True if successful."""
        try:
            import requests
            
            # Check if Ollama is running
            try:
                resp = requests.get("http://localhost:11434/api/tags", timeout=2)
                if resp.status_code != 200:
                    self.logger.info("Ollama not running, trying HuggingFace...")
                    return False
            except:
                self.logger.info("Ollama not available, trying HuggingFace...")
                return False
            
            # Map model names to Ollama models
            ollama_models = {
                "Qwen/Qwen2-1.5B-Instruct": "qwen2:1.5b",
                "microsoft/Phi-3-mini-4k-instruct": "phi3",
                "google/flan-t5-large": "flan-t5",
                "meta-llama/Llama-3.1-8B-Instruct": "llama3.1:8b",
                "mistralai/Mistral-7B-Instruct-v0.2": "mistral",
                "google/gemma-2-2b-it": "gemma2:9b",
            }
            
            ollama_model = ollama_models.get(self.model_name, "qwen2:1.5b")
            
            self.logger.info(f"Using Ollama with model: {ollama_model}")
            self.ollama_model = ollama_model
            self.use_ollama = True
            self.llm = "ollama"  # Marker that we're using Ollama
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Ollama not available: {e}")
            return False
    
    def _load_langchain_llm(self):
        """Load LLM using LangChain."""
        try:
            from langchain_huggingface import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, BitsAndBytesConfig
            import torch
            
            # Determine if model needs trust_remote_code
            needs_trust = any(name in self.model_name.lower() for name in ['phi', 'qwen', 'gemma', 'mistral'])
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=needs_trust
            )
            
            # Add pad token if missing (fixes generation issues)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Check if model is seq2seq (T5, BART, etc.) or causal LM
            is_seq2seq = any(name in self.model_name.lower() for name in ['t5', 'bart', 'pegasus', 'marian'])
            
            # Use 4-bit quantization for large models on limited VRAM
            use_4bit = device == "cuda" and any(name in self.model_name.lower() for name in ['llama', 'mistral', 'mixtral', 'phi-3', 'phi3'])
            
            if use_4bit:
                self.logger.info("Using 4-bit quantization for large model with CPU offload")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                # Limit GPU memory to leave room for embeddings
                max_memory = {0: "4GiB", "cpu": "12GiB"}
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    max_memory=max_memory,
                    low_cpu_mem_usage=True,
                    offload_folder="offload",
                    trust_remote_code=needs_trust
                )
            elif is_seq2seq:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="cuda:0" if device == "cuda" else None,
                    low_cpu_mem_usage=True,
                    max_memory={0: "5GiB"},
                    trust_remote_code=needs_trust
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="cuda:0" if device == "cuda" else None,
                    low_cpu_mem_usage=True,
                    max_memory={0: "5GiB"},
                    trust_remote_code=needs_trust
                )
            
            # Create pipeline with stable generation settings
            # Use do_sample=False (greedy) to avoid NaN issues with float16
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=config.llm.max_new_tokens,
                do_sample=False,  # Greedy decoding - more stable
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            self.tokenizer = tokenizer
            
            self.logger.info(f"LLM loaded on {device}")
            
        except ImportError as e:
            self.logger.error(f"Failed to load LangChain LLM: {e}")
            raise
    
    def _load_hf_llm(self):
        """Load LLM directly from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            
            self.device = device
            self.logger.info(f"HF model loaded on {device}")
            
        except ImportError as e:
            self.logger.error(f"Failed to load HF model: {e}")
            raise
    
    def _assemble_context(
        self,
        results: List[SearchResult],
        max_tokens: Optional[int] = None
    ) -> tuple:
        """
        Assemble context from search results.
        
        Args:
            results: Search results
            max_tokens: Maximum tokens for context
            
        Returns:
            Tuple of (context_text, citations)
        """
        max_tokens = max_tokens or self.max_context_tokens
        
        context_parts = []
        citations = []
        current_tokens = 0
        
        for i, result in enumerate(results):
            # Estimate tokens (rough: 1 token ≈ 4 chars)
            text = result.document.text
            estimated_tokens = len(text) // 4
            
            if current_tokens + estimated_tokens > max_tokens * 0.7:  # Leave room for prompt
                break
            
            # Add to context
            source_label = f"[Source {i+1}]"
            context_parts.append(f"{source_label}\n{text}")
            
            # Create citation
            metadata = result.document.metadata
            citations.append(Citation(
                source_id=result.document.id,
                source_file=metadata.get("source_file", f"Document {i+1}"),
                page=metadata.get("page_number"),
                relevance_score=result.score,
                text_snippet=text[:200]
            ))
            
            current_tokens += estimated_tokens
        
        context = "\n\n".join(context_parts)
        return context, citations
    
    def _generate_ollama(self, prompt: str) -> str:
        """Generate response using Ollama (GPU-only)."""
        import requests
        import json
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": config.llm.max_new_tokens
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                self.logger.error(f"Ollama error: {response.status_code}")
                return f"Error: Ollama returned {response.status_code}"
                
        except Exception as e:
            self.logger.error(f"Ollama generation failed: {e}")
            return f"Error: {str(e)}"
    
    def _generate(self, prompt: str) -> str:
        """Generate response from LLM."""
        self._load_llm()
        
        # Use Ollama if available (GPU-only)
        if getattr(self, 'use_ollama', False):
            return self._generate_ollama(prompt)
        
        if self.use_langchain and self.llm:
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        else:
            # Direct HF generation
            import torch
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_tokens
            )
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config.llm.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=config.llm.do_sample,
                    top_p=config.llm.top_p,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            return response.strip()
    
    def query(
        self,
        question: str,
        top_k: int = None,
        include_citations: bool = True
    ) -> RAGResponse:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            include_citations: Whether to include citations
            
        Returns:
            RAGResponse with answer and metadata
        """
        start_time = time.time()
        top_k = top_k or config.retrieval.final_k
        
        self.logger.info(f"Processing query: {question[:50]}...")
        
        # Retrieve relevant documents
        if self.retriever:
            results = self.retriever.retrieve(question, top_k=top_k)
        else:
            self.logger.warning("No retriever configured, using empty context")
            results = []
        
        # Assemble context
        context, citations = self._assemble_context(results)
        
        if not context:
            return RAGResponse(
                query=question,
                answer="I don't have any relevant documents to answer this question.",
                citations=[],
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000
            )
        
        # Build prompt
        if include_citations:
            prompt = PromptTemplates.CITATION_TEMPLATE.format(
                context=context,
                question=question
            )
        else:
            prompt = PromptTemplates.CONTEXT_TEMPLATE.format(
                context=context,
                question=question
            )
        
        # Generate answer
        answer = self._generate(prompt)
        
        # Calculate confidence based on retrieval scores
        avg_score = sum(c.relevance_score for c in citations) / len(citations) if citations else 0
        confidence = min(avg_score, 1.0)
        
        latency_ms = (time.time() - start_time) * 1000
        
        self.logger.info(f"Generated answer in {latency_ms:.0f}ms")
        
        return RAGResponse(
            query=question,
            answer=answer,
            citations=citations if include_citations else [],
            confidence=confidence,
            latency_ms=latency_ms,
            metadata={
                "model": self.model_name,
                "num_sources": len(citations),
                "context_length": len(context)
            }
        )
    
    def query_simple(self, question: str) -> str:
        """
        Simple query interface returning just the answer.
        
        Args:
            question: User question
            
        Returns:
            Answer string
        """
        response = self.query(question, include_citations=False)
        return response.answer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Pipeline Test")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    args = parser.parse_args()
    
    if args.test or args.demo:
        print("RAG Pipeline Test\n" + "=" * 50)
        print(f"Supported models:")
        for shortcut, full_name in RAGPipeline.SUPPORTED_MODELS.items():
            print(f"  {shortcut}: {full_name}")
        
        print("\nNote: Full demo requires:")
        print("  1. Vector store with indexed documents")
        print("  2. LLM model downloaded")
        print("  3. Sufficient GPU memory (for larger models)")
        
        # Create pipeline (without retriever for testing)
        pipeline = RAGPipeline(model_name="flan-t5")
        print(f"\nPipeline configured with: {pipeline.model_name}")
