"""
Vector Database Module.
Supports PostgreSQL+pgvector and FAISS for vector storage and retrieval.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from ..utils import get_logger, get_config, LoggerMixin

logger = get_logger(__name__)
config = get_config()


@dataclass
class Document:
    """Document with text and metadata."""
    
    id: str
    text: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata
        }


@dataclass
class SearchResult:
    """Search result with score."""
    
    document: Document
    score: float
    rank: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.document.id,
            "text": self.document.text,
            "score": self.score,
            "rank": self.rank,
            "metadata": self.document.metadata
        }


class VectorStore(ABC, LoggerMixin):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[np.ndarray] = None
    ) -> List[str]:
        """Add documents to the store."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[SearchResult]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def delete(self, document_ids: List[str]) -> int:
        """Delete documents by ID."""
        pass
    
    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        pass
    
    @property
    @abstractmethod
    def count(self) -> int:
        """Return number of documents in store."""
        pass


class PostgresVectorStore(VectorStore):
    """
    PostgreSQL + pgvector vector store.
    
    Features:
    - ACID compliance
    - SQL filtering
    - IVFFlat/HNSW indexing
    - Full-text search support
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        table_name: str = "document_embeddings",
        embedding_dim: int = None,
        index_type: str = None
    ):
        """
        Initialize PostgreSQL vector store.
        
        Args:
            connection_string: PostgreSQL connection string
            table_name: Name of the embeddings table
            embedding_dim: Dimension of embeddings
            index_type: Index type ("ivfflat" or "hnsw")
        """
        self.table_name = table_name
        self.embedding_dim = embedding_dim or config.embedding.embedding_dim
        self.index_type = index_type or config.database.index_type
        
        # Build connection string
        if connection_string:
            self.connection_string = connection_string
        else:
            db = config.database
            self.connection_string = (
                f"postgresql://{db.pg_user}:{db.pg_password}@"
                f"{db.pg_host}:{db.pg_port}/{db.pg_database}"
            )
        
        self.conn = None
        self._initialized = False
    
    def _connect(self):
        """Connect to database."""
        if self.conn is not None:
            return
        
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
            
            self.conn = psycopg2.connect(self.connection_string)
            register_vector(self.conn)
            self.logger.info("Connected to PostgreSQL")
            
        except ImportError:
            self.logger.error("psycopg2 or pgvector not installed")
            raise
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            raise
    
    def initialize(self):
        """Create table and indexes."""
        self._connect()
        
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    embedding vector({self.embedding_dim}),
                    metadata JSONB DEFAULT '{{}}',
                    full_text_search tsvector GENERATED ALWAYS AS (
                        to_tsvector('english', chunk_text)
                    ) STORED,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create vector index
            index_name = f"{self.table_name}_embedding_idx"
            if self.index_type == "ivfflat":
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {self.table_name}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {config.database.num_lists})
                """)
            elif self.index_type == "hnsw":
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {self.table_name}
                    USING hnsw (embedding vector_cosine_ops)
                """)
            
            # Create GIN index for full-text search
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_fts_idx
                ON {self.table_name}
                USING gin(full_text_search)
            """)
            
            self.conn.commit()
            
        self._initialized = True
        self.logger.info(f"Initialized table {self.table_name}")
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[np.ndarray] = None
    ) -> List[str]:
        """Add documents with embeddings."""
        self._connect()
        
        if not self._initialized:
            self.initialize()
        
        ids = []
        with self.conn.cursor() as cur:
            for i, doc in enumerate(documents):
                embedding = embeddings[i] if embeddings is not None else doc.embedding
                
                if embedding is None:
                    self.logger.warning(f"No embedding for document {doc.id}")
                    continue
                
                cur.execute(f"""
                    INSERT INTO {self.table_name} (id, chunk_text, embedding, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        chunk_text = EXCLUDED.chunk_text,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                """, (
                    doc.id,
                    doc.text,
                    embedding.tolist(),
                    json.dumps(doc.metadata)
                ))
                ids.append(doc.id)
            
            self.conn.commit()
        
        self.logger.info(f"Added {len(ids)} documents")
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        self._connect()
        
        with self.conn.cursor() as cur:
            # Build query
            query = f"""
                SELECT id, chunk_text, metadata,
                       1 - (embedding <=> %s) as similarity
                FROM {self.table_name}
            """
            
            params = [query_embedding.tolist()]
            
            # Add metadata filter
            if filter_metadata:
                conditions = []
                for key, value in filter_metadata.items():
                    conditions.append(f"metadata->>{key} = %s")
                    params.append(json.dumps(value))
                query += " WHERE " + " AND ".join(conditions)
            
            query += f" ORDER BY embedding <=> %s LIMIT {top_k}"
            params.append(query_embedding.tolist())
            
            cur.execute(query, params)
            rows = cur.fetchall()
        
        results = []
        for rank, (id, text, metadata, score) in enumerate(rows):
            doc = Document(
                id=id,
                text=text,
                metadata=metadata if isinstance(metadata, dict) else json.loads(metadata)
            )
            results.append(SearchResult(
                document=doc,
                score=float(score),
                rank=rank
            ))
        
        return results
    
    def full_text_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """Perform full-text search using PostgreSQL FTS."""
        self._connect()
        
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, chunk_text, metadata,
                       ts_rank(full_text_search, plainto_tsquery('english', %s)) as score
                FROM {self.table_name}
                WHERE full_text_search @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT {top_k}
            """, (query, query))
            rows = cur.fetchall()
        
        results = []
        for rank, (id, text, metadata, score) in enumerate(rows):
            doc = Document(
                id=id,
                text=text,
                metadata=metadata if isinstance(metadata, dict) else json.loads(metadata)
            )
            results.append(SearchResult(
                document=doc,
                score=float(score),
                rank=rank
            ))
        
        return results
    
    def delete(self, document_ids: List[str]) -> int:
        """Delete documents by ID."""
        self._connect()
        
        with self.conn.cursor() as cur:
            cur.execute(f"""
                DELETE FROM {self.table_name}
                WHERE id = ANY(%s)
            """, (document_ids,))
            deleted = cur.rowcount
            self.conn.commit()
        
        self.logger.info(f"Deleted {deleted} documents")
        return deleted
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        self._connect()
        
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, chunk_text, metadata
                FROM {self.table_name}
                WHERE id = %s
            """, (document_id,))
            row = cur.fetchone()
        
        if row:
            return Document(
                id=row[0],
                text=row[1],
                metadata=row[2] if isinstance(row[2], dict) else json.loads(row[2])
            )
        return None
    
    @property
    def count(self) -> int:
        """Return number of documents."""
        self._connect()
        
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            return cur.fetchone()[0]
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.logger.info("Closed PostgreSQL connection")


class FAISSVectorStore(VectorStore):
    """
    FAISS vector store for fast similarity search.
    
    Features:
    - Fast approximate nearest neighbor search
    - GPU acceleration support
    - Multiple index types (Flat, IVF, HNSW)
    """
    
    def __init__(
        self,
        embedding_dim: int = None,
        index_type: str = None,
        nlist: int = None,
        nprobe: int = None
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Index type ("flat", "ivf", "hnsw")
            nlist: Number of clusters for IVF
            nprobe: Number of clusters to search
        """
        self.embedding_dim = embedding_dim or config.embedding.embedding_dim
        self.index_type = index_type or config.database.faiss_index_type
        self.nlist = nlist or config.database.faiss_nlist
        self.nprobe = nprobe or config.database.faiss_nprobe
        
        self.index = None
        self.documents: Dict[int, Document] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.current_idx = 0
        
        self._init_index()
    
    def _init_index(self):
        """Initialize FAISS index."""
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            self.logger.error("faiss not installed")
            raise ImportError("Install faiss: pip install faiss-cpu")
        
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product
        
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                self.nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            self._needs_training = True
        
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.index.hnsw.efConstruction = 200
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        self.logger.info(f"Initialized FAISS {self.index_type} index")
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[np.ndarray] = None
    ) -> List[str]:
        """Add documents with embeddings."""
        if embeddings is None:
            embeddings = np.vstack([doc.embedding for doc in documents])
        
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings.astype('float32')
        
        # Train IVF index if needed
        if hasattr(self, '_needs_training') and self._needs_training:
            if embeddings.shape[0] >= self.nlist:
                self.index.train(embeddings)
                self._needs_training = False
            else:
                self.logger.warning(
                    f"Not enough vectors ({embeddings.shape[0]}) to train IVF index "
                    f"(need {self.nlist}). Using flat index."
                )
                self.index = self.faiss.IndexFlatIP(self.embedding_dim)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store document mapping
        ids = []
        for doc in documents:
            self.documents[self.current_idx] = doc
            self.id_to_idx[doc.id] = self.current_idx
            self.idx_to_id[self.current_idx] = doc.id
            ids.append(doc.id)
            self.current_idx += 1
        
        self.logger.info(f"Added {len(ids)} documents to FAISS")
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[SearchResult]:
        """Search for similar documents."""
        # Normalize query
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Set search parameters
        if self.index_type == "ivf" and hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            doc = self.documents.get(idx)
            if doc:
                results.append(SearchResult(
                    document=doc,
                    score=float(score),
                    rank=rank
                ))
        
        return results
    
    def delete(self, document_ids: List[str]) -> int:
        """Delete documents by ID (not well supported in FAISS)."""
        deleted = 0
        for doc_id in document_ids:
            if doc_id in self.id_to_idx:
                idx = self.id_to_idx[doc_id]
                del self.documents[idx]
                del self.id_to_idx[doc_id]
                del self.idx_to_id[idx]
                deleted += 1
        
        self.logger.warning(
            f"Marked {deleted} documents as deleted. "
            "Note: FAISS doesn't support true deletion. Rebuild index for cleanup."
        )
        return deleted
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        idx = self.id_to_idx.get(document_id)
        if idx is not None:
            return self.documents.get(idx)
        return None
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents in the store."""
        return list(self.documents.values())
    
    @property
    def count(self) -> int:
        """Return number of documents."""
        return len(self.documents)
    
    def save(self, path: Union[str, Path]):
        """Save index and documents."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        self.faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save documents and mappings
        import pickle
        with open(path / "documents.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'id_to_idx': self.id_to_idx,
                'idx_to_id': self.idx_to_id,
                'current_idx': self.current_idx
            }, f)
        
        self.logger.info(f"Saved FAISS index to {path}")
    
    def load(self, path: Union[str, Path]):
        """Load index and documents."""
        path = Path(path)
        
        # Load FAISS index
        self.index = self.faiss.read_index(str(path / "index.faiss"))
        
        # Load documents and mappings
        import pickle
        with open(path / "documents.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.id_to_idx = data['id_to_idx']
            self.idx_to_id = data['idx_to_id']
            self.current_idx = data['current_idx']
        
        self.logger.info(f"Loaded FAISS index from {path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vector DB Test")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    parser.add_argument("--init", action="store_true", help="Initialize PostgreSQL")
    args = parser.parse_args()
    
    if args.test:
        print("Vector Store Test (FAISS)\n" + "=" * 50)
        
        # Create sample documents
        np.random.seed(42)
        docs = [
            Document(id=f"doc_{i}", text=f"Sample document {i}", 
                    embedding=np.random.randn(768))
            for i in range(100)
        ]
        
        # Initialize FAISS store
        store = FAISSVectorStore(embedding_dim=768, index_type="flat")
        store.add_documents(docs)
        
        print(f"Documents in store: {store.count}")
        
        # Search
        query = np.random.randn(768)
        results = store.search(query, top_k=5)
        
        print(f"\nTop 5 results:")
        for r in results:
            print(f"  {r.document.id}: score={r.score:.4f}")
    
    if args.init:
        print("Initializing PostgreSQL Vector Store...")
        store = PostgresVectorStore()
        store.initialize()
        print("Done!")
