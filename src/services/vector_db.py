import sqlite3
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import io

logger = logging.getLogger(__name__)


def adapt_array(arr):
    """Convert numpy array to bytes for SQLite storage."""
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(blob):
    """Convert bytes back to numpy array."""
    out = io.BytesIO(blob)
    out.seek(0)
    return np.load(out)


# Register numpy array adapters for SQLite
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)


class VectorDB:
    """
    SQLite-based vector database for semantic search.
    Stores document chunks with their embeddings.
    """
    
    def __init__(self, db_path: str = "documents.db"):
        """
        Initialize vector database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
        logger.info(f"VectorDB initialized: {db_path}")
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.executescript("""
                -- Documents table for metadata
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    subject TEXT,
                    topic TEXT,
                    difficulty TEXT,
                    language TEXT DEFAULT 'kk',
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunk_count INTEGER DEFAULT 0
                );
                
                -- Chunks table with embeddings
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding array NOT NULL,
                    token_count INTEGER,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                
                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
                    ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_documents_subject 
                    ON documents(subject);
                CREATE INDEX IF NOT EXISTS idx_documents_topic 
                    ON documents(topic);
                CREATE INDEX IF NOT EXISTS idx_documents_source 
                    ON documents(source_file);
                
                -- Metadata for database versioning
                CREATE TABLE IF NOT EXISTS db_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                
                INSERT OR IGNORE INTO db_metadata (key, value) 
                VALUES ('version', '1.0'), ('created_at', datetime('now'));
            """)
            logger.info("Database schema initialized")
    
    def add_document(self, doc_id: str, source_file: str, 
                    metadata: Dict = None) -> bool:
        """
        Add document metadata to database.
        
        Args:
            doc_id: Unique document identifier
            source_file: Source filename
            metadata: Additional metadata (subject, topic, etc.)
            
        Returns:
            True if successful
        """
        try:
            metadata = metadata or {}
            
            with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO documents 
                    (id, source_file, subject, topic, difficulty, language, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_id,
                    source_file,
                    metadata.get('subject'),
                    metadata.get('topic'),
                    metadata.get('difficulty'),
                    metadata.get('language', 'kk'),
                    json.dumps(metadata)
                ))
            
            logger.debug(f"Document '{doc_id}' added to database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document '{doc_id}': {e}")
            return False
    
    def add_chunks(self, document_id: str, chunks: List[Dict], 
                   embeddings: np.ndarray) -> int:
        """
        Add chunks with embeddings to database.
        
        Args:
            document_id: Parent document ID
            chunks: List of chunk dicts (content, chunk_index, metadata)
            embeddings: Numpy array of embeddings, shape (n_chunks, embedding_dim)
            
        Returns:
            Number of chunks added
        """
        try:
            if len(chunks) != len(embeddings):
                raise ValueError(f"Chunks count ({len(chunks)}) != embeddings count ({len(embeddings)})")
            
            with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
                # Add chunks
                for chunk, embedding in zip(chunks, embeddings):
                    conn.execute("""
                        INSERT INTO chunks 
                        (document_id, chunk_index, content, embedding, token_count, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        document_id,
                        chunk['chunk_index'],
                        chunk['content'],
                        embedding,
                        chunk.get('token_count'),
                        json.dumps(chunk.get('metadata', {}))
                    ))
                
                # Update chunk count in documents table
                conn.execute("""
                    UPDATE documents 
                    SET chunk_count = ? 
                    WHERE id = ?
                """, (len(chunks), document_id))
            
            logger.info(f"Added {len(chunks)} chunks for document '{document_id}'")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error adding chunks for '{document_id}': {e}")
            return 0
    
    def search(self, query_embedding: np.ndarray, 
               top_k: int = 5,
               filters: Dict = None,
               min_similarity: float = 0.0) -> List[Dict]:
        """
        Search for most similar chunks using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            filters: Optional filters (subject, topic, difficulty, document_id)
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of dicts with chunk content, similarity score, and metadata
        """
        try:
            filters = filters or {}
            
            with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
                # Build WHERE clause for filters
                where_clauses = []
                params = []
                
                if 'subject' in filters:
                    where_clauses.append("d.subject = ?")
                    params.append(filters['subject'])
                
                if 'topic' in filters:
                    where_clauses.append("d.topic = ?")
                    params.append(filters['topic'])
                
                if 'difficulty' in filters:
                    where_clauses.append("d.difficulty = ?")
                    params.append(filters['difficulty'])
                
                if 'document_id' in filters:
                    where_clauses.append("c.document_id = ?")
                    params.append(filters['document_id'])
                
                where_sql = " AND " + " AND ".join(where_clauses) if where_clauses else ""
                
                # Fetch all chunks (with optional filters)
                query = f"""
                    SELECT 
                        c.id,
                        c.document_id,
                        c.chunk_index,
                        c.content,
                        c.embedding,
                        c.token_count,
                        c.metadata,
                        d.subject,
                        d.topic,
                        d.difficulty,
                        d.source_file
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE 1=1 {where_sql}
                """
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
            
            if not rows:
                logger.warning("No chunks found in database")
                return []
            
            # 1. Prepare matrices
            # Convert list of embeddings from DB rows into a single 2D matrix (N chunks x Dim vector)
            embeddings_matrix = np.vstack([row[4] for row in rows])
            
            # 2. Mathematical optimization (Vectorized Cosine Similarity)
            # Cosine similarity = (A · B) / (||A|| * ||B||)
            
            # Normalize query vector (L2 norm)
            query_unit = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
            
            # Normalize all vectors in chunk matrix at once
            matrix_norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True) + 1e-10
            matrix_unit = embeddings_matrix / matrix_norms
            
            # Compute dot product - this will be the cosine similarity for normalized vectors
            # Result is 1D array of similarities for all chunks
            similarities = np.dot(matrix_unit, query_unit)
            
            # 3. Filter and assemble results
            results = []
            for i, row in enumerate(rows):
                similarity = float(similarities[i])
                
                if similarity >= min_similarity:
                    results.append({
                        'chunk_id': row[0],
                        'document_id': row[1],
                        'chunk_index': row[2],
                        'content': row[3],
                        'token_count': row[5],
                        'metadata': json.loads(row[6]) if row[6] else {},
                        'subject': row[7],
                        'topic': row[8],
                        'difficulty': row[9],
                        'source_file': row[10],
                        'similarity': similarity
                    })
            
            # Sort by similarity and return top-K
            results.sort(key=lambda x: x['similarity'], reverse=True)
            top_results = results[:top_k]
            
            logger.info(f"Search returned {len(top_results)} results (from {len(rows)} chunks)")
            return top_results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        return float(np.dot(vec1_norm, vec2_norm))
    
    def get_document_stats(self, doc_id: str) -> Optional[Dict]:
        """Get statistics for a specific document."""
        try:
            with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
                cursor = conn.execute("""
                    SELECT 
                        d.*,
                        COUNT(c.id) as actual_chunk_count,
                        AVG(c.token_count) as avg_chunk_tokens
                    FROM documents d
                    LEFT JOIN chunks c ON d.id = c.document_id
                    WHERE d.id = ?
                    GROUP BY d.id
                """, (doc_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return {
                    'id': row[0],
                    'source_file': row[1],
                    'subject': row[2],
                    'topic': row[3],
                    'difficulty': row[4],
                    'language': row[5],
                    'chunk_count': row[8],
                    'avg_chunk_tokens': row[9]
                }
        except Exception as e:
            logger.error(f"Error getting stats for '{doc_id}': {e}")
            return None
    
    def get_database_stats(self) -> Dict:
        """Get overall database statistics."""
        try:
            with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
                # Count documents and chunks
                cursor = conn.execute("""
                    SELECT 
                        COUNT(DISTINCT d.id) as doc_count,
                        COUNT(c.id) as chunk_count,
                        AVG(c.token_count) as avg_chunk_size,
                        SUM(c.token_count) as total_tokens
                    FROM documents d
                    LEFT JOIN chunks c ON d.id = c.document_id
                """)
                row = cursor.fetchone()
                
                # Get subjects distribution
                cursor = conn.execute("""
                    SELECT subject, COUNT(*) as count
                    FROM documents
                    WHERE subject IS NOT NULL
                    GROUP BY subject
                    ORDER BY count DESC
                """)
                subjects = dict(cursor.fetchall())
                
                return {
                    'total_documents': row[0] or 0,
                    'total_chunks': row[1] or 0,
                    'avg_chunk_size': row[2] or 0,
                    'total_tokens': row[3] or 0,
                    'subjects_distribution': subjects
                }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document and its chunks."""
        try:
            with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
                conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                # Chunks are deleted automatically due to CASCADE
            logger.info(f"Document '{doc_id}' deleted")
            return True
        except Exception as e:
            logger.error(f"Error deleting document '{doc_id}': {e}")
            return False
    
    def clear_database(self) -> bool:
        """Clear all documents and chunks."""
        try:
            with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
                conn.execute("DELETE FROM chunks")
                conn.execute("DELETE FROM documents")
            logger.info("Database cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False