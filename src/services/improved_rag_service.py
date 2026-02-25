"""
Improved RAG Service with chunking and semantic search
Properly handles document splitting before embedding
"""

import logging
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass
from .document_loader import DocumentLoader, Document

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    doc_id: str
    chunk_index: int
    metadata: Dict
    embedding: np.ndarray = None


class ImprovedRAGService:
    """Enhanced RAG with chunking and semantic search"""
    
    def __init__(
        self, 
        document_loader: DocumentLoader, 
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size: int = 300,
        chunk_overlap: int = 50
    ):
        """
        Initialize RAG service with chunking and semantic search
        
        Args:
            document_loader: DocumentLoader instance
            model_name: Sentence transformer model
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.document_loader = document_loader
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[Chunk] = []
        self.chunk_embeddings = None
        
        # Load embedding model
        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
        
        # Initialize documents with chunking
        self._initialize_with_chunking()
    
    def _split_text_into_chunks(self, text: str, metadata: Dict, doc_id: str) -> List[Chunk]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to split
            metadata: Document metadata
            doc_id: Document ID
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        text_length = len(text)
        
        # If text is smaller than chunk size, return as single chunk
        if text_length <= self.chunk_size:
            chunks.append(Chunk(
                text=text,
                doc_id=doc_id,
                chunk_index=0,
                metadata=metadata
            ))
            return chunks
        
        # Split into overlapping chunks
        start = 0
        chunk_index = 0
        
        while start < text_length:
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < text_length:
                # Look for sentence end markers
                for marker in ['. ', '! ', '? ', '.\n', '\n']:
                    last_marker = text.rfind(marker, start, end)
                    if last_marker != -1:
                        end = last_marker + len(marker)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    metadata=metadata
                ))
                chunk_index += 1
            
            # Move start with overlap
            start = end - self.chunk_overlap
            
            # Avoid infinite loop
            if start >= text_length:
                break
        
        return chunks
    
    def _initialize_with_chunking(self):
        """Load documents, split into chunks, and create embeddings"""
        try:
            documents = self.document_loader.load_documents()
            
            if not documents:
                logger.warning("No documents loaded for RAG")
                return
            
            logger.info(f"Processing {len(documents)} documents with chunking...")
            
            # Split all documents into chunks
            all_chunks = []
            for doc in documents:
                doc_chunks = self._split_text_into_chunks(
                    text=doc.content,
                    metadata=doc.metadata,
                    doc_id=doc.id
                )
                all_chunks.extend(doc_chunks)
            
            self.chunks = all_chunks
            logger.info(f"Created {len(self.chunks)} chunks from {len(documents)} documents")
            
            # Create embeddings for all chunks
            if self.embedding_model and self.chunks:
                chunk_texts = []
                for chunk in self.chunks:
                    # Enrich chunk text with metadata for better context
                    enriched_text = chunk.text
                    if 'subject' in chunk.metadata:
                        enriched_text = f"Пән: {chunk.metadata['subject']}. {enriched_text}"
                    if 'topic' in chunk.metadata:
                        enriched_text = f"{enriched_text} Тақырып: {chunk.metadata['topic']}"
                    
                    chunk_texts.append(enriched_text)
                
                # Generate embeddings in smaller batches to avoid memory issues
                logger.info(f"Creating embeddings for {len(chunk_texts)} chunks...")
                all_embeddings = []
                batch_size = 16  # Smaller batch for CPU
                
                for i in range(0, len(chunk_texts), batch_size):
                    batch = chunk_texts[i:i+batch_size]
                    batch_embeddings = self.embedding_model.encode(
                        batch,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    all_embeddings.append(batch_embeddings)
                    
                    if (i // batch_size + 1) % 10 == 0:
                        logger.info(f"Processed {i + len(batch)}/{len(chunk_texts)} chunks...")
                
                self.chunk_embeddings = np.vstack(all_embeddings)
                logger.info(f"Created embeddings: {self.chunk_embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Error initializing documents with chunking: {e}", exc_info=True)
            self.chunks = []
            self.chunk_embeddings = None
    
    def retrieve_relevant_documents(
        self, 
        query: str, 
        top_k: int = 5, 
        similarity_threshold: float = 0.3
    ) -> str:
        """
        Retrieve relevant chunks using semantic similarity
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            Formatted context string with relevant chunks
        """
        if not self.chunks or self.chunk_embeddings is None:
            logger.warning("No chunks or embeddings available")
            return ""
        
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode(
                [query],
                convert_to_numpy=True
            )[0]
            
            # Calculate cosine similarity
            similarities = self._cosine_similarity(query_embedding, self.chunk_embeddings)
            
            # Get top k most similar chunks
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            relevant_chunks = []
            seen_docs = set()  # Avoid duplicate document IDs
            
            for idx in top_indices:
                similarity = similarities[idx]
                
                # Filter by threshold
                if similarity >= similarity_threshold:
                    chunk = self.chunks[idx]
                    
                    # Optionally deduplicate by document ID
                    # Comment out if you want multiple chunks from same doc
                    # if chunk.doc_id not in seen_docs:
                    relevant_chunks.append((chunk, similarity))
                    seen_docs.add(chunk.doc_id)
            
            if not relevant_chunks:
                logger.info(f"No relevant chunks found for query: {query[:50]}...")
                return ""
            
            # Format context
            context_parts = []
            for i, (chunk, similarity) in enumerate(relevant_chunks, 1):
                context_parts.append(
                    f"--- Дереккөз {i} (релевантность: {similarity:.2f}) ---\n"
                    f"ID: {chunk.doc_id} [chunk {chunk.chunk_index}]\n"
                    f"Пән: {chunk.metadata.get('subject', 'Белгісіз')}\n"
                    f"Тақырып: {chunk.metadata.get('topic', 'Белгісіз')}\n"
                    f"Мазмұн:\n{chunk.text}\n"
                )
            
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for query: {query[:50]}...")
            return "\n" + "="*70 + "\n" + "\n".join(context_parts)
        
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return ""
    
    def _cosine_similarity(self, query_embedding: np.ndarray, chunk_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and chunks"""
        # Normalize vectors
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        chunk_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        similarities = np.dot(chunk_norms, query_norm)
        
        return similarities
    
    def add_document(self, document: Document):
        """Add a new document, chunk it, and update embeddings"""
        try:
            # Split document into chunks
            new_chunks = self._split_text_into_chunks(
                text=document.content,
                metadata=document.metadata,
                doc_id=document.id
            )
            
            self.chunks.extend(new_chunks)
            
            # Create embeddings for new chunks
            chunk_texts = []
            for chunk in new_chunks:
                enriched_text = chunk.text
                if 'subject' in chunk.metadata:
                    enriched_text = f"Пән: {chunk.metadata['subject']}. {enriched_text}"
                if 'topic' in chunk.metadata:
                    enriched_text = f"{enriched_text} Тақырып: {chunk.metadata['topic']}"
                chunk_texts.append(enriched_text)
            
            new_embeddings = self.embedding_model.encode(
                chunk_texts,
                convert_to_numpy=True
            )
            
            # Update embeddings array
            if self.chunk_embeddings is None:
                self.chunk_embeddings = new_embeddings
            else:
                self.chunk_embeddings = np.vstack([self.chunk_embeddings, new_embeddings])
            
            logger.info(f"Added document {document.id} as {len(new_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
    
    def get_stats(self) -> dict:
        """Get RAG system statistics"""
        subjects = {}
        for chunk in self.chunks:
            subj = chunk.metadata.get('subject', 'Unknown')
            subjects[subj] = subjects.get(subj, 0) + 1
        
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(set(c.doc_id for c in self.chunks)),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embeddings_shape": self.chunk_embeddings.shape if self.chunk_embeddings is not None else None,
            "embedding_model": self.embedding_model.__class__.__name__ if self.embedding_model else None,
            "subjects": subjects
        }


# ============================================
# INTEGRATION EXAMPLE
# ============================================

"""
In your src/services/ai.py:

from .improved_rag_service import ImprovedRAGService

# In __init__:
self.rag_service = ImprovedRAGService(
    document_loader=document_loader,
    chunk_size=300,  # ~1-2 paragraphs
    chunk_overlap=50  # Keep context between chunks
)

# In generate_response():
relevant_context = self.rag_service.retrieve_relevant_documents(
    query=text, 
    top_k=5,  # More chunks = better context
    similarity_threshold=0.3
)

# To check RAG stats:
logger.info(f"RAG Stats: {self.rag_service.get_stats()}")
"""