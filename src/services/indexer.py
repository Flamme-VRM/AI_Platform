import os
import json
import glob
import logging
from typing import List, Dict, Optional
from tqdm import tqdm

from .embeddings import EmbeddingService
from .chunker import DocumentChunker
from .vector_db import VectorDB
from ..config import settings

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Orchestrates the full indexing pipeline:
    1. Load documents from RAG/ directory
    2. Chunk documents
    3. Generate embeddings
    4. Store in vector database
    """
    
    def __init__(self, 
                 rag_directory: str = "RAG",
                 db_path: str = None,
                 chunk_size: int = None,
                 overlap: int = None):
        """
        Initialize indexer with all required services.
        
        Args:
            rag_directory: Directory containing JSON documents
            db_path: Path to SQLite database
            chunk_size: Chunk size in tokens
            overlap: Overlap size in tokens
        """
        if db_path is None:
            db_path = settings.VECTOR_DB_PATH
        if chunk_size is None:
            chunk_size = settings.CHUNK_SIZE
        if overlap is None:
            overlap = settings.CHUNK_OVERLAP

        self.rag_directory = rag_directory
        self.db_path = db_path
        
        # Initialize services
        logger.info("Initializing indexing services...")
        self.embedding_service = EmbeddingService()
        self.chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
        self.vector_db = VectorDB(db_path=db_path)
        
        logger.info(f"DocumentIndexer ready: {rag_directory} -> {db_path}")
    
    def load_documents_from_directory(self) -> List[Dict]:
        """
        Load all JSON documents from RAG directory.
        
        Returns:
            List of document dictionaries
        """
        json_files = glob.glob(os.path.join(self.rag_directory, "*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {self.rag_directory}")
            return []
        
        documents = []
        logger.info(f"Found {len(json_files)} JSON files")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                
                # Add source file to metadata
                doc['source_file'] = os.path.basename(file_path)
                
                # Validate required fields
                if 'id' not in doc or 'content' not in doc:
                    logger.warning(f"Skipping {file_path}: missing 'id' or 'content'")
                    continue
                
                documents.append(doc)
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {file_path}: {e}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def index_document(self, document: Dict, show_progress: bool = False) -> bool:
        """
        Index a single document: chunk, embed, and store.
        
        Args:
            document: Document dictionary
            show_progress: Show progress for embeddings
            
        Returns:
            True if successful
        """
        doc_id = document['id']
        
        try:
            # 1. Add document metadata to DB
            logger.debug(f"Processing document: {doc_id}")
            self.vector_db.add_document(
                doc_id=doc_id,
                source_file=document.get('source_file', 'unknown'),
                metadata={k: v for k, v in document.items() 
                         if k not in ['id', 'content', 'source_file']}
            )
            
            # 2. Chunk the document
            chunks = self.chunker.chunk_document(document)
            
            if not chunks:
                logger.warning(f"No chunks created for document {doc_id}")
                return False
            
            # 3. Generate embeddings for chunks
            chunk_texts = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_service.encode(
                chunk_texts,
                show_progress=show_progress
            )
            
            # 4. Store chunks with embeddings
            added_count = self.vector_db.add_chunks(doc_id, chunks, embeddings)
            
            logger.info(f"✓ Document '{doc_id}': {added_count} chunks indexed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to index document '{doc_id}': {e}")
            return False
    
    def index_all(self, clear_existing: bool = False) -> Dict:
        """
        Index all documents from RAG directory.
        
        Args:
            clear_existing: Whether to clear existing database first
            
        Returns:
            Statistics dictionary
        """
        start_time = __import__('time').time()
        
        logger.info("="*60)
        logger.info("Starting full indexing pipeline")
        logger.info("="*60)
        
        # Clear database if requested
        if clear_existing:
            logger.info("Clearing existing database...")
            self.vector_db.clear_database()
        
        # Load documents
        logger.info(f"Loading documents from {self.rag_directory}...")
        documents = self.load_documents_from_directory()
        
        if not documents:
            logger.error("No documents to index!")
            return {'success': False, 'error': 'No documents found'}
        
        # Index each document with progress bar
        logger.info(f"Indexing {len(documents)} documents...")
        successful = 0
        failed = 0
        
        for doc in tqdm(documents, desc="Indexing documents"):
            if self.index_document(doc, show_progress=False):
                successful += 1
            else:
                failed += 1
        
        # Get final statistics
        db_stats = self.vector_db.get_database_stats()
        
        elapsed_time = __import__('time').time() - start_time
        
        results = {
            'success': True,
            'documents_processed': len(documents),
            'documents_successful': successful,
            'documents_failed': failed,
            'total_chunks': db_stats.get('total_chunks', 0),
            'total_tokens': db_stats.get('total_tokens', 0),
            'avg_chunk_size': db_stats.get('avg_chunk_size', 0),
            'elapsed_time': elapsed_time,
            'subjects': db_stats.get('subjects_distribution', {})
        }
        
        logger.info("="*60)
        logger.info("Indexing complete!")
        logger.info(f"✓ Documents indexed: {successful}/{len(documents)}")
        logger.info(f"✓ Total chunks: {results['total_chunks']}")
        logger.info(f"✓ Total tokens: {results['total_tokens']:,}")
        logger.info(f"✓ Time elapsed: {elapsed_time:.2f}s")
        logger.info("="*60)
        
        return results
    
    def reindex_document(self, doc_id: str) -> bool:
        """
        Re-index a specific document (delete old chunks and re-index).
        
        Args:
            doc_id: Document ID to re-index
            
        Returns:
            True if successful
        """
        logger.info(f"Re-indexing document: {doc_id}")
        
        # Delete existing document
        self.vector_db.delete_document(doc_id)
        
        # Find and load the document
        json_files = glob.glob(os.path.join(self.rag_directory, "*.json"))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                
                if doc.get('id') == doc_id:
                    doc['source_file'] = os.path.basename(file_path)
                    return self.index_document(doc)
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        logger.error(f"Document '{doc_id}' not found in {self.rag_directory}")
        return False
    
    def get_indexing_status(self) -> Dict:
        """Get current indexing status and statistics."""
        db_stats = self.vector_db.get_database_stats()
        
        # Count files in RAG directory
        rag_files = len(glob.glob(os.path.join(self.rag_directory, "*.json")))
        
        indexed_docs = db_stats.get('total_documents', 0)
        
        return {
            'rag_files_available': rag_files,
            'documents_indexed': indexed_docs,
            'needs_indexing': rag_files != indexed_docs,
            'database_stats': db_stats,
            'model_info': self.embedding_service.get_model_info()
        }
    
    def unload_model(self):
        """Unload embedding model from memory (useful after indexing)."""
        logger.info("Unloading embedding model from memory...")
        self.embedding_service.unload_model()
        logger.info("✓ Model unloaded. Memory freed.")


# Convenience function for quick indexing
def quick_index(rag_dir: str = "RAG", 
                db_path: str = None,
                clear_existing: bool = False) -> Dict:
    """
    Quick function to index all documents.
    
    Args:
        rag_dir: RAG directory path
        db_path: Database path
        clear_existing: Clear existing data
        
    Returns:
        Indexing statistics
    """
    indexer = DocumentIndexer(rag_directory=rag_dir, db_path=db_path)
    results = indexer.index_all(clear_existing=clear_existing)
    indexer.unload_model()  # Free memory
    return results