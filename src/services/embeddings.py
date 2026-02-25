import logging
import numpy as np
import gc
from sentence_transformers import SentenceTransformer
from typing import List, Union
import torch
import threading

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Singleton service for generating text embeddings using SentenceTransformers.
    Supports multilingual text including Kazakh language.
    """
    
    _instance = None
    _model = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-checked locking
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize embedding model (loads only once due to singleton pattern).
        
        Args:
            model_name: HuggingFace model identifier
        """
        if self._model is None:
            logger.info(f"Loading embedding model: {model_name}")
            try:
                self._model = SentenceTransformer(model_name)
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    self._model = self._model.to('cuda')
                    logger.info("Embedding model loaded on GPU")
                else:
                    logger.info("Embedding model loaded on CPU")
                    
                # Get embedding dimension
                self.embedding_dim = self._model.get_sentence_embedding_dimension()
                logger.info(f"Embedding dimension: {self.embedding_dim}")
                
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
    
    def encode(self, texts: Union[str, List[str]], 
               batch_size: int = 32,
               show_progress: bool = False,
               normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length (for cosine similarity)
            
        Returns:
            numpy array of embeddings, shape (n_texts, embedding_dim)
        """
        try:
            # Convert single string to list
            if isinstance(texts, str):
                texts = [texts]
            
            # Generate embeddings
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1, higher = more similar)
        """
        try:
            # Ensure embeddings are normalized
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Cosine similarity = dot product of normalized vectors
            similarity = np.dot(embedding1, embedding2)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def batch_similarity(self, query_embedding: np.ndarray, 
                        chunk_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarity between one query and multiple chunks (optimized).
        
        Args:
            query_embedding: Single query embedding, shape (embedding_dim,)
            chunk_embeddings: Multiple chunk embeddings, shape (n_chunks, embedding_dim)
            
        Returns:
            Array of similarity scores, shape (n_chunks,)
        """
        try:
            # Normalize query
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            
            # Normalize chunks (if not already normalized)
            chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
            chunk_embeddings_norm = chunk_embeddings / chunk_norms
            
            # Matrix multiplication for batch cosine similarity
            similarities = np.dot(chunk_embeddings_norm, query_norm)
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error in batch similarity computation: {e}")
            return np.zeros(len(chunk_embeddings))
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self._model._modules['0'].auto_model.name_or_path if self._model else None,
            "embedding_dim": self.embedding_dim if self._model else None,
            "device": str(self._model.device) if self._model else None,
            "max_seq_length": self._model.max_seq_length if self._model else None
        }
    
    def unload_model(self):
        """Выгрузка модели эмбеддингов."""
        if hasattr(self, '_model') and self._model is not None:
            del self._model
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        logger.info("Embedding model unloaded.")


# Convenience function for quick encoding
def encode_text(text: str) -> np.ndarray:
    """Quick function to encode single text."""
    service = EmbeddingService()
    return service.encode(text)[0]  # Return single embedding