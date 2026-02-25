import logging
from typing import List, Dict
import re
import tiktoken

logger = logging.getLogger(__name__)


class DocumentChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        # Используем tiktoken для точного подсчета
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4/Gemini compatible
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def count_tokens(self, text: str) -> int:
        """Точный подсчет токенов"""
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        # Точный подсчет вместо приблизительного
        total_tokens = self.count_tokens(text)
        
        if total_tokens <= self.chunk_size:
            return [{
                'content': text,
                'chunk_index': 0,
                'token_count': total_tokens,  # Точное значение!
                'metadata': metadata or {}
            }]
        
        # Разбиение на предложения
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)  # Точно!
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Сохраняем чанк с точным количеством токенов
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'content': chunk_text,
                    'chunk_index': len(chunks),
                    'token_count': self.count_tokens(chunk_text),  # Точно!
                    'metadata': metadata or {}
                })
                
                # Начинаем новый чанк с overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk, 
                    self.overlap
                )
                current_chunk = overlap_sentences + [sentence]
                current_tokens = self.count_tokens(' '.join(current_chunk))
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add last chunk if any
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'chunk_index': len(chunks),
                'token_count': self.count_tokens(chunk_text),
                'metadata': metadata or {}
            })
        
        return chunks
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Chunk a structured document (from JSON).
        
        Args:
            document: Document dict with 'id', 'content', and metadata
            
        Returns:
            List of chunk dictionaries
        """
        doc_id = document.get('id', 'unknown')
    
        # === НОВАЯ ЛОГИКА: Объединяем все текстовые поля ===
        content_parts = []
        
        # Основной контент
        if 'content' in document:
            content_parts.append(document['content'])
        
        # Контекст (часто содержит полезную инфу)
        if 'context' in document:
            content_parts.append(document['context'])
        
        # Шаги решения (для математики и информатики)
        if 'solution_steps' in document and isinstance(document['solution_steps'], list):
            content_parts.extend(document['solution_steps'])
        
        # Объяснение
        if 'explanation' in document:
            content_parts.append(document['explanation'])
        
        # Правильный ответ (для контекста)
        if 'correct_answer' in document:
            content_parts.append(f"Жауап: {document['correct_answer']}")
        
        # Код (для информатики)
        if 'code_snippet' in document:
            content_parts.append(f"Код: {document['code_snippet']}")
        
        # Объединяем всё
        full_content = " | ".join([str(p) for p in content_parts if p])
        # === КОНЕЦ НОВОЙ ЛОГИКИ ===
        
        # Extract metadata (everything except id and content)
        metadata = {k: v for k, v in document.items() 
                    if k not in ['id', 'content', 'context', 'solution_steps', 
                                'explanation', 'code_snippet']}
        metadata['document_id'] = doc_id
        
        chunks = self.chunk_text(full_content, metadata)
        
        logger.debug(f"Document '{doc_id}' chunked into {len(chunks)} pieces")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text 
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        Handles Kazakh, Russian, and English punctuation.
        """
        # Sentence boundaries: . ! ? and their combinations
        # Keep sentence-ending punctuation with the sentence
        pattern = r'(?<=[.!?])\s+(?=[А-ЯЁӘІҢҒҮҰҚӨҺA-ZƏ])'
        sentences = re.split(pattern, text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_tokens: int) -> List[str]:
        """
        Get last few sentences for overlap based on token count.
        
        Args:
            sentences: List of sentences
            overlap_tokens: Target number of overlap tokens
            
        Returns:
            List of sentences for overlap
        """
        if not sentences:
            return []
        
        overlap_sentences = []
        current_tokens = 0
        
        # Take sentences from the end until we reach overlap size
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if current_tokens + sentence_tokens > overlap_tokens:
                break
            overlap_sentences.insert(0, sentence)
            current_tokens += sentence_tokens
        
        return overlap_sentences
    
    def get_stats(self, chunks: List[Dict]) -> Dict:
        """Get statistics about chunks."""
        if not chunks:
            return {}
        
        token_counts = [c['token_count'] for c in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(token_counts) / len(token_counts),
            'min_chunk_size': min(token_counts),
            'max_chunk_size': max(token_counts),
            'total_tokens': sum(token_counts)
        }