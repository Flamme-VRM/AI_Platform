import asyncio
import logging
import os
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from .cache import CacheService
from .embeddings import EmbeddingService
from .vector_db import VectorDB
from ..config import settings

logger = logging.getLogger(__name__)


class AIService:
    def __init__(self, api_key: str, model_name: str, cache_service: CacheService, 
                 document_loader=None):  # document_loader is deprecated but kept for compatibility
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.cache = cache_service
        self.system_prompt = self._load_system_prompt_from_file()
        
        # Initialize vector search components
        self.use_vector_search = True
        try:
            self.embedding_service = EmbeddingService()
            self.vector_db = VectorDB(db_path=settings.VECTOR_DB_PATH)
            
            # Check if database has data
            db_stats = self.vector_db.get_database_stats()
            if db_stats.get('total_chunks', 0) == 0:
                logger.warning("Vector database is empty! Run 'python scripts/index_documents.py' first")
                self.use_vector_search = False
            else:
                logger.info(f"Vector search ready: {db_stats['total_chunks']} chunks available")
        except Exception as e:
            logger.error(f"Failed to initialize vector search: {e}")
            self.use_vector_search = False

        if not self.system_prompt:
            logger.warning("System prompt could not be loaded")
            raise SystemExit()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def _generate_with_retry(self, prompt: str):
        return await asyncio.to_thread(self.model.generate_content, prompt)

    def _load_system_prompt_from_file(self, filename: str = None) -> str:
        """Load system prompt from a text file."""
        if filename is None:
            filename = settings.SYSPROMPT_PATH

        try:
            project_root = os.getcwd()
            file_path = os.path.join(project_root, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                system_prompt = file.read().strip()
                logger.info(f"Successfully loaded system prompt from {filename}")
                return system_prompt
                
        except FileNotFoundError:
            logger.error(f"System prompt file '{filename}' not found in project root")
            return ""
        except Exception as e:
            logger.error(f"Error reading system prompt file '{filename}': {str(e)}")
            return ""

    def _retrieve_relevant_documents(self, query: str, top_k: int = None) -> str:
        """
        Retrieve relevant educational content using semantic search.
        
        Args:
            query: User query
            top_k: Number of top chunks to retrieve
            
        Returns:
            Formatted context string for AI
        """
        if top_k is None:
            top_k = settings.TOP_K_RESULTS

        if not self.use_vector_search:
            logger.debug("Vector search not available, skipping retrieval")
            return ""
        
        try:
            # Generate embedding for query
            query_embedding = self.embedding_service.encode(query)[0]
            
            # Search for similar chunks
            results = self.vector_db.search(
                query_embedding=query_embedding,
                top_k=top_k,
                min_similarity=settings.MIN_SIMILARITY
            )
            
            if not results:
                logger.info(f"No relevant documents found for query: {query[:50]}...")
                return ""
            
            # Format results for AI context
            context_parts = []
            for i, result in enumerate(results, 1):
                similarity_pct = result['similarity'] * 100
                
                context_parts.append(
                    f"[Дереккөз {i}] (релеванттылық: {similarity_pct:.1f}%)\n"
                    f"Пән: {result.get('subject', 'Белгісіз')} | "
                    f"Тақырып: {result.get('topic', 'Белгісіз')}\n"
                    f"Мазмұны: {result['content']}\n"
                )
            
            context = "\n" + "="*50 + "\n" + "\n".join(context_parts)
            
            # Форматируем список similarity для логирования
            similarities_str = ', '.join([f"{r['similarity']:.2f}" for r in results])
            
            logger.info(f"Retrieved {len(results)} relevant chunks for query "
                       f"(similarities: [{similarities_str}])")
            
            return context

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return ""

    async def generate_response(self, user_id: int, text: str) -> str:
        try:
            history = self.cache.get_user_history(user_id)
            history.append(f'User: {text}')

            recent_history = history[-10:]

            # Retrieve relevant educational content via semantic search
            relevant_context = self._retrieve_relevant_documents(text)

            full_prompt = f"{self.system_prompt}\n\nConversation History:\n" + "\n".join(recent_history)

            # Add RAG context if available
            if relevant_context:
                full_prompt += f"\n\nРелевантты оқу материалдары:\n{relevant_context}"
            else:
                logger.info(f"No relevant context found for user {user_id} query: {text[:50]}...")

            # Check cache
            cached_response = self.cache.get_cached_response(full_prompt)
            if cached_response:
                logger.info(f"Using cached response for user {user_id}")
                history.append(f"AsylBILIM: {cached_response}")
                self.cache.save_user_history(user_id, history)
                return cached_response

            # Generate new response with retry
            response = await self._generate_with_retry(full_prompt)

            if not response or not response.text:
                logger.error(f"Empty response from AI model for user {user_id}")
                return "Кешіріңіз, жауап алу мүмкін болмады. Қайталап көріңіз."

            history.append(f"AsylBILIM: {response.text}")
            self.cache.save_user_history(user_id, history)
            self.cache.cache_response(full_prompt, response.text)

            return response.text

        except genai.types.BlockedPromptException as e:
            logger.error(f"Content blocked for user {user_id}: {e}")
            return "Кешіріңіз, сұрақ қауіпсіздік фильтрлерімен бөгелді. Басқаша сұраңыз."

        except genai.types.StopCandidateException as e:
            logger.error(f"AI generation stopped for user {user_id}: {e}")
            return "Кешіріңіз, жауап генерациясы тоқтатылды. Қайталап көріңіз."

        except Exception as e:
            logger.error(f"Unexpected error for user {user_id}: {type(e).__name__}: {str(e)}")
            return "Кешіріңіз, техникалық қате орын алды. Кейінірек қайталап көріңіз."
         
    async def generate_response_for_chat(
        self, 
        user_id: int, 
        chat_id: str, 
        text: str, 
        history: list
    ) -> str:
        """
        Генерация ответа для конкретного чата
        """
        try:
            recent_history = history[-10:] if len(history) > 10 else history
            relevant_context = self._retrieve_relevant_documents(text)

            full_prompt = (
                f"{self.system_prompt}\n\n"
                f"Conversation History:\n" + "\n".join(recent_history)
            )

            if relevant_context:
                full_prompt += f"\n\n{relevant_context}"

            cached_response = self.cache.get_cached_response(full_prompt)
            if cached_response:
                logger.info(f"Using cached response for chat {chat_id}")
                return cached_response

            response = await self._generate_with_retry(full_prompt)

            if not response or not response.text:
                return "Кешіріңіз, жауап алу мүмкін болмады."

            self.cache.cache_response(full_prompt, response.text)
            return response.text

        except Exception as e:
            logger.error(f"Error for chat {chat_id}: {e}")
            return "Кешіріңіз, техникалық қате орын алды."