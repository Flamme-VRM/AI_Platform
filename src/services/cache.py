import logging
import hashlib
import time
from typing import Optional, Dict, List
import redis
import msgpack
from ..config import settings

logger = logging.getLogger(__name__)


class CacheService:
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, username: str = "default", password: str = None):
        # Создаем пул соединений ОДИН РАЗ при старте бота
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            username=username,
            password=password,
            max_connections=settings.REDIS_MAX_CONNECTIONS,      # Максимум 50 соединений
            decode_responses=False,
            socket_keepalive=True,   # Держим соединения живыми
            socket_timeout=5,        # Таймаут на операции
            retry_on_timeout=True    # Retry при таймаутах
        )
        
        # Клиент переиспользует соединения из пула
        self.client = redis.Redis(connection_pool=self.pool)
        self._test_connection()

    def _test_connection(self):
        try:
            self.client.ping()
            logger.info("Redis connection successful")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise SystemExit("Redis is required for bot operation")

    def close(self):
        """Закрытие соединений Redis."""
        try:
            self.client.close()
            self.pool.disconnect()
            logger.info("Redis connections closed.")
        except Exception as e:
            logger.error(f"Error closing Redis: {e}")

    def get_user_history(self, user_id: int) -> List[str]:
        try:
            key = f"user_history:{user_id}"
            history_data = self.client.get(key)
            return msgpack.unpackb(history_data, use_list=True) if history_data else []
        except Exception as e:
            logger.debug(f"Error getting user history: {e}")
            return []

    def save_user_history(self, user_id: int, history: List[str]):
        try:
            key = f"user_history:{user_id}"
            self.client.setex(key, 86400 * 7, msgpack.packb(history[-50:]))
        except Exception as e:
            logger.debug(f"Error saving user history: {e}")

    def get_cached_response(self, prompt: str) -> Optional[str]:
        try:
            key = f"ai_cache:{hashlib.md5(prompt.encode()).hexdigest()}"
            response = self.client.get(key)
            return response.decode('utf-8') if response else None
        except Exception as e:
            logger.debug(f"Error getting cached response: {e}")
            return None

    def cache_response(self, prompt: str, response: str):
        try:
            key = f"ai_cache:{hashlib.md5(prompt.encode()).hexdigest()}"
            self.client.setex(key, 3600, response)
        except Exception as e:
            logger.debug(f"Error caching response: {e}")

    def set_user_session(self, user_id: int, session_data: Dict):
        try:
            key = f"user_session:{user_id}"
            self.client.setex(key, 3600 * 24, msgpack.packb(session_data))
        except Exception as e:
            logger.debug(f"Error setting user session: {e}")

    def get_user_session(self, user_id: int) -> Dict:
        try:
            key = f"user_session:{user_id}"
            session_data = self.client.get(key)
            return msgpack.unpackb(session_data) if session_data else {}
        except Exception as e:
            logger.debug(f"Error getting user session: {e}")
            return {}
    def check_rate_limit(self, user_id: int, limit: int = None, window_hours: int = None) -> tuple[bool, int, int]:
        """
        Check if user has exceeded rate limit.
        Returns: (is_allowed, current_count, time_until_reset)
        """
        if limit is None:
            limit = settings.RATE_LIMIT
        if window_hours is None:
            window_hours = settings.RATE_WINDOW_HOURS

        try:
            key = f"rate_limit:{user_id}"
            current_time = int(time.time())
            window_seconds = window_hours * 3600
            
            # Get current rate limit data
            rate_data = self.client.get(key)
            
            if not rate_data:
                # First message - initialize counter
                data = {
                    'count': 1,
                    'window_start': current_time
                }
                self.client.setex(key, window_seconds, msgpack.packb(data))
                return True, 1, 0
            
            data = msgpack.unpackb(rate_data)
            window_start = data.get('window_start', current_time)
            count = data.get('count', 0)
            
            # Check if window has expired
            if current_time - window_start >= window_seconds:
                # Reset window
                data = {
                    'count': 1,
                    'window_start': current_time
                }
                self.client.setex(key, window_seconds, msgpack.packb(data))
                return True, 1, 0
            
            # Check if limit exceeded
            if count >= limit:
                time_until_reset = window_seconds - (current_time - window_start)
                return False, count, time_until_reset
            
            # Increment counter
            data['count'] = count + 1
            remaining_time = window_seconds - (current_time - window_start)
            self.client.setex(key, remaining_time, msgpack.packb(data))
            
            return True, count + 1, 0
            
        except Exception as e:
            logger.error(f"Error checking rate limit for user {user_id}: {e}")
            # Allow message if there's an error (fail open)
            return True, 0, 0

    def get_rate_limit_info(self, user_id: int, limit: int = None, window_hours: int = None) -> Dict:
        """Get detailed rate limit information for a user."""
        if limit is None:
            limit = settings.RATE_LIMIT
        if window_hours is None:
            window_hours = settings.RATE_WINDOW_HOURS

        try:
            key = f"rate_limit:{user_id}"
            current_time = int(time.time())
            window_seconds = window_hours * 3600
            
            rate_data = self.client.get(key)
            
            if not rate_data:
                return {
                    'count': 0,
                    'limit': limit,
                    'remaining': limit,
                    'reset_time': current_time + window_seconds
                }
            
            data = msgpack.unpackb(rate_data)
            window_start = data.get('window_start', current_time)
            count = data.get('count', 0)
            
            # Check if window has expired
            if current_time - window_start >= window_seconds:
                return {
                    'count': 0,
                    'limit': limit,
                    'remaining': limit,
                    'reset_time': current_time + window_seconds
                }
            
            reset_time = window_start + window_seconds
            remaining = max(0, limit - count)
            
            return {
                'count': count,
                'limit': limit,
                'remaining': remaining,
                'reset_time': reset_time
            }
            
        except Exception as e:
            logger.error(f"Error getting rate limit info for user {user_id}: {e}")
            return {
                'count': 0,
                'limit': limit,
                'remaining': limit,
                'reset_time': current_time + window_seconds
            }