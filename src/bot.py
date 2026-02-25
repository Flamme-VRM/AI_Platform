import logging
import torch
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from .config import settings
from .services.cache import CacheService
from .services.ai import AIService
from .services.speech_to_text import SpeechToTextService
from .handlers.message_handler import MessageHandler

logger = logging.getLogger(__name__)


class AsylBilim:
    def __init__(self):
        self.cache_service = CacheService(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            username=settings.REDIS_USERNAME,
            password=settings.REDIS_PASSWORD
        )

        self.speech_service = SpeechToTextService()

        self.ai_service = AIService(
            api_key=settings.LLM_API_KEY,
            model_name=settings.MODEL,
            cache_service=self.cache_service,
        )

        self.message_handler = MessageHandler(
            self.ai_service, 
            self.cache_service, 
            self.speech_service
        )

        bot_token = settings.BOT_TOKEN
        logger.info(f"Bot token loaded: {bot_token[:20]}..." if bot_token else "Bot token is None")
        
        self.bot = Bot(token=bot_token)
        self.dp = Dispatcher()

        self.dp.message(Command("start"))(self.message_handler.handle_start)
        self.dp.message(Command("clear"))(self.message_handler.handle_clear)
        self.dp.message(Command("status"))(self.message_handler.handle_message)
        self.dp.message(Command("help"))(self.message_handler.handle_help)
        self.dp.message(F.voice)(self.message_handler.handle_voice)
        self.dp.message()(self.message_handler.handle_message)

    def show_banner(self):
        banner = r"""
 _____ _                                   __     ______  __  __ 
|  ___| | __ _ _ __ ___  _ __ ___   ___    \ \   / /  _ \|  \/  |
| |_  | |/ _` | '_ ` _ \| '_ ` _ \ / _ \____\ \ / /| |_) | |\/| |
|  _| | | (_| | | | | | | | | | | |  __/_____\ V / |  _ <| |  | |
|_|   \_|\__,_|_| |_| |_|_| |_| |_|\___|      \_/  |_| \_\_|  |_|
        """
        print(banner)
        logger.info("AsylBILIM bot starting up...")

    async def shutdown(self):
        """Централизованное завершение работы всех компонентов."""
        logger.info("Starting graceful shutdown...")
        
        # 1. Закрываем сессию бота
        await self.bot.session.close()
        
        # 2. Закрываем Redis
        self.cache_service.close()
        
        # 3. Выгружаем ML модели
        self.speech_service.cleanup()
        
        # 4. Выгружаем эмбеддинги (если доступны через ai_service)
        if hasattr(self.ai_service, 'embedding_service'):
            self.ai_service.embedding_service.unload_model()
        
        # 5. Финальная очистка Torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✓ Graceful shutdown complete.")

    async def start(self):
        self.show_banner()
        logger.info("Bot is ready and polling for messages...")

        logger.info(f"Bot username will be checked via bot.me()...")
        try:
            await self.dp.start_polling(self.bot)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Manual shutdown triggered.")
        except Exception as e:
            logger.error(f"Unexpected error during polling: {e}")
        finally:
            await self.shutdown()