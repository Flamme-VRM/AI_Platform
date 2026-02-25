import re
import tempfile
import logging
import os
from aiogram.types import Message, Voice
from ..services.ai import AIService
from ..services.cache import CacheService
from ..services.speech_to_text import SpeechToTextService
from ..config import settings

logger = logging.getLogger(__name__)


class MessageHandler:
    def __init__(self, ai_service: AIService, cache_service: CacheService, speech_service: SpeechToTextService):
        self.ai_service = ai_service
        self.cache = cache_service
        self.speech_service = speech_service
        self.admin_user_ids = settings.parsed_admin_ids
        logger.info(f"Загружено ID администраторов: {len(self.admin_user_ids)}")
        
    def is_admin(self, user_id: int) -> bool:
        """Проверка, является ли пользователь администратором."""
        return user_id in self.admin_user_ids

    def convert_markdown_to_html(self, text: str) -> str:
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
        return text

    async def handle_start(self, message: Message):
        user_id = message.from_user.id
        session_data = {
            "first_name": message.from_user.first_name,
            "username": message.from_user.username,
            "started_at": str(message.date),
            "language": "kk"
        }
        self.cache.set_user_session(user_id, session_data)

        greeting = """🇰🇿 Сәлеметсіз бе! AsylBILIM'ге қош келдіңіз!

Мен қазақстандық студенттерге арналған ИИ көмекшісімін:

📚 ЕНТ дайындығы
✍️ Академиялық жазу көмегі  
📖 Оқу материалдарын түсіндіру
🎯 Емтихан дайындығы

Сұрағыңызды жазыңыз - барлық жауаптар қазақ тілінде! 💫"""

        await message.answer(greeting, parse_mode='HTML')

    async def handle_help(self, message: Message):
        user_id = message.from_user.id

        help_message = """
ℹ️ AsylBILIM Көмек

🔧 Қолжетімді командалар:
/start - Ботты іске қосу
/help - Бұл көмек хабарламасы
/status - Хабарлама лимитін тексеру
/clear - Әңгіме тарихын тазалау

📚 Мен не істей аламын:
• ЕНТ дайындығы
• Академиялық жазу көмегі
• Оқу материалдарын түсіндіру
• Емтихан дайындығы

🎵 Дауыстық хабарламаларды түсінемін (қазақ тілінде)

💬 Лимит: 15 хабарлама/24 сағат
"""
        try:
            await message.answer(help_message, parse_mode='HTML')
            logging.info(f"Help menu was called for: {user_id}")
        except Exception as e:
            await message.answer("Қазіргі таңда, техникалық үзілістер болып жатыр.")

    async def handle_clear(self, message: Message):
        user_id = message.from_user.id

        try:
            history_key = f"user_history:{user_id}"
            self.cache.client.delete(history_key)
            await message.answer(
                "Мәтін сәтті тазартылды!\n"
                "Енді біз жаңа әңгімені таза парақтан бастай аламыз.",
                parse_mode="HTML"
            )
            logger.info(f"Redis DB was cleared for {user_id}")
        except Exception as e:
            logger.error(f"Error with clearing data for {user_id}: {e} ")
            await message.answer("Мәтінді қазір тазалау мүмкін емес, сәл кейінірек көріңізді өтінеміз")

    async def handle_voice(self, message: Message):
        try:
            user_id = message.from_user.id
            
            # Check rate limit (skip for admin)
            if not self.is_admin(user_id):
                is_allowed, current_count, time_until_reset = self.cache.check_rate_limit(user_id)
                
                if not is_allowed:
                    hours_left = time_until_reset // 3600
                    minutes_left = (time_until_reset % 3600) // 60
                    
                    if hours_left > 0:
                        time_msg = f"{hours_left} сағат {minutes_left} минут"
                    else:
                        time_msg = f"{minutes_left} минут"
                    
                    rate_limit_msg = (
                        f"🚫 Күнделікті лимит аяқталды (15 хабарлама/24 сағат).\n\n"
                        f"Қалған уақыт: {time_msg}\n"
                        f"Қазіргі саны: {current_count}/15"
                    )
                    await message.answer(rate_limit_msg)
                    logger.info(f"Rate limit exceeded for user {user_id}: {current_count}/15 messages")
                    return

            voice: Voice = message.voice
            if voice.file_size > 10 * 1024 * 1024:
                await message.answer(
                    "🚫 Аудио файл тым үлкен (10МБ-тан асып кетті)."
                    "Қысқарақ аудио жіберіңіз."
                )
                return

            processing_msg = await message.answer("🎵 Аудионы өңдеп жатырмын...")
            file_info = await message.bot.get_file(voice.file_id)

            with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as temp_file:
                temp_path = temp_file.name

            await message.bot.download_file(file_info.file_path, temp_path)

            user_session = self.cache.get_user_session(message.from_user.id)
            language = user_session.get('language', 'kk-KZ')

            recognized_text = await self.speech_service.convert_voice_to_text(
                temp_path,
                language
            )

            if not recognized_text:
                await processing_msg.edit_text(
                    "😕 Аудионы танымады. Анығырақ сөйлеп, үндірек жіберіңіз."
                )
                return

            await processing_msg.delete()

            ai_response = await self.ai_service.generate_response(
                user_id,
                recognized_text
            )

            # Check if message is too long (Telegram limit is 4096 characters)
            if len(ai_response) > 4096:
                await message.answer("Хабарлама тым ұзын. Қысқа сұрақ қойыңыз.")
                logger.warning(f"Voice response too long for user {user_id}: {len(ai_response)} characters")
                return

            try:
                await message.answer(ai_response, parse_mode='HTML')
            except Exception:
                await message.answer(ai_response)

        except Exception as e:
            logger.error(f"Voice processing error for user {user_id}: {e}")
            await message.answer(
                "Кешіріңіз, аудионы өңдеуде қате орын алды. "
                "Қайталап көріңіз немесе мәтін түрінде жазыңыз."
            )

    async def handle_message(self, message: Message):
        if message.text.startswith("/"):
            # Handle rate limit check command
            if message.text.lower() == "/status":
                user_id = message.from_user.id
                rate_info = self.cache.get_rate_limit_info(user_id)
                
                if self.is_admin(user_id):
                    status_msg = (
                        f"👑 Админ режимі: Шектеусіз\n\n"
                        f"Сіз барлық лимиттерден босатылғансыз."
                    )
                else:
                    status_msg = (
                        f"📊 Хабарлама лимитінің мәртебесі:\n\n"
                        f"Қолданылған: {rate_info['count']}/{rate_info['limit']}\n"
                        f"Қалған: {rate_info['remaining']}\n\n"
                        f"Лимит 24 сағат сайын жаңартылады."
                    )
                await message.answer(status_msg)
            return

        MAX_MESSAGE_LENGTH = 4000
        if not message.text or len(message.text) > MAX_MESSAGE_LENGTH:
            await message.answer("⚠️ Хабарлама тым ұзын немесе бос")
            return

        # Sanitize input
        text = message.text.strip()
        text = re.sub(r'[^\w\s\-.,!?қғәіңөұүһӘҒҚҢӨҰҮІ]', '', text, flags=re.UNICODE)
        
        user_id = message.from_user.id
        
        # Check rate limit (skip for admin)
        if not self.is_admin(user_id):
            is_allowed, current_count, time_until_reset = self.cache.check_rate_limit(user_id)
            
            if not is_allowed:
                hours_left = time_until_reset // 3600
                minutes_left = (time_until_reset % 3600) // 60
                
                if hours_left > 0:
                    time_msg = f"{hours_left} сағат {minutes_left} минут"
                else:
                    time_msg = f"{minutes_left} минут"
                
                rate_limit_msg = (
                    f"🚫 Күнделікті лимит аяқталды (15 хабарлама/24 сағат).\n\n"
                    f"Қалған уақыт: {time_msg}\n"
                    f"Қазіргі саны: {current_count}/15"
                )
                await message.answer(rate_limit_msg)
                logger.info(f"Rate limit exceeded for user {user_id}: {current_count}/15 messages")
                return

        ai_response = await self.ai_service.generate_response(user_id, text)

        # Check if message is too long (Telegram limit is 4096 characters)
        if len(ai_response) > 4096:
            await message.answer("Хабарлама тым ұзын. Қысқа сұрақ қойыңыз.")
            logger.warning(f"Message too long for user {user_id}: {len(ai_response)} characters")
            return

        try:
            html_response = self.convert_markdown_to_html(ai_response)
            await message.answer(html_response, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Error sending HTML response to user {user_id}: {e}")
            try:
                await message.answer(ai_response)
            except Exception as e2:
                logger.error(f"Failed to send plain text response to user {user_id}: {e2}")
                # Check if the error is specifically about message length
                if "message is too long" in str(e2).lower():
                    await message.answer("Хабарлама тым ұзын. Қысқа сұрақ қойыңыз.")
                else:
                    await message.answer("Кешіріңіз, жауап жіберуде қате орын алды.")