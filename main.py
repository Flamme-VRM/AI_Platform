import asyncio
import logging
from dotenv import load_dotenv

from src.bot import AsylBilim


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


async def main():
    load_dotenv(".env")
    bot = AsylBilim()
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())