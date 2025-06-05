import logging
import os
import configparser
import requests
from typing import Dict, Any
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)

logger = logging.getLogger(__name__)
llm_logger = logging.getLogger('llm')
telegram_logger = logging.getLogger('telegram')

class LLMProvider:
    def __init__(self, config):
        self.config = config
        self.local_model_url = "http://llm-server:8000"
        self.min_confidence = float(config['LLM']['MIN_CONFIDENCE'])
        self.default_model = config['LLM']['DEFAULT_MODEL']
        self.available_local_models = config['LocalLLM']['AVAILABLE_MODELS'].split(',')
        llm_logger.info("LLM Provider initialized with local model URL: %s", self.local_model_url)

    async def check_text(self, text: str) -> Dict[str, Any]:
        """Проверка текста на грамматические ошибки"""
        try:
            # Используем только локальную модель
            model = self.default_model
            if model not in self.available_local_models:
                llm_logger.warning(f"Model {model} not available, using first available model")
                model = self.available_local_models[0]
            
            result = await self._check_with_local_model(text, model)
            return result
        except Exception as e:
            llm_logger.error(f"Error checking text: {str(e)}", exc_info=True)
            raise

    async def _check_with_local_model(self, text: str, model: str) -> dict:
        """Проверка текста с помощью локальной модели"""
        try:
            llm_logger.info(f"Checking text with local model {model}")
            response = requests.post(
                f"{self.local_model_url}/generate",
                json={
                    "text": text,
                    "model": model,
                    "max_tokens": int(self.config['LLM']['MAX_TOKENS']),
                    "temperature": float(self.config['LLM']['TEMPERATURE'])
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            llm_logger.info(f"Local model response: {result}")
            return result
        except requests.exceptions.RequestException as e:
            llm_logger.error(f"Error calling local model: {str(e)}", exc_info=True)
            raise Exception(f"Ошибка при обращении к локальной модели: {str(e)}")

class EnglishGrammarBot:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('settings.ini')
        self.llm = LLMProvider(self.config)
        self.min_confidence = float(self.config['Bot']['MIN_CONFIDENCE'])
        logger.info("Bot initialized with settings from settings.ini")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        user = update.effective_user
        telegram_logger.info(f"User {user.id} ({user.username}) started the bot")
        await update.message.reply_text(
            "Привет! Я бот для проверки английской грамматики. "
            "Просто отправьте мне текст на английском, и я проверю его на ошибки."
        )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        user = update.effective_user
        telegram_logger.info(f"User {user.id} ({user.username}) requested help")
        help_text = """
        Я могу помочь вам проверить английский текст на грамматические ошибки.
        
        Просто отправьте мне текст на английском, и я:
        1. Проверю его на ошибки
        2. Если найду ошибки, предложу исправленную версию
        3. Объясню ошибки на русском языке
        
        Команды:
        /start - Начать работу с ботом
        /help - Показать это сообщение
        """
        await update.message.reply_text(help_text)

    async def process_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик входящих сообщений"""
        if not update.message or not update.message.text:
            return

        user = update.effective_user
        chat = update.effective_chat
        text = update.message.text

        # Логируем информацию о чате и пользователе
        chat_type = "private" if chat.type == "private" else f"{chat.type} ({chat.title})"
        telegram_logger.info(
            f"Message in {chat_type} from {user.id} ({user.username}): {text}"
        )

        # Проверяем, является ли сообщение командой
        if text.startswith('/'):
            telegram_logger.debug(f"Ignoring command message: {text}")
            return

        if len(text) > int(self.config['Bot']['MAX_MESSAGE_LENGTH']):
            telegram_logger.warning(
                f"Message too long from user {user.id} in {chat_type}: {len(text)} chars"
            )
            if chat.type == "private":  # Отвечаем только в личных сообщениях
                await update.message.reply_text(
                    f"Извините, сообщение слишком длинное. "
                    f"Максимальная длина: {self.config['Bot']['MAX_MESSAGE_LENGTH']} символов."
                )
            return

        try:
            # Проверяем текст с помощью LLM
            result = await self.llm.check_text(text)
            
            # Если уверенность ниже минимальной, игнорируем результат
            if result['confidence'] < self.min_confidence:
                telegram_logger.warning(
                    f"Low confidence result ({result['confidence']}) for user {user.id} in {chat_type}"
                )
                return

            if result['has_errors']:
                # Формируем ответ с исправлениями
                response = (
                    f"🔍 Найдены ошибки:\n\n"
                    f"📝 Исправленный вариант:\n{result['corrected_text']}\n\n"
                    f"📚 Объяснение:\n{result['explanation']}"
                )
                telegram_logger.info(
                    f"Sending correction to user {user.id} in {chat_type}"
                )
                
                # В групповых чатах отвечаем на сообщение
                if chat.type != "private":
                    await update.message.reply_text(
                        response,
                        reply_to_message_id=update.message.message_id
                    )
                else:
                    await update.message.reply_text(response)
            else:
                telegram_logger.info(
                    f"No errors found for user {user.id} in {chat_type}"
                )
            
        except Exception as e:
            logger.error(
                f"Error processing message from user {user.id} in {chat_type}: {str(e)}",
                exc_info=True
            )
            if chat.type == "private":  # Отвечаем только в личных сообщениях
                await update.message.reply_text(
                    "Извините, произошла ошибка при обработке сообщения. "
                    "Пожалуйста, попробуйте позже."
                )

    def run(self):
        """Запуск бота"""
        logger.info("Starting bot...")
        # Создаем приложение
        application = Application.builder().token(self.config['Telegram']['BOT_TOKEN']).build()

        # Добавляем обработчики
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help))
        
        # Обработчик для всех текстовых сообщений, включая команды
        application.add_handler(MessageHandler(
            filters.TEXT | filters.ChatType.GROUPS | filters.ChatType.SUPERGROUP,
            self.process_message
        ))

        # Запускаем бота
        logger.info("Bot is ready to receive messages")
        application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    # Загружаем переменные окружения
    load_dotenv()
    
    # Создаем и запускаем бота
    bot = EnglishGrammarBot()
    bot.run() 