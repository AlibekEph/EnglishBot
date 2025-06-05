import logging
import os
import configparser
import requests
import time
from typing import Dict, Any
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import asyncio
from telegram.error import TimedOut, NetworkError, RetryAfter
import backoff  # Добавим в requirements_bot.txt
import json

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
        self.use_openai = config.getboolean('OpenAI', 'USE_OPENAI', fallback=False)
        
        if self.use_openai:
            # Настройки для OpenAI
            self.openai_api_key = config['OpenAI']['API_KEY']
            self.openai_model = config['OpenAI']['MODEL']
            self.openai_max_tokens = int(config['OpenAI'].get('MAX_TOKENS', 150))
            self.openai_temperature = float(config['OpenAI'].get('TEMPERATURE', 0.7))
            llm_logger.info("Using OpenAI API with model: %s", self.openai_model)
        else:
            # Настройки для локальной модели
            self.default_model = config['LLM']['DEFAULT_MODEL']
            self.available_local_models = config['LocalLLM']['AVAILABLE_MODELS'].split(',')
            
            # Если DEFAULT_MODEL = 'local', используем первую доступную модель
            if self.default_model == 'local':
                self.default_model = self.available_local_models[0]
                llm_logger.info(f"Using first available local model: {self.default_model}")
            elif self.default_model not in self.available_local_models:
                llm_logger.warning(
                    f"Default model {self.default_model} not available locally, "
                    f"using first available model: {self.available_local_models[0]}"
                )
                self.default_model = self.available_local_models[0]
            
            llm_logger.info("Using local model server at: %s with model: %s", 
                          self.local_model_url, self.default_model)
        
        # Настраиваем сессию с повторными попытками
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    async def _check_with_local_model(self, text: str, model: str) -> dict:
        """Проверка текста с помощью локальной модели"""
        if model not in self.available_local_models:
            raise ValueError(f"Model {model} not found in available models: {self.available_local_models}")
            
        max_retries = 3
        retry_delay = 5  # секунд
        
        for attempt in range(max_retries):
            try:
                llm_logger.info(f"Checking text with local model {model} (attempt {attempt + 1}/{max_retries})")
                
                # Ограничиваем длину текста для проверки
                if len(text) > 200:
                    text = text[:200] + "..."
                    llm_logger.warning(f"Text truncated to 200 characters")
                
                response = self.session.post(
                    f"{self.local_model_url}/generate",
                    json={
                        "text": text,
                        "model": model,
                        "max_tokens": min(int(self.config['LLM']['MAX_TOKENS']), 128),
                        "temperature": float(self.config['LLM']['TEMPERATURE'])
                    },
                    timeout=(30, 60)
                )
                response.raise_for_status()
                result = response.json()
                llm_logger.info(f"Local model response: {result}")
                return result
                
            except requests.exceptions.RequestException as e:
                llm_logger.error(f"Error calling local model (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    llm_logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Ошибка при обращении к локальной модели после {max_retries} попыток: {str(e)}")

    async def _check_with_openai(self, text: str) -> dict:
        """Проверка текста с помощью OpenAI API"""
        try:
            llm_logger.info("Checking text with OpenAI API")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            
            # Формируем промпт для проверки грамматики
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful English grammar checker. Check the text for grammar, spelling, and punctuation errors. If you find any errors, provide the corrected text and explain the corrections. If the text is correct, respond with 'No errors found.' Format your response as JSON with the following structure: {\"has_errors\": boolean, \"corrected_text\": string, \"explanation\": string, \"confidence\": float (0-1)}"
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
            
            data = {
                "model": self.openai_model,
                "messages": messages,
                "temperature": self.openai_temperature,
                "max_tokens": self.openai_max_tokens
            }
            
            response = self.session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=(30, 60)
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            try:
                # Пытаемся распарсить JSON из ответа
                parsed_result = json.loads(content)
                llm_logger.info(f"OpenAI API response: {parsed_result}")
                return parsed_result
            except json.JSONDecodeError:
                # Если не удалось распарсить JSON, создаем структурированный ответ
                llm_logger.warning(f"Could not parse OpenAI response as JSON: {content}")
                return {
                    "has_errors": "error" in content.lower(),
                    "corrected_text": text,
                    "explanation": content,
                    "confidence": 0.8 if "no error" in content.lower() else 0.6
                }
                
        except requests.exceptions.RequestException as e:
            llm_logger.error(f"Error calling OpenAI API: {str(e)}", exc_info=True)
            raise Exception(f"Ошибка при обращении к OpenAI API: {str(e)}")

    async def check_text(self, text: str) -> Dict[str, Any]:
        """Проверка текста на грамматические ошибки"""
        try:
            if self.use_openai:
                llm_logger.info("Using OpenAI API for text check")
                return await self._check_with_openai(text)
            else:
                llm_logger.info(f"Using local model {self.default_model} for text check")
                # Разбиваем длинный текст на предложения
                if len(text) > 200:
                    sentences = text.split('.')
                    results = []
                    for sentence in sentences:
                        if sentence.strip():
                            result = await self._check_with_local_model(sentence.strip(), self.default_model)
                            results.append(result)
                    
                    # Объединяем результаты
                    if results:
                        has_errors = any(r.get('has_errors', False) for r in results)
                        best_result = max(results, key=lambda x: x.get('confidence', 0))
                        return {
                            'has_errors': has_errors,
                            'corrected_text': best_result.get('corrected_text', text),
                            'explanation': best_result.get('explanation', ''),
                            'confidence': best_result.get('confidence', 0)
                        }
                
                return await self._check_with_local_model(text, self.default_model)
                
        except Exception as e:
            llm_logger.error(f"Error checking text: {str(e)}", exc_info=True)
            raise

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

    @backoff.on_exception(
        backoff.expo,
        (TimedOut, NetworkError, RetryAfter),
        max_tries=5,
        max_time=300
    )
    async def initialize_bot(self, application: Application):
        """Инициализация бота с повторными попытками"""
        try:
            logger.info("Initializing bot...")
            await application.initialize()
            logger.info("Bot initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing bot: {str(e)}", exc_info=True)
            raise

    def run(self):
        """Запуск бота"""
        logger.info("Starting bot...")
        
        # Создаем приложение с увеличенными таймаутами
        application = (
            Application.builder()
            .token(self.config['Telegram']['BOT_TOKEN'])
            .connect_timeout(30.0)  # Увеличиваем таймаут подключения
            .read_timeout(30.0)     # Увеличиваем таймаут чтения
            .write_timeout(30.0)    # Увеличиваем таймаут записи
            .pool_timeout(30.0)     # Увеличиваем таймаут пула
            .build()
        )

        # Добавляем обработчики
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help))
        application.add_handler(MessageHandler(
            filters.TEXT | filters.ChatType.GROUPS | filters.ChatType.SUPERGROUP,
            self.process_message
        ))

        # Запускаем бота
        try:
            logger.info("Bot is ready to receive messages")
            application.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,  # Игнорируем старые обновления
                pool_timeout=30.0,          # Таймаут для пула
                read_timeout=30.0,          # Таймаут для чтения
                write_timeout=30.0,         # Таймаут для записи
                connect_timeout=30.0        # Таймаут для подключения
            )
        except Exception as e:
            logger.error(f"Error running bot: {str(e)}", exc_info=True)
            # Даем время на корректное завершение
            time.sleep(5)
            # Перезапускаем бота
            self.run()

    @backoff.on_exception(
        backoff.expo,
        (TimedOut, NetworkError, RetryAfter),
        max_tries=3,
        max_time=60
    )
    async def process_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик входящих сообщений с повторными попытками"""
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

if __name__ == '__main__':
    # Загружаем переменные окружения
    load_dotenv()
    
    # Создаем и запускаем бота
    bot_instance = EnglishGrammarBot()
    bot_instance.run() 