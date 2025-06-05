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

    async def check_text(self, text: str) -> Dict[str, Any]:
        """Проверка текста на грамматические ошибки"""
        try:
            # Проверяем, какую модель использовать
            if not self.use_openai:
                llm_logger.info(f"Using local model {self.default_model} for text check")
                return await self._check_with_local_model(text)
            
            # Если USE_OPENAI = true, используем OpenAI API
            llm_logger.info(f"Using OpenAI API with model {self.openai_model} for text check")
            return await self._check_with_openai(text)
                
        except Exception as e:
            llm_logger.error(f"Error checking text: {str(e)}", exc_info=True)
            raise

    async def _check_with_local_model(self, text: str) -> Dict[str, Any]:
        """Проверка текста с помощью локальной модели"""
        if self.use_openai:
            raise ValueError("Cannot use local model when USE_OPENAI is true")
            
        try:
            llm_logger.info(f"Checking text with local model: {self.default_model}")
            
            # Проверяем, что модель доступна
            if self.default_model not in self.available_local_models:
                raise ValueError(f"Model {self.default_model} not found in available models: {self.available_local_models}")
            
            # Формируем промпт для проверки грамматики
            prompt = (
                "Check the following text for grammar, spelling, and punctuation errors. "
                "If you find any errors, provide the corrected text and explain the corrections. "
                "If the text is correct, respond with 'No errors found.' "
                "Format your response as JSON with the following structure: "
                '{"has_errors": boolean, "corrected_text": string, "explanation": string, "confidence": float (0-1)}\n\n'
                f"Text to check: {text}"
            )
            
            # Отправляем запрос к локальному API
            request_data = {
                "model": self.default_model,
                "text": prompt,
                "temperature": 0.7,
                "max_tokens": 256
            }
            llm_logger.debug(f"Sending request to local API: {self.local_model_url}/generate with data: {request_data}")
            response = self.session.post(
                f"{self.local_model_url}/generate",
                json=request_data,
                timeout=(30, 60)
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['text'].strip()
            
            try:
                # Пытаемся распарсить JSON из ответа
                parsed_result = json.loads(content)
                llm_logger.info(f"Local model response: {parsed_result}")
                return parsed_result
            except json.JSONDecodeError:
                # Если не удалось распарсить JSON, создаем структурированный ответ
                llm_logger.warning(f"Could not parse local model response as JSON: {content}")
                return {
                    "has_errors": "error" in content.lower(),
                    "corrected_text": text,
                    "explanation": content,
                    "confidence": 0.8 if "no error" in content.lower() else 0.6
                }
                
        except requests.exceptions.RequestException as e:
            llm_logger.error(f"Error calling local model API: {str(e)}", exc_info=True)
            raise Exception(f"Ошибка при обращении к локальной модели: {str(e)}")

    async def _check_with_openai(self, text: str) -> Dict[str, Any]:
        """Проверка текста с помощью OpenAI API"""
        if not self.use_openai:
            raise ValueError("Cannot use OpenAI API when USE_OPENAI is false")
            
        try:
            llm_logger.info(f"Checking text with OpenAI API using model: {self.openai_model}")
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

class EnglishGrammarBot:
    def __init__(self):
        """Инициализация бота"""
        try:
            # Загружаем конфигурацию
            self.config = configparser.ConfigParser()
            config_path = os.path.join(os.path.dirname(__file__), 'settings.ini')
            llm_logger.info(f"Loading configuration from: {config_path}")
            
            if not self.config.read(config_path):
                raise ValueError(f"Could not read configuration file: {config_path}")
            
            # Проверяем наличие всех необходимых секций
            required_sections = ['Telegram', 'LLM', 'OpenAI', 'LocalLLM', 'Bot']
            for section in required_sections:
                if section not in self.config:
                    raise ValueError(f"Missing required section in settings.ini: {section}")
            
            # Инициализируем провайдер LLM
            self.llm_provider = LLMProvider(self.config)
            
            # Настройки бота
            self.token = self.config['Telegram']['BOT_TOKEN']
            self.min_confidence = float(self.config['Bot']['MIN_CONFIDENCE'])
            self.max_message_length = int(self.config['Bot']['MAX_MESSAGE_LENGTH'])
            
            llm_logger.info("Bot initialized successfully with settings from settings.ini")
            llm_logger.info(f"USE_OPENAI setting: {self.llm_provider.use_openai}")
            
        except Exception as e:
            llm_logger.error(f"Error initializing bot: {str(e)}", exc_info=True)
            raise

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

    async def process_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработка входящих сообщений"""
        try:
            # Получаем информацию о сообщении
            message = update.message
            if not message or not message.text:
                return

            # Получаем информацию о чате
            chat = message.chat
            chat_type = chat.type
            chat_id = chat.id
            user_id = message.from_user.id
            username = message.from_user.username

            # Логируем информацию о сообщении
            telegram_logger.info(
                f"Received message in {chat_type} chat {chat_id} "
                f"from user {user_id} (@{username}): {message.text[:50]}..."
            )

            # Игнорируем команды
            if message.text.startswith('/'):
                return

            # Проверяем длину сообщения
            if len(message.text) > self.max_message_length:
                await message.reply_text(
                    f"Сообщение слишком длинное. Максимальная длина: {self.max_message_length} символов."
                )
                return

            # Проверяем текст с помощью выбранной модели
            try:
                result = await self.llm_provider.check_text(message.text)
                
                if result.get('has_errors', False):
                    # Формируем ответ с исправлениями
                    corrected_text = result.get('corrected_text', '')
                    explanation = result.get('explanation', '')
                    confidence = result.get('confidence', 0.0)
                    
                    # Проверяем уверенность модели
                    if confidence >= self.min_confidence:
                        # Формируем сообщение с исправлениями
                        response = f"💡 Исправленный текст:\n{corrected_text}\n\n"
                        if explanation:
                            response += f"📝 Объяснение:\n{explanation}"
                        
                        # Отправляем ответ
                        await message.reply_text(response)
                        llm_logger.info(
                            f"Sent correction in {chat_type} chat {chat_id} "
                            f"with confidence {confidence:.2f}"
                        )
                    else:
                        llm_logger.debug(
                            f"Confidence {confidence:.2f} below threshold {self.min_confidence}, "
                            f"ignoring correction"
                        )
                else:
                    llm_logger.debug("No errors found in the text")
                    
            except Exception as e:
                llm_logger.error(f"Error checking text: {str(e)}", exc_info=True)
                await message.reply_text(
                    "Извините, произошла ошибка при проверке текста. Попробуйте позже."
                )

        except Exception as e:
            llm_logger.error(f"Error processing message: {str(e)}", exc_info=True)
            if message:
                await message.reply_text(
                    "Извините, произошла ошибка при обработке сообщения. Попробуйте позже."
                )

    def run(self):
        """Запуск бота"""
        try:
            llm_logger.info("Starting bot...")
            
            # Создаем приложение
            application = Application.builder().token(self.config['Telegram']['BOT_TOKEN']).build()
            
            # Добавляем обработчики
            application.add_handler(MessageHandler(
                filters.TEXT & ~filters.COMMAND,  # Обрабатываем все текстовые сообщения, кроме команд
                self.process_message
            ))
            
            # Запускаем бота
            llm_logger.info("Bot started successfully")
            application.run_polling(
                allowed_updates=Update.ALL_TYPES,  # Получаем все типы обновлений
                drop_pending_updates=True,        # Игнорируем накопившиеся обновления при старте
                connect_timeout=30,               # Таймаут подключения
                read_timeout=30,                  # Таймаут чтения
                write_timeout=30,                 # Таймаут записи
                pool_timeout=30                   # Таймаут пула соединений
            )
            
        except Exception as e:
            llm_logger.error(f"Error running bot: {str(e)}", exc_info=True)
            raise

if __name__ == '__main__':
    # Загружаем переменные окружения
    load_dotenv()
    
    # Создаем и запускаем бота
    bot_instance = EnglishGrammarBot()
    bot_instance.run() 