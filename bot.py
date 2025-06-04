import os
import logging
import configparser
from typing import Optional, Dict, Any
import openai
from anthropic import Anthropic
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class LLMProvider:
    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        self.openai_client = openai.OpenAI(api_key=config['OpenAI']['API_KEY'])
        self.anthropic_client = Anthropic(api_key=config['Anthropic']['API_KEY'])
        self.default_model = config['LLM']['DEFAULT_MODEL']
        self.local_api_url = config['LocalLLM']['API_URL']
        self.available_local_models = config['LocalLLM']['AVAILABLE_MODELS'].split(',')

    async def check_text(self, text: str) -> Dict[str, Any]:
        """Проверяет текст на ошибки используя выбранную LLM модель"""
        model = self.default_model
        
        if model.startswith('gpt'):
            return await self._check_with_openai(text, model)
        elif model.startswith('claude'):
            return await self._check_with_anthropic(text, model)
        elif model in self.available_local_models:
            return await self._check_with_local_model(text, model)
        else:
            raise ValueError(f"Unsupported model: {model}")

    async def _check_with_local_model(self, text: str, model: str) -> Dict[str, Any]:
        """Проверка текста с помощью локальной модели"""
        try:
            response = requests.post(
                f"{self.local_api_url}/generate",
                json={
                    "text": text,
                    "model": model,
                    "temperature": float(self.config['LocalLLM'].get('TEMPERATURE', 0.7)),
                    "max_tokens": int(self.config['LocalLLM'].get('MAX_TOKENS', 500))
                }
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling local LLM API: {e}")
            raise

    async def _check_with_openai(self, text: str, model: str) -> Dict[str, Any]:
        """Проверка текста с помощью OpenAI"""
        prompt = f"""Analyze this English sentence for grammar and usage errors. 
        If there are errors, provide:
        1. The corrected version
        2. Explanation of the errors in Russian
        3. Confidence score (0.0 to 1.0)
        
        Format the response as JSON:
        {{
            "has_errors": true/false,
            "corrected_text": "corrected version",
            "explanation": "explanation in Russian",
            "confidence": 0.95
        }}
        
        If there are no errors, set has_errors to false and provide a confidence score.
        
        Text to analyze: {text}"""

        response = await self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(self.config['OpenAI']['TEMPERATURE']),
            max_tokens=int(self.config['OpenAI']['MAX_TOKENS'])
        )
        
        return eval(response.choices[0].message.content)

    async def _check_with_anthropic(self, text: str, model: str) -> Dict[str, Any]:
        """Проверка текста с помощью Anthropic"""
        prompt = f"""Analyze this English sentence for grammar and usage errors. 
        If there are errors, provide:
        1. The corrected version
        2. Explanation of the errors in Russian
        3. Confidence score (0.0 to 1.0)
        
        Format the response as JSON:
        {{
            "has_errors": true/false,
            "corrected_text": "corrected version",
            "explanation": "explanation in Russian",
            "confidence": 0.95
        }}
        
        If there are no errors, set has_errors to false and provide a confidence score.
        
        Text to analyze: {text}"""

        response = await self.anthropic_client.messages.create(
            model=model,
            max_tokens=int(self.config['Anthropic']['MAX_TOKENS']),
            temperature=float(self.config['Anthropic']['TEMPERATURE']),
            messages=[{"role": "user", "content": prompt}]
        )
        
        return eval(response.content[0].text)

class EnglishGrammarBot:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('settings.ini')
        self.llm = LLMProvider(self.config)
        self.min_confidence = float(self.config['Bot']['MIN_CONFIDENCE'])

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        await update.message.reply_text(
            "Привет! Я бот для проверки английской грамматики. "
            "Просто отправьте мне текст на английском, и я проверю его на ошибки."
        )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
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

        text = update.message.text
        if len(text) > int(self.config['Bot']['MAX_MESSAGE_LENGTH']):
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
                return

            if result['has_errors']:
                # Формируем ответ с исправлениями
                response = (
                    f"🔍 Найдены ошибки:\n\n"
                    f"📝 Исправленный вариант:\n{result['corrected_text']}\n\n"
                    f"📚 Объяснение:\n{result['explanation']}"
                )
                await update.message.reply_text(response)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await update.message.reply_text(
                "Извините, произошла ошибка при обработке сообщения. "
                "Пожалуйста, попробуйте позже."
            )

    def run(self):
        """Запуск бота"""
        # Создаем приложение
        application = Application.builder().token(self.config['Telegram']['BOT_TOKEN']).build()

        # Добавляем обработчики
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.process_message))

        # Запускаем бота
        application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    # Загружаем переменные окружения
    load_dotenv()
    
    # Создаем и запускаем бота
    bot = EnglishGrammarBot()
    bot.run() 