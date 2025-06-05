#!/usr/bin/env python

import bot

# Загружаем переменные окружения
from dotenv import load_dotenv
load_dotenv()

# Создаем и запускаем бота
bot_instance = bot.EnglishGrammarBot()
bot_instance.run() 