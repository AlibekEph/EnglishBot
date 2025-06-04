# English Grammar Checker Bot

Telegram бот для проверки английской грамматики с использованием передовых LLM моделей (OpenAI GPT, Anthropic Claude, Llama 2, Mistral).

## Возможности

- Проверка английских предложений на грамматические ошибки
- Автоматическое исправление найденных ошибок
- Подробные объяснения ошибок на русском языке
- Поддержка различных LLM моделей:
  - Облачные модели:
    - OpenAI GPT (GPT-4, GPT-3.5-turbo)
    - Anthropic Claude (Claude-3-opus, Claude-3-sonnet)
  - Локальные модели:
    - Llama 2 (7B, 13B)
    - Mistral (7B, 7B-instruct)
- Настраиваемые параметры через конфигурационный файл

## Установка

### Вариант 1: Установка через Docker (рекомендуется)

1. Установите Docker и Docker Compose:
   - [Docker](https://docs.docker.com/get-docker/)
   - [Docker Compose](https://docs.docker.com/compose/install/)

2. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd <repository-name>
```

3. Создайте файл `.env` и добавьте необходимые API ключи:
```
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

4. Скачайте модели в директорию `models/`:
```bash
mkdir -p models
cd models

# Llama 2 7B Chat
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf -O llama2-7b-chat.gguf

# Mistral 7B Instruct
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O mistral-7b-instruct-v0.2.gguf
```

5. Настройте `settings.ini`:
```ini
[LLM]
DEFAULT_MODEL = llama2-7b  # или mistral-7b-instruct

[LocalLLM]
API_URL = http://llm-server:8000  # Важно: используйте имя сервиса из docker-compose
```

6. Запустите через Docker Compose:
```bash
docker-compose up -d
```

Для просмотра логов:
```bash
# Логи всех сервисов
docker-compose logs -f

# Логи только бота
docker-compose logs -f telegram-bot

# Логи только LLM сервера
docker-compose logs -f llm-server
```

Для остановки:
```bash
docker-compose down
```

### Вариант 2: Установка без Docker

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Создайте файл `.env` и добавьте в него необходимые API ключи:
```
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

3. Настройте `settings.ini`:
- Добавьте токен вашего Telegram бота
- Выберите предпочитаемую LLM модель
- Настройте параметры моделей (температура, максимальное количество токенов)
- Настройте параметры бота (минимальная уверенность, язык объяснений)

## Установка локальных моделей

### Подготовка окружения

1. Убедитесь, что у вас установлен Python 3.8+ и pip

2. Для работы с локальными моделями рекомендуется использовать виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # для Linux/Mac
# или
venv\Scripts\activate  # для Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

### Установка моделей

1. Создайте директорию для моделей:
```bash
mkdir models
cd models
```

2. Скачайте нужные модели:

#### Llama 2
```bash
# Скачайте GGUF версии моделей с Hugging Face:
# Llama 2 7B Chat
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf -O llama2-7b-chat.gguf

# Llama 2 13B Chat
wget https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf -O llama2-13b-chat.gguf
```

#### Mistral
```bash
# Mistral 7B Instruct
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O mistral-7b-instruct-v0.2.gguf
```

### Запуск локального API сервера

1. Запустите локальный API сервер:
```bash
python local_llm_server.py
```

Сервер будет доступен по адресу `http://localhost:8000`

### Требования к системе

### Для Docker установки:
- Docker и Docker Compose
- NVIDIA Container Toolkit (для GPU поддержки)
- Минимум 20GB свободного места на диске
- Рекомендуется:
  - CPU: 8+ ядер
  - RAM: 16+ GB
  - GPU: NVIDIA GPU с 8+ GB VRAM

### Для установки без Docker:
- Python 3.8+
- Telegram Bot Token
- OpenAI API Key (для GPT моделей)
- Anthropic API Key (для Claude моделей)
- Достаточно мощное железо для локальных моделей

## Настройка

Файл `settings.ini` позволяет настроить:

- Выбор LLM модели
- Параметры моделей (температура, токены)
- Параметры бота (уверенность, язык)
- Ограничения на длину сообщений
- Параметры локального API сервера

### Примеры конфигурации для разных моделей

1. Использование Llama 2 7B:
```ini
[LLM]
DEFAULT_MODEL = llama2-7b

[LocalLLM]
API_URL = http://localhost:8000
AVAILABLE_MODELS = llama2-7b
```

2. Использование Mistral 7B:
```ini
[LLM]
DEFAULT_MODEL = mistral-7b-instruct

[LocalLLM]
API_URL = http://localhost:8000
AVAILABLE_MODELS = mistral-7b-instruct
```

3. Использование GPT-3.5:
```ini
[LLM]
DEFAULT_MODEL = gpt-3.5-turbo

[OpenAI]
API_KEY = your_openai_api_key_here
```

## Настройка Docker

### Использование GPU

Для использования GPU в Docker:

1. Установите NVIDIA Container Toolkit:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. Проверьте установку:
```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Настройка ресурсов

Вы можете настроить ресурсы для контейнеров в `docker-compose.yml`:

```yaml
services:
  llm-server:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Персистентность данных

- Модели хранятся в директории `models/` на хост-машине
- Конфигурация в `settings.ini` монтируется в контейнеры
- Переменные окружения в `.env` монтируются в контейнер бота

## Обновление

Для обновления бота:

1. Остановите контейнеры:
```bash
docker-compose down
```

2. Получите последние изменения:
```bash
git pull
```

3. Пересоберите и запустите контейнеры:
```bash
docker-compose up -d --build
```

## Устранение неполадок

### Проблемы с GPU

Если GPU не определяется в контейнере:

1. Проверьте установку NVIDIA Container Toolkit:
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

2. Убедитесь, что в `docker-compose.yml` правильно настроены GPU ресурсы

### Проблемы с памятью

Если контейнер падает из-за нехватки памяти:

1. Уменьшите размер модели (используйте Q4_K_M версии)
2. Настройте лимиты памяти в `docker-compose.yml`
3. Увеличьте swap-файл на хост-машине

### Проблемы с сетью

Если бот не может подключиться к LLM серверу:

1. Проверьте настройки в `settings.ini`:
```ini
[LocalLLM]
API_URL = http://llm-server:8000  # Должно совпадать с именем сервиса в docker-compose
```

2. Проверьте логи:
```bash
docker-compose logs -f
```

## Лицензия

MIT 