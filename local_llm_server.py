from fastapi import FastAPI, HTTPException
# Handle compatibility with both Pydantic v1 and v2
try:
    # Try Pydantic v2 imports first
    from pydantic import BaseModel, Field
except ImportError:
    # Fall back to Pydantic v1 imports
    from pydantic import BaseModel, Field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ctransformers import AutoModelForCausalLM as CTAutoModelForCausalLM
from llama_cpp import Llama
import os
from typing import Optional, Dict, Any
import json
import logging

# Configure FastAPI to work with Pydantic v1
app = FastAPI()

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextRequest(BaseModel):
    text: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 500
    
    # Make the model work with both Pydantic v1 and v2
    class Config:
        arbitrary_types_allowed = True

class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.model_paths = {
            "mistral-7b-instruct-v0.2.gguf": "/app/models/mistral-7b-instruct-v0.2.gguf",
            "llama2-7b-chat.gguf": "/app/models/llama2-7b-chat.gguf"
        }
        logger.info(f"Initialized ModelManager with paths: {self.model_paths}")
        
    def load_model(self, model_name: str):
        """Загрузка модели в зависимости от её типа"""
        logger.info(f"Attempting to load model: {model_name}")
        
        if model_name in self.models:
            logger.info(f"Model {model_name} already loaded")
            return

        model_path = self.model_paths.get(model_name)
        logger.info(f"Looking for model at path: {model_path}")
        
        if not model_path:
            available_models = list(self.model_paths.keys())
            logger.error(f"Model {model_name} not found in available models: {available_models}")
            raise ValueError(f"Model {model_name} not found in available models: {available_models}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at path: {model_path}")
            logger.debug(f"Current working directory: {os.getcwd()}")
            logger.debug(f"Directory contents: {os.listdir(os.path.dirname(model_path))}")
            raise ValueError(f"Model file not found at path: {model_path}")
        
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Оптимизированные параметры для Mistral
            if "mistral" in model_name.lower():
                self.models[model_name] = Llama(
                    model_path=model_path,
                    n_ctx=512,          # Уменьшаем контекст для ускорения
                    n_threads=8,        # Увеличиваем количество потоков
                    n_batch=1024,       # Увеличиваем размер батча
                    n_gpu_layers=0,     # Отключаем GPU слои, если нет GPU
                    f16_kv=True,        # Используем float16 для ключей и значений
                    embedding=False,    # Отключаем эмбеддинги
                    vocab_only=False,   # Загружаем полную модель
                    use_mlock=True,     # Блокируем память для модели
                    use_mmap=True,      # Используем memory mapping
                    numa=True,          # Включаем NUMA оптимизации
                    rope_scaling=None,  # Отключаем RoPE scaling
                    offload_kqv=True,   # Включаем offloading для KQV
                    tensor_split=None,  # Отключаем разделение тензоров
                    seed=42,            # Фиксируем seed для воспроизводимости
                    n_keep=0,           # Не сохраняем токены
                    n_draft=0,          # Отключаем draft tokens
                    n_chunks=1,         # Используем один чанк
                    n_parallel=1,       # Отключаем параллельную обработку
                    n_sequences=1,      # Обрабатываем одну последовательность
                    p_split=0.0,        # Отключаем разделение вероятностей
                    main_gpu=0,         # Используем основной GPU
                    tensor_parallel=1,  # Отключаем тензорный параллелизм
                    rope_freq_base=10000,  # Стандартная базовая частота RoPE
                    rope_freq_scale=1.0,   # Стандартный масштаб RoPE
                    yarn_ext_factor=1.0,   # Отключаем YaRN
                    yarn_attn_factor=1.0,  # Отключаем YaRN attention
                    yarn_beta_fast=32.0,   # Стандартный YaRN beta
                    yarn_beta_slow=1.0,    # Стандартный YaRN beta
                    yarn_orig_ctx=2048,    # Стандартный контекст YaRN
                    logits_all=False,      # Отключаем все логиты
                    embedding=False,       # Отключаем эмбеддинги
                    offload_kqv=True,      # Включаем offloading для KQV
                    offload_inp=True,      # Включаем offloading для входных данных
                    offload_out=True,      # Включаем offloading для выходных данных
                    numa=True,             # Включаем NUMA оптимизации
                    numa_strategy=0,       # Стандартная стратегия NUMA
                    numa_max_nodes=1,      # Используем один NUMA узел
                    numa_skip_self=True,   # Пропускаем self в NUMA
                    numa_prefer_node=0,    # Предпочитаем узел 0
                    numa_avoid_node=-1,    # Не избегаем узлов
                    numa_balance_nodes=False,  # Отключаем балансировку узлов
                    numa_balance_size=0,   # Отключаем балансировку по размеру
                    numa_balance_count=0,  # Отключаем балансировку по количеству
                    numa_balance_ratio=0.0 # Отключаем балансировку по соотношению
                )
            else:
                # Стандартные параметры для других моделей
                self.models[model_name] = Llama(
                    model_path=model_path,
                    n_ctx=1024,
                    n_threads=4,
                    n_batch=512,
                    n_gpu_layers=0,
                    f16_kv=True,
                    embedding=False,
                    vocab_only=False,
                    use_mlock=True,
                    use_mmap=True
                )
            
            logger.info(f"Successfully loaded model: {model_name} from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}", exc_info=True)
            raise ValueError(f"Error loading model {model_name}: {str(e)}")

    def generate_response(self, model_name: str, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> Dict[str, Any]:
        """Генерация ответа с помощью выбранной модели"""
        logger.info(f"Generating response for model: {model_name}")
        
        if model_name not in self.models:
            logger.info(f"Model {model_name} not loaded, attempting to load")
            self.load_model(model_name)

        model = self.models[model_name]
        
        # Оптимизированный промпт для Mistral
        if "mistral" in model_name.lower():
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            logger.debug(f"Formatted prompt for Mistral: {formatted_prompt}")
            
            # Оптимизированные параметры генерации для Mistral
            response = model(
                formatted_prompt,
                max_tokens=min(max_tokens, 128),  # Ограничиваем токены
                temperature=temperature,
                top_p=0.95,           # Увеличиваем top_p для более быстрой генерации
                top_k=40,             # Уменьшаем top_k для ускорения
                repeat_penalty=1.1,    # Уменьшаем штраф за повторения
                presence_penalty=0.0,  # Отключаем штраф за присутствие
                frequency_penalty=0.0, # Отключаем штраф за частоту
                mirostat_mode=0,      # Отключаем mirostat
                mirostat_tau=5.0,     # Стандартное tau
                mirostat_eta=0.1,     # Стандартное eta
                stop=["</s>", "[INST]"],  # Добавляем стоп-токены
                echo=False,           # Отключаем эхо
                stream=False,         # Отключаем стриминг
                logits_processor=None # Отключаем обработку логитов
            )
        else:
            # Стандартный промпт для других моделей
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            response = model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

        try:
            # Парсим ответ
            response_text = response['choices'][0]['text'].strip()
            logger.debug(f"Raw response: {response_text}")
            
            # Пытаемся распарсить JSON
            try:
                result = json.loads(response_text)
                logger.info(f"Parsed JSON response: {result}")
                return result
            except json.JSONDecodeError:
                # Если не удалось распарсить JSON, создаем структурированный ответ
                logger.warning(f"Could not parse response as JSON: {response_text}")
                return {
                    "has_errors": "error" in response_text.lower(),
                    "corrected_text": prompt,
                    "explanation": response_text,
                    "confidence": 0.8 if "no error" in response_text.lower() else 0.6
                }
                
        except Exception as e:
            logger.error(f"Error processing model response: {str(e)}", exc_info=True)
            raise ValueError(f"Error processing model response: {str(e)}")

model_manager = ModelManager()

@app.post("/generate")
async def generate_text(request: TextRequest) -> Dict[str, Any]:
    """Эндпоинт для генерации текста"""
    try:
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
        
        Text to analyze: {request.text}"""

        response = model_manager.generate_response(
            request.model,
            prompt,
            request.temperature,
            request.max_tokens
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=1000,  # Увеличиваем время поддержания соединения
        timeout_graceful_shutdown=1000  # Увеличиваем время на graceful shutdown
    ) 