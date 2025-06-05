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
            self.models[model_name] = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=4
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
        
        # Форматируем промпт для Mistral
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        logger.debug(f"Formatted prompt: {formatted_prompt}")

        try:
            logger.info("Generating response from model")
            response = model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "[INST]"],
                echo=False
            )
            
            # Извлекаем только сгенерированный текст
            generated_text = response["choices"][0]["text"].strip()
            logger.debug(f"Generated text: {generated_text}")
            
            try:
                # Пытаемся распарсить JSON из ответа
                result = json.loads(generated_text)
                logger.info("Successfully parsed JSON response")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {str(e)}")
                # Если не получилось распарсить JSON, возвращаем структурированный ответ
                return {
                    "has_errors": True,
                    "corrected_text": generated_text,
                    "explanation": "Не удалось распарсить ответ модели в JSON формат",
                    "confidence": 0.5
                }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise ValueError(f"Error generating response: {str(e)}")

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
    uvicorn.run(app, host="0.0.0.0", port=8000) 