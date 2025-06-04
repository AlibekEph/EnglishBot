from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ctransformers import AutoModelForCausalLM as CTAutoModelForCausalLM
from llama_cpp import Llama
import os
from typing import Optional, Dict, Any
import json

app = FastAPI()

class TextRequest(BaseModel):
    text: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 500

class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.model_paths = {
            "llama2-7b": "models/llama2-7b-chat.gguf",
            "llama2-13b": "models/llama2-13b-chat.gguf",
            "mistral-7b": "models/mistral-7b-instruct-v0.2.gguf",
            "mistral-7b-instruct": "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        }
        
    def load_model(self, model_name: str):
        """Загрузка модели в зависимости от её типа"""
        if model_name in self.models:
            return

        if model_name.startswith("llama2") or model_name.startswith("mistral"):
            # Используем llama.cpp для GGUF моделей
            model_path = self.model_paths.get(model_name)
            if not model_path or not os.path.exists(model_path):
                raise ValueError(f"Model {model_name} not found at {model_path}")
            
            self.models[model_name] = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=4
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def generate_response(self, model_name: str, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> Dict[str, Any]:
        """Генерация ответа с помощью выбранной модели"""
        if model_name not in self.models:
            self.load_model(model_name)

        model = self.models[model_name]
        
        if model_name.startswith("llama2") or model_name.startswith("mistral"):
            # Форматируем промпт для чат-моделей
            if model_name.startswith("llama2"):
                formatted_prompt = f"[INST] {prompt} [/INST]"
            else:  # mistral
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"

            response = model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "[INST]"],
                echo=False
            )
            
            # Извлекаем только сгенерированный текст
            generated_text = response["choices"][0]["text"].strip()
            
            try:
                # Пытаемся распарсить JSON из ответа
                return json.loads(generated_text)
            except json.JSONDecodeError:
                # Если не получилось распарсить JSON, возвращаем структурированный ответ
                return {
                    "has_errors": True,
                    "corrected_text": generated_text,
                    "explanation": "Не удалось распарсить ответ модели в JSON формат",
                    "confidence": 0.5
                }

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