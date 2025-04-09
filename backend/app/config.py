import os
from pydantic_settings import BaseSettings
import logging


class Settings(BaseSettings):
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@db:5432/chatdb"
    )
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    openai_base_url: str = os.getenv(
        "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model_name: str = os.getenv("MODEL_NAME", "qwen-plus")
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))

    api_title: str = "Rose Agent"
    api_version: str = "0.1.0"

    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    class Config:
        env_file = ".env"


settings = Settings()

log_level_int = getattr(logging, settings.log_level, logging.INFO)
settings.log_level = logging.getLevelName(log_level_int)
