import os
from pydantic_settings import BaseSettings
import logging
import json


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

    mcp_servers_config: str = os.getenv("MCP_SERVERS_CONFIG", "")

    @property
    def mcp_servers(self) -> dict:
        try:
            if self.mcp_servers_config and self.mcp_servers_config.strip():
                return json.loads(self.mcp_servers_config)
            else:
                logging.warning(
                    "MCP_SERVERS_CONFIG is empty or not set. No MCP servers will be loaded."
                )
                return {}
        except json.JSONDecodeError as e:
            problematic_part = self.mcp_servers_config[max(0, e.pos - 10) : e.pos + 10]
            logging.error(
                f"Failed to parse MCP_SERVERS_CONFIG JSON: {e}. Near: '...{problematic_part}...'",
                exc_info=False,
            )
            return {}

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


settings = Settings()

log_level_int = getattr(logging, settings.log_level, logging.INFO)
logging.getLogger().setLevel(log_level_int)
settings.log_level = logging.getLevelName(log_level_int)

sensitive_keys = {"openai_api_key"}
safe_settings = {
    k: v for k, v in settings.model_dump().items() if k not in sensitive_keys
}
logging.debug(f"Loaded application settings: {safe_settings}")
logging.debug(
    f"Raw MCP_SERVERS_CONFIG: {'<empty>' if not settings.mcp_servers_config else settings.mcp_servers_config[:100] + '...'}"
)
logging.debug(f"Parsed MCP Servers config: {settings.mcp_servers}")
