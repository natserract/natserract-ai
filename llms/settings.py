from typing import Optional

from langchain.pydantic_v1 import BaseSettings, SecretStr
from langchain.schema import SystemMessage

from llms.system_message import system_message

import importlib

config = importlib.import_module('config', __name__)


class Settings(BaseSettings):
    VERBOSE: bool = True

    # Models
    OPENAI_API_KEY: Optional[str] = config.Config.OPENAI_API_KEY

    # LLM Settings
    MODEL: str = config.Config.OPENAI_MODEL
    EMBEDDING_MODEL: str = config.Config.EMBEDDING_MODEL
    TEMPERATURE: float = 0.0
    DETAILED_ERROR: bool = True
    SYSTEM_MESSAGE: SystemMessage = system_message
    REQUEST_TIMEOUT: int = 1 * 60
    MAX_ITERATIONS: int = 12
    MAX_RETRY: int = 3


settings = Settings()
