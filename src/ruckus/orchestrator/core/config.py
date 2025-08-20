"""
Configuration settings for the Ruckus Orchestrator.

This module defines Pydantic settings classes that handle configuration
from environment variables with appropriate defaults.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    All settings can be overridden via environment variables with
    the ORCHESTRATOR_ prefix.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ORCHESTRATOR_",
        case_sensitive=False
    )

    # Server settings
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # API settings
    API_V1_STR: str = Field(default="/api/v1", description="API v1 prefix")
    SECRET_KEY: str = Field(description="Secret key for JWT tokens")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Token expiration")
    ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    
    # Database settings
    DATABASE_URL: str = Field(default="sqlite:///./ruckus.db", description="Database URL")
    DATABASE_ECHO: bool = Field(default=False, description="SQLAlchemy echo")
    
    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(default="json", description="Log format")


settings = Settings()
