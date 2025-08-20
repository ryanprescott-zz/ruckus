"""
Configuration settings for the Ruckus Agent.

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
    the AGENT_ prefix.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AGENT_",
        case_sensitive=False
    )

    # Server settings
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8001, description="Server port")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # API settings
    API_V1_STR: str = Field(default="/api/v1", description="API v1 prefix")
    
    # Orchestrator settings
    ORCHESTRATOR_URL: str = Field(
        default="http://localhost:8000",
        description="Orchestrator service URL"
    )
    
    # Agent settings
    AGENT_NAME: str = Field(default="ruckus-agent", description="Agent name")
    HEARTBEAT_INTERVAL: int = Field(default=30, description="Heartbeat interval in seconds")
    
    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(default="json", description="Log format")
    
    # Runtime settings
    MAX_CONCURRENT_JOBS: int = Field(default=1, description="Maximum concurrent jobs")
    JOB_TIMEOUT: int = Field(default=3600, description="Job timeout in seconds")


settings = Settings()
