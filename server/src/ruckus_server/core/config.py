"""Server configuration."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Server settings."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    workers: int = 4

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/ruckus.db"
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # Scheduler
    max_concurrent_jobs: int = 10
    job_timeout_default: int = 3600
    scheduler_interval: int = 5

    # Agent Management
    agent_timeout: int = 60
    heartbeat_interval: int = 30
    max_agents: int = 100

    # API
    api_prefix: str = "/api/v1"
    cors_origins: list = ["*"]

    # Authentication (future)
    auth_enabled: bool = False
    api_key: Optional[str] = None

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    class Config:
        env_prefix = "RUCKUS_SERVER_"
        env_file = ".env"


settings = Settings()