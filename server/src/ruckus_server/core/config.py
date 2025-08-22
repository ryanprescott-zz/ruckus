"""Server configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
from enum import Enum


class StorageBackendType(str, Enum):
    """Supported storage backend types."""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"


class ServerSettings(BaseSettings):
    """Server configuration settings."""
    
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    workers: int = Field(default=4, description="Number of worker processes")
    
    # API configuration
    api_prefix: str = Field(default="/api/v1", description="API prefix path")
    openapi_prefix: str = Field(default="/api/static", description="OpenAPI static files prefix")
    cors_origins: list = Field(default=["*"], description="CORS allowed origins")
    
    # Authentication (future)
    auth_enabled: bool = Field(default=False, description="Enable authentication")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    
    class Config:
        env_prefix = "RUCKUS_SERVER_"
        env_file = ".env"
        case_sensitive = False


class PostgreSQLSettings(BaseSettings):
    """PostgreSQL storage backend settings."""
    
    database_url: str = Field(
        default="postgresql+asyncpg://ruckus:ruckus@localhost:5432/ruckus",
        description="PostgreSQL database URL"
    )
    pool_size: int = Field(default=10, description="Database connection pool size")
    max_overflow: int = Field(default=20, description="Database max overflow connections")
    echo_sql: bool = Field(default=False, description="Echo SQL queries for debugging")
    max_retries: int = Field(default=3, description="Maximum retry attempts for database operations")
    retry_delay: float = Field(default=1.0, description="Base delay between retries in seconds")
    
    class Config:
        env_prefix = "RUCKUS_POSTGRES_"
        env_file = ".env"
        case_sensitive = False


class SQLiteSettings(BaseSettings):
    """SQLite storage backend settings."""
    
    database_path: str = Field(
        default="data/ruckus.db",
        description="Path to SQLite database file"
    )
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/ruckus.db",
        description="SQLite database URL"
    )
    echo_sql: bool = Field(default=False, description="Echo SQL queries for debugging")
    max_retries: int = Field(default=3, description="Maximum retry attempts for database operations")
    retry_delay: float = Field(default=1.0, description="Base delay between retries in seconds")
    
    class Config:
        env_prefix = "RUCKUS_SQLITE_"
        env_file = ".env"
        case_sensitive = False


class AgentSettings(BaseSettings):
    """Agent management settings."""
    
    agent_timeout: int = Field(default=60, description="Agent timeout in seconds")
    agent_heartbeat_timeout: int = Field(
        default=300, 
        description="Agent heartbeat timeout in seconds"
    )
    agent_registration_timeout: int = Field(
        default=60,
        description="Agent registration timeout in seconds"
    )
    heartbeat_interval: int = Field(default=30, description="Heartbeat interval in seconds")
    max_agents: int = Field(default=100, description="Maximum number of agents")
    
    class Config:
        env_prefix = "RUCKUS_AGENT_"
        env_file = ".env"
        case_sensitive = False


class SchedulerSettings(BaseSettings):
    """Job scheduler settings."""
    
    max_concurrent_jobs: int = Field(default=10, description="Maximum concurrent jobs")
    max_concurrent_jobs_per_agent: int = Field(
        default=1,
        description="Maximum concurrent jobs per agent"
    )
    job_timeout_default: int = Field(default=3600, description="Default job timeout in seconds")
    job_timeout_seconds: int = Field(
        default=3600,
        description="Default job timeout in seconds"
    )
    job_poll_interval: int = Field(
        default=30,
        description="Job polling interval in seconds"
    )
    scheduler_interval: int = Field(default=5, description="Scheduler interval in seconds")
    
    class Config:
        env_prefix = "RUCKUS_SCHEDULER_"
        env_file = ".env"
        case_sensitive = False


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or text)")
    log_config_file: str = Field(
        default="logging.yml",
        description="Path to logging configuration file"
    )
    
    class Config:
        env_prefix = "RUCKUS_LOG_"
        env_file = ".env"
        case_sensitive = False


class Settings(BaseSettings):
    """Main server settings that combines all configuration sections."""
    
    # Storage backend configuration
    storage_backend: StorageBackendType = Field(
        default=StorageBackendType.SQLITE,
        description="Storage backend type"
    )
    
    # Nested settings
    server: ServerSettings = Field(default_factory=ServerSettings)
    postgresql: PostgreSQLSettings = Field(default_factory=PostgreSQLSettings)
    sqlite: SQLiteSettings = Field(default_factory=SQLiteSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    class Config:
        env_prefix = "RUCKUS_"
        env_file = ".env"
        case_sensitive = False


settings = Settings()
