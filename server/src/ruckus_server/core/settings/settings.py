"""Settings configuration for the RUCKUS server."""

from pydantic import Field
from pydantic_settings import BaseSettings
from enum import Enum


class StorageBackendType(str, Enum):
    """Supported storage backend types."""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"


class PostgresStorageSettings(BaseSettings):
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
        """Pydantic configuration."""
        env_prefix = "RUCKUS_POSTGRES_"
        env_file = ".env"
        case_sensitive = False


class SQLiteStorageSettings(BaseSettings):
    """SQLite storage backend settings."""
    
    database_path: str = Field(
        default="data/ruckus.db",
        description="Path to SQLite database file"
    )
    echo_sql: bool = Field(default=False, description="Echo SQL queries for debugging")
    max_retries: int = Field(default=3, description="Maximum retry attempts for database operations")
    retry_delay: float = Field(default=1.0, description="Base delay between retries in seconds")
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "RUCKUS_SQLITE_"
        env_file = ".env"
        case_sensitive = False


class RuckusServerSettings(BaseSettings):
    """Configuration settings for the RUCKUS server.
    
    Uses pydantic-settings to load configuration from environment variables
    and configuration files.
    """
    
    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Storage backend configuration
    storage_backend: StorageBackendType = Field(
        default=StorageBackendType.SQLITE,
        description="Storage backend type"
    )
    
    # Agent management
    agent_heartbeat_timeout: int = Field(
        default=300, 
        description="Agent heartbeat timeout in seconds"
    )
    agent_registration_timeout: int = Field(
        default=60,
        description="Agent registration timeout in seconds"
    )
    
    # Job management
    job_timeout_seconds: int = Field(
        default=3600,
        description="Default job timeout in seconds"
    )
    job_poll_interval: int = Field(
        default=30,
        description="Job polling interval in seconds"
    )
    max_concurrent_jobs_per_agent: int = Field(
        default=1,
        description="Maximum concurrent jobs per agent"
    )
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_config_file: str = Field(
        default="logging.yml",
        description="Path to logging configuration file"
    )
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "RUCKUS_SERVER_"
        env_file = ".env"
        case_sensitive = False
