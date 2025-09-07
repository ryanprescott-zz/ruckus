"""Server configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
from enum import Enum


class AppSettings(BaseSettings):
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
        default="datastore/ruckus.db",
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


class StorageBackendType(str, Enum):
    """Supported storage backend types."""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"


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
    
    # Agent protocol settings
    info_endpoint_path: str = Field(
        default="/api/v1/info", 
        description="Path to agent info endpoint"
    )
    
    class Config:
        env_prefix = "RUCKUS_AGENT_"
        env_file = ".env"
        case_sensitive = False


class HttpClientSettings(BaseSettings):
    """HTTP client configuration settings."""
    
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    initial_backoff: float = Field(default=1.0, description="Initial backoff delay in seconds")
    max_backoff: float = Field(default=3.0, description="Maximum backoff delay in seconds")
    retry_status_codes: list = Field(
        default=[500, 502, 503, 504, 408, 429], 
        description="HTTP status codes that trigger retries"
    )
    connection_timeout: float = Field(default=10.0, description="Connection timeout in seconds")
    read_timeout: float = Field(default=30.0, description="Read timeout in seconds")
    
    class Config:
        env_prefix = "RUCKUS_HTTP_"
        env_file = ".env"
        case_sensitive = False
        

class StorageSettings(BaseSettings):
    """Storage backend configuration settings."""
    
    storage_backend: StorageBackendType = Field(
        default=StorageBackendType.SQLITE,
        description="Storage backend type"
    )
    postgresql: PostgreSQLSettings = Field(default_factory=PostgreSQLSettings)
    sqlite: SQLiteSettings = Field(default_factory=SQLiteSettings)
    
    class Config:
        env_prefix = "RUCKUS_STORAGE_"
        env_file = ".env"
        case_sensitive = False


class AgentManagerSettings(BaseSettings):
    """Agent manager configuration settings containing everything it needs."""
    
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_config_file: str = Field(
        default="logging.yml",
        description="Path to logging configuration file"
    )
    
    # Component settings
    storage: StorageSettings = Field(default_factory=StorageSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    http_client: HttpClientSettings = Field(default_factory=HttpClientSettings)
    
    class Config:
        env_prefix = "RUCKUS_AGENT_MANAGER_"
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


class ExperimentManagerSettings(BaseSettings):
    """Experiment manager configuration settings containing everything it needs."""
    
    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_config_file: str = Field(
        default="logging.yml",
        description="Path to logging configuration file"
    )
    
    # Component settings
    storage: StorageSettings = Field(default_factory=StorageSettings)
    
    class Config:
        env_prefix = "RUCKUS_EXPERIMENT_MANAGER_"
        env_file = ".env"
        case_sensitive = False


class Settings(BaseSettings):
    """Main server settings that combines all configuration sections."""
    
    # Nested settings
    app: AppSettings = Field(default_factory=AppSettings)
    agent_manager: AgentManagerSettings = Field(default_factory=AgentManagerSettings)
    experiment_manager: ExperimentManagerSettings = Field(default_factory=ExperimentManagerSettings)
    postgresql: PostgreSQLSettings = Field(default_factory=PostgreSQLSettings)
    sqlite: SQLiteSettings = Field(default_factory=SQLiteSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    class Config:
        env_prefix = "RUCKUS_"
        env_file = ".env"
        case_sensitive = False


settings = Settings()
