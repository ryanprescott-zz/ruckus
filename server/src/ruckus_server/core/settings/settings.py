"""Settings configuration for the RUCKUS server."""

from pydantic import Field
from pydantic_settings import BaseSettings


class RuckusServerSettings(BaseSettings):
    """Configuration settings for the RUCKUS server.
    
    Uses pydantic-settings to load configuration from environment variables
    and configuration files.
    """
    
    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Database configuration
    database_url: str = Field(
        default="postgresql://ruckus:ruckus@localhost:5432/ruckus",
        description="PostgreSQL database URL"
    )
    database_pool_size: int = Field(default=10, description="Database connection pool size")
    database_max_overflow: int = Field(default=20, description="Database max overflow connections")
    
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
