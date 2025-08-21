"""Agent configuration."""

from pydantic_settings import BaseSettings
from typing import Optional
from ruckus_common.models import AgentType


class Settings(BaseSettings):
    """Agent settings."""

    # Agent Identity
    agent_id: str = "agent-default"
    agent_name: Optional[str] = None
    agent_type: AgentType = AgentType.WHITE_BOX

    # Server
    host: str = "0.0.0.0"
    port: int = 8081
    debug: bool = False

    # Orchestrator
    orchestrator_url: Optional[str] = None
    heartbeat_interval: int = 30
    registration_timeout: int = 60

    # Job Execution
    max_concurrent_jobs: int = 1
    job_timeout_default: int = 3600

    # Model Management
    model_cache_dir: str = "/models"
    max_cached_models: int = 5

    # Monitoring
    enable_gpu_monitoring: bool = True
    enable_memory_monitoring: bool = True
    metrics_collection_interval: int = 1

    # Frameworks
    enable_transformers: bool = True
    enable_vllm: bool = False
    enable_pytorch: bool = True

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    class Config:
        env_prefix = "RUCKUS_AGENT_"
        env_file = ".env"


settings = Settings()