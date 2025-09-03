"""Agent configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
from ruckus_common.models import AgentType


class Settings(BaseSettings):
    """Agent settings."""

    # Agent Identity
    agent_id: str = Field(default="agent-default", description="Unique identifier for this agent instance")
    agent_name: Optional[str] = Field(default=None, description="Human-readable name for this agent")
    agent_type: AgentType = Field(default=AgentType.WHITE_BOX, description="Type of agent (white_box, black_box, or hybrid)")

    # Server
    host: str = Field(default="0.0.0.0", description="Host address to bind the agent server to")
    port: int = Field(default=8081, description="Port number for the agent server")
    debug: bool = Field(default=False, description="Enable debug mode for development")

    # API configuration
    api_prefix: str = Field(default="/api/v1", description="API prefix path for all endpoints")
    openapi_prefix: str = Field(default="/api/static", description="Prefix path for OpenAPI static files")
    cors_origins: list = Field(default=["*"], description="List of allowed CORS origins for cross-origin requests")


    # Job Execution
    max_concurrent_jobs: int = Field(default=1, description="Maximum number of jobs that can run concurrently")
    job_timeout_default: int = Field(default=3600, description="Default timeout in seconds for job execution")

    # Model Management
    model_cache_dir: str = Field(default="/ruckus/models", description="Directory path for cached models (typically volume mounted)")
    max_cached_models: int = Field(default=5, description="Maximum number of models to keep in cache")

    # Monitoring
    enable_gpu_monitoring: bool = Field(default=True, description="Enable GPU utilization and memory monitoring")
    enable_memory_monitoring: bool = Field(default=True, description="Enable system memory monitoring")
    metrics_collection_interval: int = Field(default=1, description="Interval in seconds for collecting system metrics")

    # Frameworks
    enable_transformers: bool = Field(default=True, description="Enable Hugging Face Transformers framework support")
    enable_vllm: bool = Field(default=False, description="Enable vLLM high-performance inference engine support")
    enable_pytorch: bool = Field(default=True, description="Enable raw PyTorch model support")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    log_format: str = Field(default="json", description="Log output format (json or text)")

    class Config:
        env_prefix = "RUCKUS_AGENT_"
        env_file = ".env"


settings = Settings()
