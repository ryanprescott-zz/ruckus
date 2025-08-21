"""Agent-specific models."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

from ruckus_common.models import AgentType, AgentCapabilitiesBase


class GPUInfo(BaseModel):
    """GPU device information."""
    index: int
    name: str
    uuid: Optional[str] = None
    memory_total_mb: int
    memory_available_mb: int
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    temperature: Optional[float] = None
    power_limit_w: Optional[int] = None
    utilization_percent: Optional[float] = None


class CPUInfo(BaseModel):
    """CPU information."""
    model: str
    cores_physical: int
    cores_logical: int
    frequency_mhz: float
    architecture: str
    cache_size_kb: Optional[int] = None


class SystemInfo(BaseModel):
    """System information."""
    hostname: str
    os: str
    os_version: str
    kernel: str
    python_version: str
    total_memory_gb: float
    available_memory_gb: float
    disk_total_gb: float
    disk_available_gb: float


class FrameworkInfo(BaseModel):
    """ML framework information."""
    name: str
    version: str
    available: bool
    capabilities: Dict[str, bool] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    """Available model information."""
    name: str
    path: str
    size_gb: float
    format: str  # pytorch, safetensors, gguf
    framework_compatible: List[str]
    loaded: bool = False


class HookInfo(BaseModel):
    """System hook/tool information."""
    name: str
    type: str  # gpu_monitor, cpu_monitor, profiler
    executable_path: str
    version: Optional[str] = None
    working: bool


class MetricCapability(BaseModel):
    """Metric collection capability."""
    name: str
    type: str  # performance, resource, quality
    available: bool
    collection_method: str
    requires: List[str] = Field(default_factory=list)


class AgentCapabilities(AgentCapabilitiesBase):
    """Extended agent capabilities."""
    # Hardware capabilities
    gpu_count: int = 0
    gpu_memory_total_gb: float = 0

    # Software capabilities
    frameworks_available: List[FrameworkInfo] = Field(default_factory=list)
    models_available: List[str] = Field(default_factory=list)

    # Monitoring capabilities
    metrics_available: List[MetricCapability] = Field(default_factory=list)
    hooks_available: List[str] = Field(default_factory=list)

    # Advanced capabilities
    quantization_support: List[str] = Field(default_factory=list)
    optimization_support: List[str] = Field(default_factory=list)


class AgentRegistration(BaseModel):
    """Complete agent registration."""
    agent_id: str
    agent_name: Optional[str] = None
    agent_type: AgentType
    agent_version: str = "0.1.0"

    # Hardware
    system: Optional[SystemInfo] = None
    cpu: Optional[CPUInfo] = None
    gpus: List[GPUInfo] = Field(default_factory=list)

    # Software
    frameworks: List[FrameworkInfo] = Field(default_factory=list)
    models: List[ModelInfo] = Field(default_factory=list)
    hooks: List[HookInfo] = Field(default_factory=list)

    # Capabilities
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    metrics_available: List[MetricCapability] = Field(default_factory=list)

    # Configuration
    max_concurrent_jobs: int = 1
    max_batch_size: int = 1

    # Metadata
    registered_at: datetime = Field(default_factory=datetime.utcnow)


class AgentStatus(BaseModel):
    """Agent status information."""
    agent_id: str
    status: str  # idle, busy, error
    running_jobs: List[str] = Field(default_factory=list)
    queued_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    resource_usage: Dict[str, float] = Field(default_factory=dict)
    last_job_completed: Optional[datetime] = None
    uptime_seconds: float = 0