"""Shared data models for RUCKUS system."""

from enum import Enum
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
from datetime import datetime


# Enumerations
class AgentType(str, Enum):
    """Agent capability levels."""
    WHITE_BOX = "white_box"  # Full system access
    GRAY_BOX = "gray_box"    # API access to models
    BLACK_BOX = "black_box"  # External API only


class JobStatus(str, Enum):
    """Job execution status."""
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class JobStage(str, Enum):
    """Detailed job execution stages."""
    INITIALIZING = "initializing"
    LOADING_MODEL = "loading_model"
    PREPARING_DATA = "preparing_data"
    WARMING_UP = "warming_up"
    RUNNING_INFERENCE = "running_inference"
    COLLECTING_METRICS = "collecting_metrics"
    FINALIZING = "finalizing"


class MetricType(str, Enum):
    """Categories of metrics."""
    PERFORMANCE = "performance"  # Latency, throughput
    RESOURCE = "resource"        # Memory, GPU usage
    QUALITY = "quality"          # ROUGE, accuracy
    COST = "cost"               # Estimated cost metrics


class TaskType(str, Enum):
    """Supported benchmark task types."""
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    QUESTION_ANSWERING = "qa"
    TRANSLATION = "translation"
    CUSTOM = "custom"


# Base Models
class TimestampedModel(BaseModel):
    """Base model with timestamp fields."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# Experiment Models
class ExperimentSpec(TimestampedModel):
    """Specification for a benchmark experiment."""
    experiment_id: str
    name: str
    description: Optional[str] = None
    models: List[str]
    frameworks: List[str]
    hardware_targets: List[str] = Field(default_factory=lambda: ["any"])
    task_type: TaskType
    task_config: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=0, ge=0, le=10)
    owner: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    @validator("experiment_id")
    def validate_experiment_id(cls, v):
        if not v or not v.strip():
            raise ValueError("experiment_id cannot be empty")
        return v


# Job Models
class JobSpec(TimestampedModel):
    """Specification for a single job."""
    job_id: str
    experiment_id: str
    agent_id: Optional[str] = None
    model: str
    framework: str
    hardware_target: str
    task_type: TaskType
    task_config: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=3600, gt=0)
    max_retries: int = Field(default=3, ge=0)
    priority: int = Field(default=0, ge=0, le=10)
    status: JobStatus = JobStatus.QUEUED
    retry_count: int = Field(default=0, ge=0)


class JobRequest(BaseModel):
    """Request to execute a job (server -> agent)."""
    job_id: str
    experiment_id: str
    model: str
    framework: str
    task_type: TaskType
    task_config: Dict[str, Any]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required_metrics: List[str] = Field(default_factory=list)
    optional_metrics: List[str] = Field(default_factory=list)
    timeout_seconds: int = Field(default=3600, gt=0)
    callback_url: Optional[str] = None


class JobUpdate(BaseModel):
    """Progress update for a running job (agent -> server)."""
    job_id: str
    status: JobStatus
    stage: Optional[JobStage] = None
    progress: Optional[int] = Field(None, ge=0, le=100)
    message: Optional[str] = None
    metrics_snapshot: Optional[Dict[str, Any]] = None
    estimated_remaining_seconds: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class JobResult(BaseModel):
    """Final result of a completed job (agent -> server)."""
    job_id: str
    experiment_id: str
    status: JobStatus
    started_at: datetime
    completed_at: datetime
    duration_seconds: float

    # Results
    output: Optional[Any] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)

    # Metadata
    model_actual: Optional[str] = None  # Actual model used (may differ)
    framework_version: Optional[str] = None
    hardware_info: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)

    # Error information
    error: Optional[str] = None
    error_type: Optional[str] = None
    traceback: Optional[str] = None

    @validator("duration_seconds")
    def validate_duration(cls, v):
        if v < 0:
            raise ValueError("duration_seconds must be non-negative")
        return v


# Agent Models
class AgentCapabilitiesBase(BaseModel):
    """Base capabilities that all agents must report."""
    agent_id: str
    agent_type: AgentType

    # Core capabilities
    model_loading: bool = False
    tokenization: bool = False
    streaming: bool = False
    batch_processing: bool = False

    # Monitoring capabilities
    memory_monitoring: bool = False
    gpu_monitoring: bool = False

    # Supported frameworks
    frameworks_supported: List[str] = Field(default_factory=list)

    # Constraints
    max_concurrent_jobs: int = Field(default=1, gt=0)
    max_batch_size: int = Field(default=1, gt=0)


# Metric Models
class MetricDefinition(BaseModel):
    """Definition of a metric."""
    name: str
    type: MetricType
    unit: Optional[str] = None
    description: Optional[str] = None
    higher_is_better: bool = True
    requires_capabilities: List[str] = Field(default_factory=list)


class MetricValue(BaseModel):
    """A single metric measurement."""
    name: str
    value: Any
    unit: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Task Models
class TaskConfig(BaseModel):
    """Configuration for a benchmark task."""
    task_type: TaskType
    dataset: Optional[str] = None
    data_path: Optional[str] = None
    prompt_template: Optional[str] = None
    evaluation_metrics: List[str] = Field(default_factory=list)
    samples: Optional[int] = None
    seed: Optional[int] = 42
    custom_config: Dict[str, Any] = Field(default_factory=dict)


# Error Models
class ErrorInfo(BaseModel):
    """Error information."""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    traceback: Optional[str] = None


# Agent Registration Models
class AgentRegistrationResponse(BaseModel):
    """Response to agent registration."""
    agent_id: str
    agent_name: Optional[str] = None
    message: Optional[str] = None
    server_time: datetime = Field(default_factory=datetime.utcnow)


class AgentInfo(BaseModel):
    """Agent system information."""
    agent_id: str
    agent_name: Optional[str] = None
    agent_type: AgentType
    system_info: Dict[str, Any] = Field(default_factory=dict)
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class AgentInfoResponse(BaseModel):
    """Response containing agent system information."""
    agent_info: AgentInfo


# Response Models
class HealthStatus(BaseModel):
    """Health check response."""
    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    uptime_seconds: float
    current_jobs: int = 0
    total_completed_jobs: int = 0
    last_job_completed: Optional[datetime] = None
    issues: List[str] = Field(default_factory=list)