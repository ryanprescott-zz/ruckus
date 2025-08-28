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


class AgentStatusEnum(str, Enum):
    """Agent status values."""
    ACTIVE = "active"          # Agent is running one or more jobs
    IDLE = "idle"             # Agent is not running any jobs
    ERROR = "error"           # Agent is in an error state and cannot run jobs
    UNAVAILABLE = "unavailable"  # Agent cannot be contacted


# Base Models
class TimestampedModel(BaseModel):
    """Base model with timestamp fields."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# Parameter Sweep Models
class ParameterSweep(BaseModel):
    """Definition of parameter sweep configurations."""
    name: str
    values: List[Any]  # List of values to sweep over
    sweep_type: str = "grid"  # "grid", "random", "adaptive"
    
class ParameterGrid(BaseModel):
    """Complete parameter grid specification."""
    parameters: Dict[str, ParameterSweep] = Field(default_factory=dict)
    samples_per_config: int = Field(default=1, ge=1, description="Number of repetitions per parameter combination")
    random_seed: Optional[int] = None
    
    def generate_configurations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for this grid."""
        if not self.parameters:
            return [{}] * self.samples_per_config
            
        import itertools
        param_names = list(self.parameters.keys())
        param_values = [self.parameters[name].values for name in param_names]
        
        configs = []
        for combination in itertools.product(*param_values):
            config = dict(zip(param_names, combination))
            # Repeat each config samples_per_config times
            for sample_idx in range(self.samples_per_config):
                config_with_sample = config.copy()
                config_with_sample["_sample_idx"] = sample_idx
                if self.random_seed is not None:
                    config_with_sample["_seed"] = self.random_seed + hash(str(config)) + sample_idx
                configs.append(config_with_sample)
        
        return configs

# Agent Selection Models  
class AgentRequirements(BaseModel):
    """Requirements for agent selection."""
    # Model requirements
    required_models: List[str] = Field(default_factory=list, description="Models that must be available")
    preferred_models: List[str] = Field(default_factory=list, description="Preferred models (bonus in scoring)")
    
    # Framework requirements
    required_frameworks: List[str] = Field(default_factory=list)
    preferred_frameworks: List[str] = Field(default_factory=list)
    
    # Hardware requirements
    min_gpu_count: int = Field(default=0, ge=0)
    min_gpu_memory_mb: Optional[int] = None
    preferred_gpu_types: List[str] = Field(default_factory=list)
    
    # Agent capabilities
    required_capabilities: List[str] = Field(default_factory=list)
    preferred_capabilities: List[str] = Field(default_factory=list)
    
    # Performance requirements  
    max_concurrent_jobs: Optional[int] = None
    agent_types: List[AgentType] = Field(default_factory=list, description="Allowed agent types")
    
    # Exclusions
    excluded_agents: List[str] = Field(default_factory=list, description="Agent IDs to exclude")
    excluded_tags: List[str] = Field(default_factory=list)

# Result Specification Models
class ExpectedOutput(BaseModel):
    """Specification of expected job outputs."""
    required_metrics: List[str] = Field(default_factory=list, description="Metrics that must be captured")
    optional_metrics: List[str] = Field(default_factory=list, description="Additional metrics to capture if available")
    artifacts: List[str] = Field(default_factory=list, description="Files/artifacts to collect")
    output_format: str = "json"  # "json", "csv", "parquet"
    
class AggregationStrategy(BaseModel):
    """How to aggregate results across multiple jobs."""
    method: str = "collect_all"  # "collect_all", "average", "best", "statistical"
    group_by: List[str] = Field(default_factory=list, description="Fields to group results by")
    metrics_aggregation: Dict[str, str] = Field(default_factory=dict, description="Per-metric aggregation (mean, median, max, etc.)")

# Enhanced Experiment Models
class ExperimentSpec(TimestampedModel):
    """Enhanced specification for a benchmark experiment with orchestration support."""
    experiment_id: str
    name: str
    description: Optional[str] = None
    
    # Target models and task specification
    models: List[str]  # Models to benchmark
    task_type: TaskType
    task_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Parameter sweep configuration
    parameter_grid: ParameterGrid = Field(default_factory=ParameterGrid)
    base_parameters: Dict[str, Any] = Field(default_factory=dict, description="Base parameters applied to all jobs")
    
    # Agent selection requirements
    agent_requirements: AgentRequirements = Field(default_factory=AgentRequirements)
    
    # Output and aggregation specification
    expected_output: ExpectedOutput = Field(default_factory=ExpectedOutput)
    aggregation_strategy: AggregationStrategy = Field(default_factory=AggregationStrategy)
    
    # Execution configuration
    priority: int = Field(default=0, ge=0, le=10)
    timeout_seconds: int = Field(default=3600, gt=0)
    max_retries: int = Field(default=2, ge=0)
    max_parallel_jobs: Optional[int] = Field(default=None, description="Max concurrent jobs for this experiment")
    
    # Metadata
    owner: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)

    @validator("experiment_id")
    def validate_experiment_id(cls, v):
        if not v or not v.strip():
            raise ValueError("experiment_id cannot be empty")
        return v
    
    def estimate_job_count(self) -> int:
        """Estimate total number of jobs this experiment will generate."""
        config_count = len(self.parameter_grid.generate_configurations())
        return config_count * len(self.models)

# Orchestration Execution Models
class ExperimentExecution(TimestampedModel):
    """Tracks the server-side execution state of an experiment."""
    experiment_id: str
    spec: ExperimentSpec
    status: str = "pending"  # pending, planning, running, completed, failed, cancelled
    
    # Job planning results (server maintains the job queue)
    total_jobs: int = 0
    planned_jobs: List[str] = Field(default_factory=list, description="List of all planned job IDs")
    
    # Execution tracking (server-side job queue management)
    queued_jobs: List[str] = Field(default_factory=list, description="Jobs waiting to be dispatched")
    running_jobs: Dict[str, str] = Field(default_factory=dict, description="job_id -> agent_id mapping")
    completed_jobs: List[str] = Field(default_factory=list)
    failed_jobs: List[str] = Field(default_factory=list)
    
    # Progress tracking
    progress_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results aggregation
    results_summary: Dict[str, Any] = Field(default_factory=dict)
    error_summary: Optional[str] = None
    aggregated_results: Optional[Dict[str, Any]] = Field(default=None, description="Aggregated results from all completed jobs")
    
    def calculate_progress(self) -> float:
        """Calculate current progress percentage."""
        if self.total_jobs == 0:
            return 0.0
        completed = len(self.completed_jobs) + len(self.failed_jobs)
        return min(100.0, (completed / self.total_jobs) * 100.0)


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
    runs_per_job: int = Field(default=1, ge=1, description="Number of runs to execute sequentially for statistical reliability")
    status: JobStatus = JobStatus.QUEUED
    retry_count: int = Field(default=0, ge=0)
    
    # Results (populated when job completes)
    results: Optional[Dict[str, Any]] = Field(default=None, description="Job execution results and metrics")


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
    runs_per_job: int = Field(default=1, ge=1, description="Number of runs to execute sequentially for statistical reliability")
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


class SingleRunResult(BaseModel):
    """Results from a single run within a multi-run job."""
    run_id: int = Field(ge=0, description="Sequential run number (0-indexed)")
    is_cold_start: bool = Field(description="Whether this run included model loading")
    started_at: datetime
    completed_at: datetime
    duration_seconds: float = Field(ge=0)
    
    # Performance metrics for this specific run
    metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Model loading metrics (only present for cold start runs)
    model_load_time_seconds: Optional[float] = Field(default=None, ge=0)
    model_load_memory_mb: Optional[float] = Field(default=None, ge=0)
    
    # Error information (if this run failed)
    error: Optional[str] = None
    error_type: Optional[str] = None
    
    @validator("model_load_time_seconds")
    def validate_cold_start_timing(cls, v, values):
        if v is not None and not values.get("is_cold_start", False):
            raise ValueError("model_load_time_seconds should only be set for cold start runs")
        return v


class MetricStatistics(BaseModel):
    """Statistical summary for a metric across multiple runs."""
    mean: float
    std: float
    min: float
    max: float
    median: float
    count: int = Field(ge=1)
    raw_values: List[float] = Field(description="All raw values for reference")
    outliers: List[int] = Field(default_factory=list, description="Run IDs identified as outliers (>2σ)")
    
    @validator("raw_values")
    def validate_raw_values_count(cls, v, values):
        count = values.get("count", 0)
        if len(v) != count:
            raise ValueError(f"raw_values length ({len(v)}) must match count ({count})")
        return v


class MultiRunJobResult(BaseModel):
    """Enhanced result for multi-run jobs with cold start separation and statistical analysis."""
    job_id: str
    experiment_id: str
    total_runs: int = Field(ge=1)
    successful_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    
    # Individual run data
    individual_runs: List[SingleRunResult] = Field(description="Results from each individual run")
    
    # Aggregated statistics (computed from warm runs only, excluding cold start)
    summary_stats: Dict[str, MetricStatistics] = Field(
        default_factory=dict,
        description="Statistical summary for each metric across warm runs"
    )
    
    # Cold start specific data (separate from warm run statistics)
    cold_start_data: Optional[SingleRunResult] = Field(
        default=None,
        description="Cold start run data with model loading metrics"
    )
    
    # Overall job metadata
    started_at: datetime
    completed_at: datetime
    total_duration_seconds: float = Field(ge=0)
    model_actual: Optional[str] = None
    framework_version: Optional[str] = None
    hardware_info: Dict[str, Any] = Field(default_factory=dict)
    
    # GPU benchmarking results (if performed)
    gpu_benchmark_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Memory bandwidth and FLOPS benchmark results"
    )
    
    @validator("successful_runs", "failed_runs")
    def validate_run_counts(cls, v, values):
        total = values.get("total_runs", 0)
        if "successful_runs" in values and "failed_runs" in values:
            if values["successful_runs"] + values["failed_runs"] != total:
                raise ValueError("successful_runs + failed_runs must equal total_runs")
        return v


# Orchestrator Communication Models
class AgentScore(BaseModel):
    """Agent compatibility score for a specific job."""
    agent_id: str
    agent_name: str
    score: float = Field(ge=0.0, le=100.0, description="Compatibility score (0-100)")
    
    # Score breakdown
    model_compatibility: float = 0.0
    framework_compatibility: float = 0.0
    hardware_suitability: float = 0.0
    capability_match: float = 0.0
    availability_bonus: float = 0.0
    
    # Detailed reasons
    compatible_models: List[str] = Field(default_factory=list)
    missing_requirements: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    estimated_queue_time_seconds: Optional[float] = None

class JobAssignment(BaseModel):
    """Assignment of a job to a specific agent."""
    job_id: str
    agent_id: str
    agent_name: str
    assigned_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Assignment details
    score: float = Field(description="Agent compatibility score")
    expected_duration_seconds: Optional[float] = None
    assignment_reason: str = "automatic"  # "automatic", "manual", "retry", "rebalance"
    
    # Job context for the agent
    job_context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for job execution")

class ExperimentSubmission(BaseModel):
    """User submission of an experiment to the orchestrator."""
    spec: ExperimentSpec
    submit_immediately: bool = Field(default=True, description="Whether to start execution immediately")
    dry_run: bool = Field(default=False, description="If true, only plan jobs without executing")

class ExperimentSubmissionResponse(BaseModel):
    """Response to experiment submission."""
    experiment_id: str
    status: str  # "accepted", "rejected", "queued"
    message: Optional[str] = None
    
    # Planning results
    estimated_jobs: int = 0
    estimated_duration_hours: Optional[float] = None
    estimated_cost: Optional[float] = None
    
    # Execution tracking URL/info
    tracking_url: Optional[str] = None
    submitted_at: datetime = Field(default_factory=datetime.utcnow)

class AgentMatchingRequest(BaseModel):
    """Request to find compatible agents for jobs."""
    jobs: List[str] = Field(description="Job IDs to find agents for")
    requirements: AgentRequirements
    max_agents_per_job: int = Field(default=3, ge=1, description="Max agent candidates to return per job")
    include_unavailable: bool = Field(default=False, description="Include agents that are currently busy")

class AgentMatchingResponse(BaseModel):
    """Response with agent compatibility scores."""
    job_assignments: Dict[str, List[AgentScore]] = Field(default_factory=dict, description="job_id -> list of compatible agents")
    unassignable_jobs: List[str] = Field(default_factory=list, description="Jobs with no compatible agents")
    warnings: List[str] = Field(default_factory=list)
    processed_at: datetime = Field(default_factory=datetime.utcnow)


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


class RegisteredAgentInfo(AgentInfo):
    """Registered agent information that extends AgentInfo with server-side metadata."""
    agent_url: str = Field(..., description="Base URL of the registered agent")
    registered_at: datetime = Field(default_factory=datetime.utcnow, description="When the agent was registered")


# Response Models
class AgentStatus(BaseModel):
    """Agent status information."""
    agent_id: str
    status: AgentStatusEnum
    running_jobs: List[str] = Field(default_factory=list, description="List of currently running job IDs")
    queued_jobs: List[str] = Field(default_factory=list, description="List of queued job IDs")
    uptime_seconds: float = Field(description="Agent uptime in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Current timestamp")


class HealthStatus(BaseModel):
    """Health check response."""
    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    uptime_seconds: float
    current_jobs: int = 0
    total_completed_jobs: int = 0
    last_job_completed: Optional[datetime] = None
    issues: List[str] = Field(default_factory=list)