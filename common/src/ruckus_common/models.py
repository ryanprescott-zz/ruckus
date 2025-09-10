"""Shared data models for RUCKUS system."""

from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, computed_field
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib


# Enumerations
class AgentType(str, Enum):
    """Agent capability levels."""
    WHITE_BOX = "white_box"  # Full system access
    GRAY_BOX = "gray_box"    # API access to models
    BLACK_BOX = "black_box"  # External API only


class JobStatusEnum(str, Enum):
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


class JobStatus(BaseModel):
    """Job status with timestamp and message."""
    status: JobStatusEnum
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message: str = ""


class MetricType(str, Enum):
    """Categories of metrics."""
    PERFORMANCE = "performance"  # Latency, throughput
    RESOURCE = "resource"        # Memory, GPU usage
    QUALITY = "quality"          # ROUGE, accuracy
    COST = "cost"               # Estimated cost metrics


class TaskType(str, Enum):
    """Supported benchmark task types."""
    LLM_GENERATION = "llm_generation"
    GPU_BENCHMARK = "gpu_benchmark"
    MEMORY_BENCHMARK = "memory_benchmark"
    COMPUTE_BENCHMARK = "compute_benchmark"


class AgentStatusEnum(str, Enum):
    """Agent status values."""
    ACTIVE = "active"          # Agent is running one or more jobs
    IDLE = "idle"             # Agent is not running any jobs
    ERROR = "error"           # Agent is in an error state and cannot run jobs
    UNAVAILABLE = "unavailable"  # Agent cannot be contacted


class JobResultType(str, Enum):
    """Types of job execution results."""
    SUCCESS = "success"             # Job completed successfully
    PARTIAL_SUCCESS = "partial"     # Some runs succeeded, some failed
    CONSTRAINT_FAILURE = "constraint"  # Hardware/software constraint hit
    EXECUTION_FAILURE = "execution"    # Code/runtime error
    CANCELLED = "cancelled"         # Job was cancelled


# System Detection Enums
class OSType(str, Enum):
    """Operating system types."""
    LINUX = "Linux"
    WINDOWS = "Windows"
    DARWIN = "Darwin"  # macOS
    UNKNOWN = "Unknown"


class CPUArchitecture(str, Enum):
    """CPU architecture types."""
    X86_64 = "x86_64"
    AMD64 = "amd64"
    ARM64 = "arm64"
    AARCH64 = "aarch64"
    ARM = "arm"
    I386 = "i386"
    UNKNOWN = "unknown"


# GPU Detection Enums
class GPUVendor(str, Enum):
    """GPU vendor types."""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    UNKNOWN = "unknown"


class TensorCoreGeneration(str, Enum):
    """NVIDIA Tensor Core generations."""
    FIRST_GEN = "1st_gen"      # Volta (V100, Titan V)
    SECOND_GEN = "2nd_gen"     # Turing (RTX 20 series)
    THIRD_GEN = "3rd_gen"      # Ampere (RTX 30 series, A100)
    FOURTH_GEN = "4th_gen"     # Ada Lovelace (RTX 40 series), Hopper (H100)
    NONE = "none"              # No tensor cores


class PrecisionType(str, Enum):
    """Supported precision types for GPU operations."""
    FP64 = "fp64"    # Double precision
    FP32 = "fp32"    # Single precision
    TF32 = "tf32"    # TensorFloat-32 (Ampere+)
    FP16 = "fp16"    # Half precision
    BF16 = "bf16"    # Brain Float 16
    INT8 = "int8"    # 8-bit integer
    FP8 = "fp8"      # 8-bit float (Hopper+)


class DetectionMethod(str, Enum):
    """GPU detection methods."""
    PYNVML = "pynvml"              # Primary NVIDIA method
    NVIDIA_SMI = "nvidia_smi"      # NVIDIA fallback
    PYTORCH_CUDA = "pytorch_cuda"  # PyTorch CUDA
    PYTORCH_MPS = "pytorch_mps"    # Apple Silicon
    PYTORCH_ROCM = "pytorch_rocm"  # AMD ROCm
    UNKNOWN = "unknown"


# Framework Enums
class FrameworkName(str, Enum):
    """ML framework names."""
    PYTORCH = "pytorch"
    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    TENSORRT = "tensorrt"
    ONNX = "onnx"
    TRITON = "triton"
    UNKNOWN = "unknown"


# Hook/Tool Enums
class HookType(str, Enum):
    """System monitoring hook types."""
    GPU_MONITOR = "gpu_monitor"      # nvidia-smi, rocm-smi
    CPU_MONITOR = "cpu_monitor"      # htop, top
    MEMORY_MONITOR = "memory_monitor"  # free, vmstat
    DISK_MONITOR = "disk_monitor"    # iotop, iostat
    PROCESS_MONITOR = "process_monitor"  # ps, pidstat
    NETWORK_MONITOR = "network_monitor"  # netstat, iftop
    PROFILER = "profiler"           # nsight, perf
    UNKNOWN = "unknown"


# Metric Collection Enums
class MetricCollectionMethod(str, Enum):
    """Methods for collecting metrics."""
    TIMER = "timer"                    # Built-in timing
    CALCULATION = "calculation"        # Computed metrics
    NVIDIA_SMI = "nvidia_smi"         # NVIDIA GPU metrics
    PYNVML = "pynvml"                 # NVIDIA ML library
    PSUTIL = "psutil"                 # System metrics
    PYTORCH = "pytorch"               # PyTorch built-in metrics
    CUSTOM = "custom"                 # Custom collection logic
    EXTERNAL_API = "external_api"     # External service API
    UNKNOWN = "unknown"


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

class PromptRole(str, Enum):
    """Roles for prompt messages."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class PromptMessage(BaseModel):
    """Message in a prompt template."""
    role: PromptRole
    content: str

class PromptTemplate(BaseModel):
    """Template for prompts used in generation tasks."""
    messages: List[PromptMessage] = Field(default_factory=list)
    extra_body: Optional[Dict] = None


class TaskSpec(BaseModel):
    """Specification for a task to be executed."""
    name: str
    type: TaskType
    description: Optional[str] = None
    params: Any


class LLMGenerationParams(BaseModel):
    """Parameters for LLM generation tasks."""
    prompt_template: PromptTemplate


class FrameworkSpec(BaseModel):
    """Specification for a framework to be used."""
    name: FrameworkName
    params: Any


class MetricsSpec(BaseModel):
    """Specification for metrics to be collected."""
    metrics: Dict


# Capability Requirement Models
class GPURequirements(BaseModel):
    """Requirements for GPU hardware experiments."""
    min_gpu_count: int = Field(default=1, ge=1, description="Minimum number of GPUs required")
    min_memory_mb: Optional[int] = Field(default=None, description="Minimum GPU memory in MB")
    required_vendors: List[GPUVendor] = Field(default_factory=list, description="Required GPU vendors (empty = any)")
    required_precision_types: List[PrecisionType] = Field(default_factory=list, description="Required precision support")
    tensor_cores_required: bool = Field(default=False, description="Whether tensor cores are required")
    min_tensor_core_generation: Optional[TensorCoreGeneration] = Field(default=None, description="Minimum tensor core generation")


class ModelRequirements(BaseModel):
    """Requirements for model inference experiments."""
    required_frameworks: List[FrameworkName] = Field(default_factory=list, description="Required ML frameworks")
    model_architectures: List[str] = Field(default_factory=list, description="Compatible model architectures (e.g., 'llama', 'bert')")
    min_gpu_memory_gb: Optional[float] = Field(default=None, description="Minimum GPU memory for model loading")
    tokenization_required: bool = Field(default=True, description="Whether tokenization capability is required")
    streaming_support: bool = Field(default=False, description="Whether streaming generation is required")


class SystemRequirements(BaseModel):
    """General system requirements for experiments."""
    min_total_memory_gb: Optional[float] = Field(default=None, description="Minimum system RAM in GB")
    required_os_types: List[OSType] = Field(default_factory=list, description="Required operating systems (empty = any)")
    required_cpu_architectures: List[CPUArchitecture] = Field(default_factory=list, description="Required CPU architectures (empty = any)")
    min_cpu_cores: Optional[int] = Field(default=None, description="Minimum CPU cores")


class ExperimentCapabilityRequirements(BaseModel):
    """Complete capability requirements for an experiment type."""
    required_capabilities: List[str] = Field(default_factory=list, description="Required agent capabilities")
    required_metrics: List[str] = Field(default_factory=list, description="Metrics that must be collectable")
    optional_metrics: List[str] = Field(default_factory=list, description="Additional metrics to collect if available")
    gpu_requirements: Optional[GPURequirements] = Field(default=None, description="GPU-specific requirements")
    model_requirements: Optional[ModelRequirements] = Field(default=None, description="Model-specific requirements") 
    system_requirements: Optional[SystemRequirements] = Field(default=None, description="General system requirements")
    timeout_seconds: int = Field(default=3600, gt=0, description="Default timeout for this experiment type")


# Specialized Task Parameter Models
class GPUBenchmarkParams(BaseModel):
    """Parameters for GPU benchmark experiments."""
    test_memory_bandwidth: bool = Field(default=True, description="Test memory bandwidth")
    test_compute_flops: bool = Field(default=True, description="Test FLOPS across precisions")
    test_tensor_cores: bool = Field(default=False, description="Test tensor core performance specifically")
    max_memory_usage_percent: float = Field(default=80.0, ge=10.0, le=95.0, description="Max GPU memory to use for tests")
    benchmark_duration_seconds: int = Field(default=30, ge=5, le=300, description="Duration for each benchmark")


class MemoryBenchmarkParams(BaseModel):
    """Parameters for memory bandwidth benchmark experiments."""
    test_sizes_mb: List[int] = Field(default_factory=lambda: [64, 256, 1024], description="Memory test sizes in MB")
    test_patterns: List[str] = Field(default_factory=lambda: ["sequential", "random"], description="Memory access patterns")
    iterations_per_size: int = Field(default=10, ge=1, le=100, description="Iterations per test size")


class ComputeBenchmarkParams(BaseModel):
    """Parameters for compute benchmark experiments."""
    precision_types: List[PrecisionType] = Field(default_factory=lambda: [PrecisionType.FP32, PrecisionType.FP16], description="Precision types to test")
    matrix_sizes: List[int] = Field(default_factory=lambda: [1024, 2048, 4096], description="Matrix sizes for FLOPS tests")
    include_tensor_ops: bool = Field(default=True, description="Include tensor operation benchmarks")


# Enhanced Experiment Models
class ExperimentSpec(TimestampedModel):
    """Enhanced specification for a benchmark experiment with orchestration support."""
    name: str
    description: Optional[str] = None
    model: str
    task: TaskSpec
    framework: FrameworkSpec
    metrics: MetricsSpec
    capability_requirements: Optional[ExperimentCapabilityRequirements] = Field(default=None, description="Agent capability requirements for this experiment")

    @computed_field
    @property
    def id(self) -> str:
        """Generate a short experiment ID based on the name."""
        name_hash = hashlib.md5(self.name.encode()).hexdigest()[:6]
        return f"exp_{name_hash}"

    def estimate_job_count(self) -> int:
        """Estimate total number of jobs this experiment will generate."""
        return 1  # Simplified for new structure


# Specialized Experiment Types
class GPUBenchmarkExperiment(ExperimentSpec):
    """Specialized experiment for GPU hardware benchmarking."""
    
    def __init__(self, **data):
        # Set defaults for GPU benchmark experiments
        if 'task' not in data:
            data['task'] = TaskSpec(
                name="GPU Benchmark",
                type=TaskType.GPU_BENCHMARK,
                description="Comprehensive GPU hardware benchmarking",
                params=GPUBenchmarkParams()
            )
        
        if 'model' not in data:
            data['model'] = "hardware-test"  # Not applicable for hardware tests
            
        if 'framework' not in data:
            data['framework'] = FrameworkSpec(
                name=FrameworkName.PYTORCH,  # Use PyTorch for GPU ops
                params={}
            )
            
        if 'metrics' not in data:
            data['metrics'] = MetricsSpec(
                metrics={
                    "memory_bandwidth_gb_s": {"required": True, "type": "performance"},
                    "compute_flops_fp32": {"required": True, "type": "performance"},
                    "compute_flops_fp16": {"required": False, "type": "performance"},
                    "gpu_temperature_celsius": {"required": False, "type": "resource"},
                    "gpu_power_usage_watts": {"required": False, "type": "resource"}
                }
            )
            
        if 'capability_requirements' not in data:
            data['capability_requirements'] = ExperimentCapabilityRequirements(
                required_capabilities=["gpu_monitoring", "memory_monitoring"],
                required_metrics=["memory_bandwidth", "flops_fp32"],
                optional_metrics=["flops_fp16", "flops_bf16", "gpu_temperature", "gpu_power"],
                gpu_requirements=GPURequirements(
                    min_gpu_count=1,
                    min_memory_mb=1024  # At least 1GB GPU memory
                ),
                timeout_seconds=600  # 10 minutes for hardware tests
            )
        
        super().__init__(**data)


class LLMInferenceExperiment(ExperimentSpec):
    """Specialized experiment for LLM inference benchmarking."""
    
    def __init__(self, **data):
        # Set defaults for LLM inference experiments
        if 'task' not in data:
            data['task'] = TaskSpec(
                name="LLM Inference",
                type=TaskType.LLM_GENERATION,
                description="Large Language Model inference performance testing",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[PromptMessage(role=PromptRole.USER, content="What is the capital of France?")]
                    )
                )
            )
        
        if 'framework' not in data:
            data['framework'] = FrameworkSpec(
                name=FrameworkName.TRANSFORMERS,
                params={}
            )
            
        if 'metrics' not in data:
            data['metrics'] = MetricsSpec(
                metrics={
                    "inference_time_seconds": {"required": True, "type": "performance"},
                    "tokens_per_second": {"required": True, "type": "performance"},
                    "first_token_latency_ms": {"required": False, "type": "performance"},
                    "model_load_time_seconds": {"required": False, "type": "performance"},
                    "peak_memory_usage_mb": {"required": False, "type": "resource"},
                    "gpu_utilization_percent": {"required": False, "type": "resource"}
                }
            )
            
        if 'capability_requirements' not in data:
            data['capability_requirements'] = ExperimentCapabilityRequirements(
                required_capabilities=["model_loading", "tokenization"],
                required_metrics=["inference_time", "throughput"],
                optional_metrics=["first_token_latency", "memory_usage", "gpu_utilization"],
                model_requirements=ModelRequirements(
                    required_frameworks=[FrameworkName.TRANSFORMERS, FrameworkName.VLLM],
                    tokenization_required=True,
                    min_gpu_memory_gb=4.0  # Minimum for small models
                ),
                gpu_requirements=GPURequirements(
                    min_gpu_count=1,
                    min_memory_mb=4096  # 4GB minimum for LLM inference
                ),
                timeout_seconds=1800  # 30 minutes for model experiments
            )
        
        super().__init__(**data)

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


class ExecuteJobRequest(BaseModel):
    """Request to execute a job with experiment specification."""
    experiment_spec: ExperimentSpec
    job_id: str


class JobUpdate(BaseModel):
    """Progress update for a running job (agent -> server)."""
    job_id: str
    status: JobStatusEnum
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
    status: JobStatusEnum
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

    @field_validator("duration_seconds")
    @classmethod
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
    
    @model_validator(mode='after')
    def validate_cold_start_timing(self):
        if self.model_load_time_seconds is not None and not self.is_cold_start:
            raise ValueError("model_load_time_seconds should only be set for cold start runs")
        return self


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
    
    @model_validator(mode='after')
    def validate_raw_values_count(self):
        if len(self.raw_values) != self.count:
            raise ValueError(f"raw_values length ({len(self.raw_values)}) must match count ({self.count})")
        return self


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
    
    @field_validator("successful_runs", "failed_runs")
    @classmethod
    def validate_run_counts(cls, v):
        # Cross-field validation will be handled at model level in Pydantic v2
        return v


# Orchestrator Communication Models
class AgentCompatibility(BaseModel):
    """Binary agent compatibility with rich capability information."""
    agent_id: str
    agent_name: str
    can_run: bool = Field(description="Whether this agent can execute the experiment")
    
    # Rich capability information for user decision-making
    available_capabilities: List[str] = Field(default_factory=list, description="Capabilities this agent provides")
    missing_requirements: List[str] = Field(default_factory=list, description="Required capabilities this agent lacks")
    supported_features: List[str] = Field(default_factory=list, description="Optional features this agent supports")
    warnings: List[str] = Field(default_factory=list, description="Potential limitations or concerns")
    
    # Hardware and software details
    hardware_summary: Dict[str, Any] = Field(default_factory=dict, description="Key hardware specs (GPU, memory, etc.)")
    framework_versions: Dict[str, str] = Field(default_factory=dict, description="Available framework versions")
    compatible_models: List[str] = Field(default_factory=list, description="Models this agent has loaded")
    
    # Operational context
    estimated_queue_time_seconds: Optional[float] = Field(default=None, description="Expected wait time if agent is busy")
    last_capability_check: datetime = Field(default_factory=datetime.utcnow, description="When compatibility was last verified")

class JobAssignment(BaseModel):
    """Assignment of a job to a specific agent."""
    job_id: str
    agent_id: str
    agent_name: str
    assigned_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Assignment details
    is_compatible: bool = Field(description="Whether agent meets experiment requirements")
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
    """Response with agent compatibility information."""
    job_assignments: Dict[str, List[AgentCompatibility]] = Field(default_factory=dict, description="job_id -> list of agents with compatibility details")
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
    current_job_id: Optional[str] = Field(default=None, description="Currently executing job ID")
    current_experiment_id: Optional[str] = Field(default=None, description="Currently executing experiment ID")
    uptime_seconds: float = Field(description="Agent uptime in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Current timestamp")
    available_results: List[Dict[str, Any]] = Field(default_factory=list, description="List of available job results")


class HealthStatus(BaseModel):
    """Health check response."""
    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    uptime_seconds: float
    current_jobs: int = 0
    total_completed_jobs: int = 0
    last_job_completed: Optional[datetime] = None
    issues: List[str] = Field(default_factory=list)


# Agent Capability Detection Models
class SystemDetectionResult(BaseModel):
    """System information detection result."""
    hostname: str
    os: OSType
    os_version: str
    kernel: str
    python_version: str
    total_memory_gb: float = Field(ge=0)
    available_memory_gb: float = Field(ge=0)
    disk_total_gb: float = Field(ge=0)
    disk_available_gb: float = Field(ge=0)


class CPUDetectionResult(BaseModel):
    """CPU information detection result."""
    model: str
    cores_physical: int = Field(ge=0)
    cores_logical: int = Field(ge=0)
    frequency_mhz: float = Field(ge=0)
    architecture: CPUArchitecture


class GPUDetectionResult(BaseModel):
    """Comprehensive GPU detection result."""
    index: int = Field(ge=0)
    name: str
    uuid: Optional[str] = None
    vendor: GPUVendor = GPUVendor.UNKNOWN
    memory_total_mb: int = Field(ge=0)
    memory_available_mb: int = Field(ge=0)
    memory_used_mb: Optional[int] = Field(default=None, ge=0)
    
    # CUDA/Compute capabilities
    compute_capability: Optional[str] = None
    tensor_cores: List[TensorCoreGeneration] = Field(default_factory=list)
    supported_precisions: List[PrecisionType] = Field(default_factory=list)
    
    # Live metrics (when available)
    current_utilization_percent: Optional[float] = Field(default=None, ge=0, le=100)
    memory_utilization_percent: Optional[float] = Field(default=None, ge=0, le=100) 
    temperature_celsius: Optional[float] = Field(default=None, ge=0)
    power_usage_watts: Optional[float] = Field(default=None, ge=0)
    
    # Detection metadata
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    multiprocessor_count: Optional[int] = Field(default=None, ge=0)
    detection_method: DetectionMethod = DetectionMethod.UNKNOWN


class FrameworkCapabilities(BaseModel):
    """Framework-specific capability information."""
    text_generation: bool = False
    tokenization: bool = False
    streaming: bool = False
    continuous_batching: bool = False
    cuda: bool = False
    mps: bool = False  # Apple Silicon


class FrameworkDetectionResult(BaseModel):
    """ML framework detection result."""
    name: FrameworkName
    version: str
    available: bool
    capabilities: FrameworkCapabilities = Field(default_factory=FrameworkCapabilities)


class HookDetectionResult(BaseModel):
    """System monitoring hook detection result."""
    name: str
    type: HookType
    executable_path: str
    version: Optional[str] = None
    working: bool


class MetricDetectionResult(BaseModel):
    """Available metric detection result."""
    name: str
    type: MetricType
    available: bool
    collection_method: MetricCollectionMethod
    requires: List[str] = Field(default_factory=list, description="Required tools/capabilities")
    unit: Optional[str] = None
    description: Optional[str] = None


class AgentCapabilityDetectionResult(BaseModel):
    """Complete agent capability detection result container."""
    agent_id: str
    detection_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Core system information
    system: SystemDetectionResult
    cpu: CPUDetectionResult
    gpus: List[GPUDetectionResult] = Field(default_factory=list)
    
    # Software capabilities
    frameworks: List[FrameworkDetectionResult] = Field(default_factory=list)
    models: List[Dict[str, Any]] = Field(default_factory=list, description="Available models (legacy dict format for now)")
    
    # Monitoring and metrics
    hooks: List[HookDetectionResult] = Field(default_factory=list)
    metrics: List[MetricDetectionResult] = Field(default_factory=list)
    
    # Summary information (computed from other fields)
    capabilities_summary: Dict[str, Any] = Field(default_factory=dict)
    
    def compute_summary_fields(self):
        """Compute derived summary fields from detection results."""
        # Compute total GPUs
        total_gpus = len(self.gpus)
        
        # Compute total GPU memory
        total_gpu_memory_mb = sum(gpu.memory_total_mb for gpu in self.gpus)
        
        # Compute available frameworks
        frameworks_available = [f.name for f in self.frameworks if f.available]
        
        # Update capabilities summary
        self.capabilities_summary.update({
            "total_gpus": total_gpus,
            "total_gpu_memory_mb": total_gpu_memory_mb,
            "frameworks_available": frameworks_available,
            "frameworks_count": len(self.frameworks),
            "models_count": len(self.models),
            "hooks_count": len(self.hooks),
            "metrics_count": len(self.metrics)
        })