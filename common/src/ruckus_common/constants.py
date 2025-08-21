"""System-wide constants for RUCKUS."""

# Version Information
API_VERSION = "v1"
PROTOCOL_VERSION = "1.0.0"
SYSTEM_VERSION = "0.1.0"

# Timeouts (seconds)
DEFAULT_TIMEOUT = 3600  # 1 hour
DEFAULT_HEARTBEAT_INTERVAL = 30
DEFAULT_REGISTRATION_TIMEOUT = 60
DEFAULT_JOB_TIMEOUT = 3600
MAX_JOB_TIMEOUT = 86400  # 24 hours

# Retries
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 2.0  # Exponential backoff multiplier

# Limits
MAX_CONCURRENT_JOBS = 10
MAX_BATCH_SIZE = 32
MAX_MODELS_CACHED = 5
MAX_PAYLOAD_SIZE_MB = 100

# Supported Frameworks
FRAMEWORKS = [
    "transformers",
    "vllm",
    "pytorch",
    "tensorflow",
    "onnx",
    "tensorrt",
]

# Supported Task Types
TASK_TYPES = [
    "summarization",
    "classification",
    "generation",
    "question_answering",
    "translation",
    "custom",
]

# Model Formats
MODEL_FORMATS = [
    "pytorch",
    "safetensors",
    "onnx",
    "tensorflow",
    "gguf",
    "ggml",
]

# Quantization Types
QUANTIZATION_TYPES = [
    "fp32",
    "fp16",
    "bf16",
    "int8",
    "int4",
    "nf4",  # QLoRA
]

# Default Paths
DEFAULT_MODEL_CACHE_DIR = "/models"
DEFAULT_DATA_DIR = "/data"
DEFAULT_RESULTS_DIR = "/results"
DEFAULT_LOG_DIR = "/logs"

# Search paths for models
MODEL_SEARCH_PATHS = [
    "/models",
    "./models",
    "~/.cache/huggingface/hub",
    "/opt/models",
    "/usr/local/models",
]

# Environment Variable Names
ENV_ORCHESTRATOR_URL = "RUCKUS_ORCHESTRATOR_URL"
ENV_AGENT_ID = "RUCKUS_AGENT_ID"
ENV_AGENT_TYPE = "RUCKUS_AGENT_TYPE"
ENV_MODEL_CACHE = "RUCKUS_MODEL_CACHE"
ENV_LOG_LEVEL = "RUCKUS_LOG_LEVEL"
ENV_API_TOKEN = "RUCKUS_API_TOKEN"

# HTTP Headers
HEADER_AGENT_ID = "X-Ruckus-Agent-Id"
HEADER_JOB_ID = "X-Ruckus-Job-Id"
HEADER_API_VERSION = "X-Ruckus-Api-Version"

# Status Codes (Custom)
STATUS_JOB_ACCEPTED = 202
STATUS_JOB_REJECTED = 409
STATUS_AGENT_BUSY = 503

# Metric Names (Standardized)
METRIC_LATENCY_MS = "latency_ms"
METRIC_THROUGHPUT_TPS = "throughput_tokens_per_second"
METRIC_TIME_TO_FIRST_TOKEN = "time_to_first_token_ms"
METRIC_MEMORY_USED_MB = "memory_used_mb"
METRIC_GPU_MEMORY_MB = "gpu_memory_mb"
METRIC_GPU_UTILIZATION = "gpu_utilization_percent"
METRIC_MODEL_LOAD_TIME = "model_load_time_seconds"

# Quality Metrics
METRIC_ROUGE_1 = "rouge_1"
METRIC_ROUGE_2 = "rouge_2"
METRIC_ROUGE_L = "rouge_l"
METRIC_BLEU = "bleu"
METRIC_PERPLEXITY = "perplexity"

# Logging Formats
LOG_FORMAT_JSON = "json"
LOG_FORMAT_TEXT = "text"
DEFAULT_LOG_FORMAT = LOG_FORMAT_JSON

# Database
DEFAULT_DB_URL = "sqlite:///./ruckus.db"
DB_POOL_SIZE = 10
DB_MAX_OVERFLOW = 20

# Feature Flags
ENABLE_GPU_MONITORING = True
ENABLE_PROFILING = False
ENABLE_TRACING = False
ENABLE_METRICS_EXPORT = True

# Error Codes
ERROR_AGENT_NOT_FOUND = "E001"
ERROR_MODEL_NOT_FOUND = "E002"
ERROR_FRAMEWORK_NOT_SUPPORTED = "E003"
ERROR_RESOURCE_EXHAUSTED = "E004"
ERROR_TIMEOUT = "E005"
ERROR_INVALID_CONFIG = "E006"
ERROR_AUTHENTICATION_FAILED = "E007"