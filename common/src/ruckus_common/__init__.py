"""RUCKUS Common - Shared components for server and agent."""

__version__ = "0.1.0"

from .models import (
    AgentType,
    JobStatus,
    JobStage,
    MetricType,
    ExperimentSpec,
    JobSpec,
    JobRequest,
    JobUpdate,
    JobResult,
    AgentCapabilitiesBase,
)

from .protocol import (
    MessageType,
    Message,
    Request,
    Response,
    ErrorResponse,
)

from .constants import (
    API_VERSION,
    DEFAULT_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    FRAMEWORKS,
    TASK_TYPES,
)

__all__ = [
    # Models
    "AgentType",
    "JobStatus",
    "JobStage",
    "MetricType",
    "ExperimentSpec",
    "JobSpec",
    "JobRequest",
    "JobUpdate",
    "JobResult",
    "AgentCapabilitiesBase",
    # Protocol
    "MessageType",
    "Message",
    "Request",
    "Response",
    "ErrorResponse",
    # Constants
    "API_VERSION",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    "FRAMEWORKS",
    "TASK_TYPES",
]