"""Pydantic models for API v1 endpoints."""

from pydantic import BaseModel, Field, field_validator
from urllib.parse import urlparse
import re

from datetime import datetime
from typing import List, Dict, Optional, Any
from ruckus_common.models import RegisteredAgentInfo, AgentStatus, ExperimentSpec, JobResult, JobStatusEnum, AgentCompatibility
from ...core.models import JobInfo


class RegisterAgentRequest(BaseModel):
    """Request model for registering an agent."""
    
    agent_url: str = Field(..., description="Base URL of the agent to register")
    
    @field_validator('agent_url')
    @classmethod
    def validate_agent_url(cls, v: str) -> str:
        """Validate that agent_url is a valid URL."""
        if not v:
            raise ValueError("agent_url cannot be empty")
        
        # Parse the URL
        try:
            parsed = urlparse(v)
        except Exception as e:
            raise ValueError(f"Invalid URL format: {str(e)}")
        
        # Check for required components
        if not parsed.scheme:
            raise ValueError("URL must include a scheme (http or https)")
        
        if parsed.scheme not in ('http', 'https'):
            raise ValueError("URL scheme must be http or https")
        
        if not parsed.netloc:
            raise ValueError("URL must include a hostname")
        
        # Check for valid hostname format
        hostname_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*$'
        hostname = parsed.netloc.split(':')[0]  # Remove port if present
        
        if not re.match(hostname_pattern, hostname) and hostname != 'localhost':
            raise ValueError("Invalid hostname format")
        
        return v


class RegisterAgentResponse(BaseModel):
    """Response model for agent registration."""
    
    agent_id: str = Field(..., description="ID of the registered agent")
    registered_at: datetime = Field(..., description="Timestamp when the agent was registered")


class UnregisterAgentRequest(BaseModel):
    """Request model for unregistering an agent."""
    
    agent_id: str = Field(..., description="ID of the agent to unregister", min_length=1)


class UnregisterAgentResponse(BaseModel):
    """Response model for agent unregistration."""
    
    agent_id: str = Field(..., description="ID of the unregistered agent")
    unregistered_at: datetime = Field(..., description="Timestamp when the agent was unregistered")


class ListAgentInfoResponse(BaseModel):
    """Response model for getting all registered agent information."""
    
    agents: List[RegisteredAgentInfo] = Field(..., description="List of all registered agents")


class GetAgentInfoResponse(BaseModel):
    """Response model for getting specific agent information."""
    
    agent: RegisteredAgentInfo = Field(..., description="Registered agent information")


class ListAgentStatusResponse(BaseModel):
    """Response model for getting all agent status information."""
    
    agents: List[AgentStatus] = Field(..., description="List of agent status information")


class GetAgentStatusResponse(BaseModel):
    """Response model for getting specific agent status information."""
    
    agent: AgentStatus = Field(..., description="Agent status information")


# Experiment-related models
class CreateExperimentRequest(BaseModel):
    """Request model for creating a new experiment."""
    
    experiment_spec: ExperimentSpec = Field(..., description="ExperimentSpec containing experiment details")


class CreateExperimentResponse(BaseModel):
    """Response model for experiment creation."""
    
    experiment_id: str = Field(..., description="ID of the created experiment")
    created_at: datetime = Field(..., description="Timestamp when the experiment was created")


class DeleteExperimentResponse(BaseModel):
    """Response model for experiment deletion."""
    
    experiment_id: str = Field(..., description="ID of the deleted experiment")
    deleted_at: datetime = Field(..., description="Timestamp when the experiment was deleted")


class ExperimentSummary(BaseModel):
    """Summary of an experiment including metadata."""
    
    id: str = Field(..., description="Experiment ID")
    name: str = Field(..., description="Experiment name")
    description: str = Field(None, description="Experiment description")
    spec_data: dict = Field(..., description="Complete ExperimentSpec data as JSON")
    status: str = Field(..., description="Current experiment status")
    created_at: datetime = Field(..., description="Timestamp when experiment was created")
    updated_at: datetime = Field(..., description="Timestamp when experiment was last updated")


class ListExperimentsResponse(BaseModel):
    """Response model for listing experiments."""
    
    experiments: List[ExperimentSpec] = Field(..., description="List of all ExperimentSpec objects")


class GetExperimentResponse(BaseModel):
    """Response model for getting a specific experiment."""
    
    experiment: ExperimentSpec = Field(..., description="Complete ExperimentSpec of the requested experiment")


# Job-related models
class CreateJobRequest(BaseModel):
    """Request model for creating a new job."""
    
    experiment_id: str = Field(..., description="ID of the experiment to run")
    agent_id: str = Field(..., description="ID of the agent to run the job on")


class CreateJobResponse(BaseModel):
    """Response model for job creation."""
    
    job_id: str = Field(..., description="ID of the created job")


class ListJobsResponse(BaseModel):
    """Response model for listing jobs grouped by agent."""
    
    jobs: Dict[str, List[JobInfo]] = Field(..., description="Dictionary of JobInfo lists keyed by agent_id, sorted by timestamp (newest first)")


# Results-related models
class ExperimentResult(BaseModel):
    """Enhanced result combining JobResult with additional display fields."""
    
    # Core identification fields for the UI
    job_id: str = Field(..., description="ID of the job")
    experiment_id: str = Field(..., description="ID of the experiment")
    agent_id: str = Field(..., description="ID of the agent that executed the job")
    status: JobStatusEnum = Field(..., description="Current job status")
    
    # Timing information
    started_at: datetime = Field(..., description="When the job started")
    completed_at: Optional[datetime] = Field(None, description="When the job completed")
    duration_seconds: Optional[float] = Field(None, description="Duration of job execution")
    
    # Results data from JobResult
    output: Optional[Any] = Field(None, description="Job output data")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    
    # Metadata
    model_actual: Optional[str] = Field(None, description="Actual model used")
    framework_version: Optional[str] = Field(None, description="Framework version used")
    hardware_info: Dict[str, Any] = Field(default_factory=dict, description="Hardware information")
    artifacts: List[str] = Field(default_factory=list, description="Generated artifacts")
    
    # Error information
    error: Optional[str] = Field(None, description="Error message if failed")
    error_type: Optional[str] = Field(None, description="Type of error")
    traceback: Optional[str] = Field(None, description="Error traceback")
    
    @classmethod
    def from_job_result(cls, job_result: JobResult, agent_id: str) -> "ExperimentResult":
        """Create ExperimentResult from JobResult and agent_id."""
        return cls(
            job_id=job_result.job_id,
            experiment_id=job_result.experiment_id,
            agent_id=agent_id,
            status=job_result.status,
            started_at=job_result.started_at,
            completed_at=job_result.completed_at,
            duration_seconds=job_result.duration_seconds,
            output=job_result.output,
            metrics=job_result.metrics,
            model_actual=job_result.model_actual,
            framework_version=job_result.framework_version,
            hardware_info=job_result.hardware_info,
            artifacts=job_result.artifacts,
            error=job_result.error,
            error_type=job_result.error_type,
            traceback=job_result.traceback,
        )


class ListExperimentResultsResponse(BaseModel):
    """Response model for listing experiment results."""
    
    results: List[ExperimentResult] = Field(..., description="List of completed experiment results")


class GetExperimentResultResponse(BaseModel):
    """Response model for getting a specific experiment result by job_id."""
    
    result: ExperimentResult = Field(..., description="Experiment result for the specified job")


class CheckAgentCompatibilityRequest(BaseModel):
    """Request to check agent compatibility for an experiment type."""
    
    experiment_spec: ExperimentSpec = Field(..., description="Experiment specification to check compatibility against")
    agent_ids: Optional[List[str]] = Field(default=None, description="Specific agent IDs to check (if None, checks all agents)")


class CheckAgentCompatibilityResponse(BaseModel):
    """Response with agent compatibility information."""
    
    compatibility_results: List[AgentCompatibility] = Field(..., description="Compatibility results for each agent")
    experiment_name: str = Field(..., description="Name of the experiment that was checked")
    total_agents_checked: int = Field(..., description="Total number of agents evaluated")
    compatible_agents_count: int = Field(..., description="Number of agents that can run this experiment")
    checked_at: datetime = Field(default_factory=datetime.utcnow, description="When the compatibility check was performed")