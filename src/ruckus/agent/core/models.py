"""
Pydantic models for the Ruckus Agent API.

This module defines the data models used for API requests and responses
in the agent service.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class JobExecutionStatus(str, Enum):
    """Enumeration of job execution statuses."""
    RECEIVED = "received"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentCapabilities(BaseModel):
    """Model for agent capabilities."""
    runtime: str = Field(..., description="LLM runtime (e.g., transformers, vllm)")
    platform: str = Field(..., description="Platform (e.g., cuda, cpu)")
    max_memory: Optional[int] = Field(None, description="Maximum memory in GB")
    gpu_count: Optional[int] = Field(None, description="Number of GPUs")
    gpu_type: Optional[str] = Field(None, description="GPU type")


class AgentStatus(BaseModel):
    """Model for agent status information."""
    agent_id: UUID = Field(..., description="Agent identifier")
    name: str = Field(..., description="Agent name")
    status: str = Field(..., description="Current status")
    capabilities: AgentCapabilities = Field(..., description="Agent capabilities")
    current_jobs: int = Field(default=0, description="Number of current jobs")
    total_jobs_completed: int = Field(default=0, description="Total completed jobs")
    last_heartbeat: Optional[datetime] = Field(None, description="Last heartbeat timestamp")


class JobExecution(BaseModel):
    """Model for job execution information."""
    job_id: UUID = Field(..., description="Job identifier")
    experiment_id: UUID = Field(..., description="Experiment identifier")
    status: JobExecutionStatus = Field(..., description="Execution status")
    config: Dict[str, Any] = Field(..., description="Job configuration")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    results: Optional[Dict[str, Any]] = Field(None, description="Job results")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class JobRequest(BaseModel):
    """Model for job execution requests."""
    job_id: UUID = Field(..., description="Job identifier")
    experiment_id: UUID = Field(..., description="Experiment identifier")
    config: Dict[str, Any] = Field(..., description="Job configuration")
    model_name: str = Field(..., description="Model name")
    runtime: str = Field(..., description="Runtime")
    platform: str = Field(..., description="Platform")
    task_config: Dict[str, Any] = Field(..., description="Task configuration")
    data_config: Dict[str, Any] = Field(..., description="Data configuration")


class JobResult(BaseModel):
    """Model for job execution results."""
    job_id: UUID = Field(..., description="Job identifier")
    status: JobExecutionStatus = Field(..., description="Execution status")
    results: Optional[Dict[str, Any]] = Field(None, description="Job results")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")


class HealthCheck(BaseModel):
    """Model for health check responses."""
    status: str = Field(..., description="Health status")
    agent_id: Optional[UUID] = Field(None, description="Agent identifier")
    uptime: float = Field(..., description="Uptime in seconds")
    current_jobs: int = Field(..., description="Number of current jobs")
    system_info: Dict[str, Any] = Field(..., description="System information")
