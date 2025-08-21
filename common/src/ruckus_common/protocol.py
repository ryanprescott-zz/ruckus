"""RUCKUS wire protocol definitions."""

from pydantic import BaseModel
from typing import Dict, Any, Optional
from enum import Enum


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobRequest(BaseModel):
    """Job request message."""
    job_id: str
    experiment_id: str
    model_name: str
    task_config: Dict[str, Any]


class JobResponse(BaseModel):
    """Job response message."""
    job_id: str
    status: JobStatus
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AgentRegistration(BaseModel):
    """Agent registration message."""
    agent_id: str
    capabilities: Dict[str, Any]
    endpoint: str
