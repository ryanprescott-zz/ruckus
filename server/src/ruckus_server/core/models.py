"""Server-side Pydantic models."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

from ruckus_common.models import (
    ExperimentSpec,
    JobSpec,
    JobStatus,
    AgentType,
)


class AgentInfo(BaseModel):
    """Agent information for server tracking."""
    agent_id: str
    agent_type: AgentType
    hostname: str
    ip_address: str
    status: str = "offline"
    capabilities: Dict[str, bool]
    frameworks: List[str]
    hardware_info: Dict[str, Any]
    last_heartbeat: datetime
    current_jobs: List[str] = Field(default_factory=list)
    total_completed: int = 0
    total_failed: int = 0


class ExperimentStatus(BaseModel):
    """Experiment execution status."""
    experiment_id: str
    status: str
    progress: float
    jobs_total: int
    jobs_queued: int
    jobs_running: int
    jobs_completed: int
    jobs_failed: int
    estimated_remaining_seconds: Optional[int]
    started_at: Optional[datetime]
    updated_at: datetime


class JobAssignment(BaseModel):
    """Job assignment to agent."""
    job_id: str
    agent_id: str
    assigned_at: datetime
    timeout_at: datetime
    priority: int = 0


class SchedulerState(BaseModel):
    """Scheduler state information."""
    total_agents: int
    active_agents: int
    total_jobs: int
    queued_jobs: int
    running_jobs: int
    completed_jobs: int
    failed_jobs: int
    scheduler_running: bool
    last_schedule_time: Optional[datetime]