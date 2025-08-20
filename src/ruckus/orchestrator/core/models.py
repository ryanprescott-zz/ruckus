"""
Pydantic models for the Ruckus Orchestrator API.

This module defines the data models used for API requests and responses,
including models for experiments, jobs, agents, and related entities.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Enumeration of possible job statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(str, Enum):
    """Enumeration of possible agent statuses."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"


class ExperimentBase(BaseModel):
    """Base model for experiment data."""
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    model_name: str = Field(..., description="LLM model name")
    runtime: str = Field(..., description="LLM runtime (e.g., vllm, transformers)")
    platform: str = Field(..., description="Platform (e.g., cuda, cpu)")
    task_config: Dict[str, Any] = Field(..., description="Task configuration")
    data_config: Dict[str, Any] = Field(..., description="Data configuration")


class ExperimentCreate(ExperimentBase):
    """Model for creating a new experiment."""
    pass


class ExperimentUpdate(BaseModel):
    """Model for updating an experiment."""
    name: Optional[str] = None
    description: Optional[str] = None
    model_name: Optional[str] = None
    runtime: Optional[str] = None
    platform: Optional[str] = None
    task_config: Optional[Dict[str, Any]] = None
    data_config: Optional[Dict[str, Any]] = None


class Experiment(ExperimentBase):
    """Complete experiment model with database fields."""
    id: UUID = Field(default_factory=uuid4, description="Experiment ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

    class Config:
        from_attributes = True


class JobBase(BaseModel):
    """Base model for job data."""
    experiment_id: UUID = Field(..., description="Associated experiment ID")
    agent_id: Optional[UUID] = Field(None, description="Assigned agent ID")
    status: JobStatus = Field(default=JobStatus.PENDING, description="Job status")
    config: Dict[str, Any] = Field(..., description="Job configuration")


class JobCreate(JobBase):
    """Model for creating a new job."""
    pass


class JobUpdate(BaseModel):
    """Model for updating a job."""
    agent_id: Optional[UUID] = None
    status: Optional[JobStatus] = None
    config: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None


class Job(JobBase):
    """Complete job model with database fields."""
    id: UUID = Field(default_factory=uuid4, description="Job ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    results: Optional[Dict[str, Any]] = Field(None, description="Job results")

    class Config:
        from_attributes = True


class AgentBase(BaseModel):
    """Base model for agent data."""
    name: str = Field(..., description="Agent name")
    host: str = Field(..., description="Agent host address")
    port: int = Field(..., description="Agent port")
    capabilities: Dict[str, Any] = Field(..., description="Agent capabilities")
    status: AgentStatus = Field(default=AgentStatus.OFFLINE, description="Agent status")


class AgentCreate(AgentBase):
    """Model for creating a new agent."""
    pass


class AgentUpdate(BaseModel):
    """Model for updating an agent."""
    name: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    capabilities: Optional[Dict[str, Any]] = None
    status: Optional[AgentStatus] = None


class Agent(AgentBase):
    """Complete agent model with database fields."""
    id: UUID = Field(default_factory=uuid4, description="Agent ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    last_heartbeat: Optional[datetime] = Field(None, description="Last heartbeat timestamp")

    class Config:
        from_attributes = True


class ExperimentList(BaseModel):
    """Model for paginated experiment list response."""
    experiments: List[Experiment]
    total: int
    page: int
    size: int


class JobList(BaseModel):
    """Model for paginated job list response."""
    jobs: List[Job]
    total: int
    page: int
    size: int


class AgentList(BaseModel):
    """Model for paginated agent list response."""
    agents: List[Agent]
    total: int
    page: int
    size: int
