"""Shared data models."""

from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime


class Experiment(BaseModel):
    """Experiment model."""
    id: str
    name: str
    description: Optional[str] = None
    config: Dict[str, Any]
    created_at: datetime
    status: str


class Job(BaseModel):
    """Job model."""
    id: str
    experiment_id: str
    agent_id: str
    status: str
    config: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class Agent(BaseModel):
    """Agent model."""
    id: str
    endpoint: str
    capabilities: Dict[str, Any]
    status: str
    last_seen: datetime
