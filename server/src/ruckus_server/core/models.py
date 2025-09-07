"""Server-side Pydantic models."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

from ruckus_common.models import (
    ExperimentSpec,
    AgentType,
    AgentInfo,
    RegisteredAgentInfo,
)



class ExperimentStatus(BaseModel):
    """Experiment execution status."""
    experiment_id: str
    status: str
    progress: float
    estimated_remaining_seconds: Optional[int]
    started_at: Optional[datetime]
    updated_at: datetime


