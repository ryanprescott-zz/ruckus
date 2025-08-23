"""Base storage interface and models for RUCKUS server."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from sqlalchemy import Column, String, DateTime, Text, Integer, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class StorageBackendType(str, Enum):
    """Supported storage backend types."""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"


class Agent(Base):
    """Agent database model."""
    __tablename__ = "agents"
    
    id = Column(String, primary_key=True)  # This is the agent_id
    agent_name = Column(String)
    agent_type = Column(String)
    agent_url = Column(String, nullable=False)
    system_info = Column(JSON, default=dict)
    capabilities = Column(JSON, default=dict)
    status = Column(String, default="active")
    last_heartbeat = Column(DateTime)
    last_updated = Column(DateTime)
    registered_at = Column(DateTime, default=func.now())
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Experiment(Base):
    """Experiment database model."""
    __tablename__ = "experiments"
    
    id = Column(String, primary_key=True)
    name = Column(String)
    description = Column(Text)
    config = Column(JSON)
    status = Column(String, default="created")
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Job(Base):
    """Job database model."""
    __tablename__ = "jobs"
    
    id = Column(String, primary_key=True)
    experiment_id = Column(String)
    agent_id = Column(String)
    config = Column(JSON)
    status = Column(String, default="scheduled")
    results = Column(JSON)
    error_message = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend and create tables if needed."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend connections."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the storage backend is healthy."""
        pass
    
    # Agent management
    @abstractmethod
    async def register_agent(self, agent_info) -> bool:
        """Register a new agent with full information.
        
        Args:
            agent_info: RegisteredAgentInfo object containing all agent details
            
        Returns:
            True if registration successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def update_agent_status(self, agent_id: str, status: str) -> bool:
        """Update agent status."""
        pass
    
    @abstractmethod
    async def update_agent_heartbeat(self, agent_id: str) -> bool:
        """Update agent last heartbeat timestamp."""
        pass
    
    @abstractmethod
    async def agent_exists(self, agent_id: str) -> bool:
        """Check if an agent exists by ID.
        
        Args:
            agent_id: ID of the agent to check
            
        Returns:
            True if agent exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_registered_agent_info(self, agent_id: str):
        """Get registered agent info by ID.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            RegisteredAgentInfo object or None if not found
        """
        pass
    
    @abstractmethod
    async def list_registered_agent_info(self):
        """List all registered agent info.
        
        Returns:
            List of RegisteredAgentInfo objects
        """
        pass
    
    @abstractmethod
    async def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent."""
        pass
    
    # Experiment management
    @abstractmethod
    async def create_experiment(self, experiment_id: str, name: str, 
                              description: str, config: Dict[str, Any]) -> bool:
        """Create a new experiment."""
        pass
    
    @abstractmethod
    async def update_experiment_status(self, experiment_id: str, status: str) -> bool:
        """Update experiment status."""
        pass
    
    @abstractmethod
    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by ID."""
        pass
    
    @abstractmethod
    async def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        pass
    
    @abstractmethod
    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        pass
    
    # Job management
    @abstractmethod
    async def create_job(self, job_id: str, experiment_id: str, 
                        config: Dict[str, Any]) -> bool:
        """Create a new job."""
        pass
    
    @abstractmethod
    async def assign_job_to_agent(self, job_id: str, agent_id: str) -> bool:
        """Assign a job to an agent."""
        pass
    
    @abstractmethod
    async def update_job_status(self, job_id: str, status: str, 
                               results: Optional[Dict[str, Any]] = None,
                               error_message: Optional[str] = None) -> bool:
        """Update job status and results."""
        pass
    
    @abstractmethod
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID."""
        pass
    
    @abstractmethod
    async def list_jobs(self, experiment_id: Optional[str] = None,
                       agent_id: Optional[str] = None,
                       status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List jobs with optional filtering."""
        pass
    
    @abstractmethod
    async def get_jobs_for_agent(self, agent_id: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get jobs assigned to a specific agent."""
        pass
