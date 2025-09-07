"""Base storage interface and models for RUCKUS server."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from sqlalchemy import Column, String, DateTime, Text, Integer, Boolean, JSON
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class ExperimentAlreadyExistsException(Exception):
    """Exception raised when attempting to create an experiment that already exists."""
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        super().__init__(f"Experiment {experiment_id} already exists")


class ExperimentNotFoundException(Exception):
    """Exception raised when attempting to access an experiment that doesn't exist."""
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        super().__init__(f"Experiment {experiment_id} not found")




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
    
    id = Column(String, primary_key=True)  # This is the experiment_id from ExperimentSpec
    name = Column(String)
    description = Column(Text)
    spec_data = Column(JSON)  # Complete ExperimentSpec serialized as JSON
    status = Column(String, default="created")
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Job(Base):
    """Job database model."""
    __tablename__ = "jobs"
    
    job_id = Column(String, primary_key=True)
    experiment_id = Column(String, nullable=False)
    agent_id = Column(String, nullable=False)
    status = Column(String, nullable=False)  # "queued", "assigned", "running", "completed", "failed", "cancelled"
    status_message = Column(Text)
    created_time = Column(DateTime, default=func.now())
    started_time = Column(DateTime, nullable=True)
    completed_time = Column(DateTime, nullable=True)
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
    async def get_agent(self, agent_id: str):
        """Get agent by ID.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            RegisteredAgentInfo object or None if not found
        """
        pass
    
    @abstractmethod
    async def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent."""
        pass
    
    # Experiment management
    @abstractmethod
    async def create_experiment(self, experiment_spec) -> Dict[str, Any]:
        """Create a new experiment from ExperimentSpec.
        
        Args:
            experiment_spec: ExperimentSpec object containing experiment details
            
        Returns:
            Dict containing the created experiment with 'experiment_id' and 'created_at'
            
        Raises:
            ExperimentAlreadyExistsException: If experiment with same ID already exists
        """
        pass
    
    @abstractmethod
    async def update_experiment_status(self, experiment_id: str, status: str) -> bool:
        """Update experiment status."""
        pass
    
    @abstractmethod
    async def get_experiment(self, experiment_id: str):
        """Get experiment by ID.
        
        Args:
            experiment_id: ID of the experiment to retrieve
            
        Returns:
            ExperimentSpec object
            
        Raises:
            ExperimentNotFoundException: If experiment with given ID doesn't exist
        """
        pass
    
    @abstractmethod
    async def list_experiments(self):
        """List all experiments.
        
        Returns:
            List of ExperimentSpec objects
        """
        pass
    
    @abstractmethod
    async def delete_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Delete an experiment by ID.
        
        Args:
            experiment_id: ID of the experiment to delete
            
        Returns:
            Dict containing the deleted experiment's ID and deletion timestamp
            
        Raises:
            ExperimentNotFoundException: If experiment with given ID doesn't exist
        """
        pass
    
    # Job management
    @abstractmethod
    async def get_running_job(self, agent_id: str):
        """Get the currently running job for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            JobInfo object or None if no running job
        """
        pass
    
    @abstractmethod
    async def set_running_job(self, agent_id: str, job_info):
        """Set the running job for an agent.
        
        Args:
            agent_id: ID of the agent
            job_info: JobInfo object
        """
        pass
    
    @abstractmethod
    async def clear_running_job(self, agent_id: str):
        """Clear the running job for an agent.
        
        Args:
            agent_id: ID of the agent
        """
        pass
    
    @abstractmethod
    async def update_running_job(self, agent_id: str, job_info):
        """Update the running job for an agent.
        
        Args:
            agent_id: ID of the agent
            job_info: JobInfo object
        """
        pass
    
    @abstractmethod
    async def get_queued_jobs(self, agent_id: str):
        """Get queued jobs for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of JobInfo objects
        """
        pass
    
    @abstractmethod
    async def add_queued_job(self, agent_id: str, job_info):
        """Add a job to the queue for an agent.
        
        Args:
            agent_id: ID of the agent
            job_info: JobInfo object
        """
        pass
    
    @abstractmethod
    async def remove_queued_job(self, agent_id: str, job_id: str):
        """Remove a job from the queue for an agent.
        
        Args:
            agent_id: ID of the agent
            job_id: ID of the job to remove
        """
        pass
    
    @abstractmethod
    async def get_completed_jobs(self, agent_id: str):
        """Get completed jobs for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of JobInfo objects
        """
        pass
    
    @abstractmethod
    async def add_completed_job(self, agent_id: str, job_info):
        """Add a job to the completed jobs for an agent.
        
        Args:
            agent_id: ID of the agent
            job_info: JobInfo object
        """
        pass
    
    @abstractmethod
    async def get_failed_jobs(self, agent_id: str):
        """Get failed jobs for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of JobInfo objects
        """
        pass
    
    @abstractmethod
    async def add_failed_job(self, agent_id: str, job_info):
        """Add a job to the failed jobs for an agent.
        
        Args:
            agent_id: ID of the agent
            job_info: JobInfo object
        """
        pass
    
    @abstractmethod
    async def save_experiment_results(self, experiment_id: str, results: Dict[str, Any]):
        """Save experiment results.
        
        Args:
            experiment_id: ID of the experiment
            results: Results dictionary
        """
        pass

    @abstractmethod
    async def store_experiment_result(self, experiment_result):
        """Store an ExperimentResult object.
        
        Args:
            experiment_result: ExperimentResult object to store
        """
        pass

    @abstractmethod
    async def get_experiment_result_by_job_id(self, job_id: str):
        """Get ExperimentResult by job ID.
        
        Args:
            job_id: Job ID to get result for
            
        Returns:
            ExperimentResult object or None if not found
        """
        pass

    @abstractmethod
    async def list_experiment_results(self):
        """Get all stored experiment results.
        
        Returns:
            List of ExperimentResult objects
        """
        pass
    
