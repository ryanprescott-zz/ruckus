"""RUCKUS server core implementation."""

import asyncio
import logging
import logging.config
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .config import RuckusServerSettings, StorageBackendType
from .storage.factory import StorageFactory
from .storage.base import StorageBackend
from .models import RegisteredAgentInfo
from .agent import AgentProtocolUtility
from .clients.http import ConnectionError, ServiceUnavailableError


class AgentAlreadyRegisteredException(Exception):
    """Exception raised when attempting to register an agent that is already registered."""
    
    def __init__(self, agent_id: str, registered_at: str):
        self.agent_id = agent_id
        self.registered_at = registered_at
        super().__init__(f"Agent {agent_id} is already registered (registered at {registered_at})")


class AgentNotRegisteredException(Exception):
    """Exception raised when attempting to unregister an agent that is not registered."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        super().__init__(f"Agent {agent_id} is not registered")


class RuckusServer:
    """RUCKUS server implementation.
    
    The central brain of RUCKUS, responsible for managing experiments,
    coordinating agents, scheduling jobs, and aggregating results.
    """
    
    def __init__(self, settings: Optional[RuckusServerSettings] = None):
        """Initialize the RUCKUS server.
        
        Args:
            settings: Server configuration settings. If None, will load from environment.
        """
        self.settings = settings or RuckusServerSettings()
        self.logger = self._setup_logging()
        self.storage: Optional[StorageBackend] = None
        
        self.logger.info("RUCKUS server initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration.
        
        Returns:
            Configured logger instance.
        """
        # Try to load logging config from YAML file
        log_config_path = Path(__file__).parent.parent / self.settings.log_config_file
        
        if log_config_path.exists():
            try:
                with open(log_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logging.config.dictConfig(config)
            except Exception as e:
                # Fallback to basic logging if config file fails
                logging.basicConfig(
                    level=getattr(logging, self.settings.log_level.upper()),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                logging.getLogger(__name__).warning(
                    f"Failed to load logging config from {log_config_path}: {e}"
                )
        else:
            # Basic logging configuration
            logging.basicConfig(
                level=getattr(logging, self.settings.log_level.upper()),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        return logging.getLogger(__name__)
    
    async def start(self) -> None:
        """Start the RUCKUS server backend.
        
        Initializes the database connection and starts background tasks
        for agent monitoring and job scheduling.
        """
        self.logger.info("Starting RUCKUS server backend...")
        
        # Setup database connection
        await self._setup_database()
        
        # Start background tasks
        await self._start_background_tasks()
        
        self.logger.info("RUCKUS server backend started")
    
    async def stop(self) -> None:
        """Stop the RUCKUS server backend.
        
        Gracefully shuts down background tasks and closes database connections.
        """
        self.logger.info("Stopping RUCKUS server backend...")
        
        # Stop background tasks
        await self._stop_background_tasks()
        
        # Close database connections
        await self._cleanup_database()
        
        self.logger.info("RUCKUS server backend stopped")
    
    async def _setup_database(self) -> None:
        """Setup database connection and initialize schema."""
        self.logger.info("Setting up storage backend...")
        
        # Import storage backends
        from .storage.postgresql import PostgreSQLStorageBackend
        from .storage.sqlite import SQLiteStorageBackend
        
        # Create storage backend based on configuration using the embedded settings
        if self.settings.storage_backend == StorageBackendType.POSTGRESQL:
            self.storage = PostgreSQLStorageBackend(self.settings.postgresql)
        elif self.settings.storage_backend == StorageBackendType.SQLITE:
            self.storage = SQLiteStorageBackend(self.settings.sqlite)
        else:
            raise ValueError(f"Unsupported storage backend: {self.settings.storage_backend}")
        
        # Initialize the storage backend
        await self.storage.initialize()
        
        self.logger.info(f"Storage backend ({self.settings.storage_backend}) initialized successfully")
    
    async def _cleanup_database(self) -> None:
        """Cleanup database connections."""
        self.logger.info("Cleaning up storage backend...")
        if self.storage:
            await self.storage.close()
    
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks for agent monitoring and job scheduling."""
        self.logger.info("Starting background tasks...")
        
        # TODO: Start agent heartbeat monitoring task
        # TODO: Start job scheduling task
        # TODO: Start result aggregation task
        pass
    
    async def _stop_background_tasks(self) -> None:
        """Stop background tasks."""
        self.logger.info("Stopping background tasks...")
        # TODO: Implement graceful shutdown of background tasks
        pass
    
    async def register_agent(self, agent_url: str) -> dict:
        """Register a new agent with the server.
        
        Args:
            agent_url: Base URL of the agent to register
            
        Returns:
            Dict with agent_id and registered_at timestamp
            
        Raises:
            ConnectionError: If cannot connect to agent
            ServiceUnavailableError: If agent is unavailable after retries
            ValueError: If agent info is invalid
        """
        
        self.logger.info(f"Registering agent at {agent_url}")
        
        if not self.storage:
            raise RuntimeError("Storage backend not initialized")
        
        # Create agent protocol utility
        agent_util = AgentProtocolUtility(
            self.settings.agent, 
            self.settings.http_client
        )
        
        try:
            # Get agent info from the agent's /info endpoint
            agent_info_response = await agent_util.get_agent_info(agent_url)
            
            # Check if agent is already registered
            agent_id = agent_info_response.agent_info.agent_id
            existing_agent = await self.storage.get_agent(agent_id)
            if existing_agent:
                # Agent already exists, raise conflict exception
                registered_at = existing_agent.get('registered_at')
                registered_at_str = registered_at.isoformat() if registered_at else 'unknown'
                raise AgentAlreadyRegisteredException(
                    agent_id=agent_id,
                    registered_at=registered_at_str
                )
            
            # Create RegisteredAgentInfo object
            registered_info = agent_util.create_registered_agent_info(
                agent_info_response, 
                agent_url
            )
            
            # Store in database
            success = await self.storage.register_agent(registered_info)
            
            if not success:
                raise RuntimeError("Failed to store agent information in database")
            
            self.logger.info(f"Agent {registered_info.agent_info.agent_id} registered successfully")
            return {
                "agent_id": registered_info.agent_info.agent_id,
                "registered_at": registered_info.registered_at
            }
            
        except ConnectionError as e:
            self.logger.error(f"Failed to connect to agent at {agent_url}: {str(e)}")
            raise
        except ServiceUnavailableError as e:
            self.logger.error(f"Agent at {agent_url} is unavailable: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error registering agent at {agent_url}: {str(e)}")
            raise ValueError(f"Failed to register agent: {str(e)}")
    
    async def unregister_agent(self, agent_id: str) -> dict:
        """Unregister an agent from the server.
        
        Args:
            agent_id: ID of the agent to unregister
            
        Returns:
            Dict with agent_id and unregistered_at timestamp
            
        Raises:
            AgentNotRegisteredException: If agent is not registered
            RuntimeError: If storage backend not initialized
        """
        self.logger.info(f"Unregistering agent {agent_id}")
        
        if not self.storage:
            raise RuntimeError("Storage backend not initialized")
        
        try:
            # Check if agent exists
            agent_exists = await self.storage.agent_exists(agent_id)
            if not agent_exists:
                self.logger.warning(f"Attempted to unregister non-existent agent {agent_id}")
                raise AgentNotRegisteredException(agent_id)
            
            # Remove agent from database
            success = await self.storage.remove_agent(agent_id)
            
            if not success:
                raise RuntimeError("Failed to remove agent from database")
            
            from datetime import datetime
            unregistered_at = datetime.utcnow()
            
            self.logger.info(f"Agent {agent_id} unregistered successfully")
            return {
                "agent_id": agent_id,
                "unregistered_at": unregistered_at
            }
            
        except AgentNotRegisteredException:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error unregistering agent {agent_id}: {str(e)}")
            raise RuntimeError(f"Failed to unregister agent: {str(e)}")
    
    async def schedule_job(self, experiment_id: str, job_config: dict) -> Optional[str]:
        """Schedule a job for execution on an appropriate agent.
        
        Args:
            experiment_id: ID of the experiment this job belongs to.
            job_config: Job configuration and requirements.
            
        Returns:
            Job ID if scheduled successfully, None otherwise.
        """
        self.logger.info(f"Scheduling job for experiment {experiment_id}")
        
        if not self.storage:
            self.logger.error("Storage backend not initialized")
            return None
        
        # TODO: Find suitable agent based on job requirements
        # Generate unique job ID
        import uuid
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        success = await self.storage.create_job(job_id, experiment_id, job_config)
        
        if success:
            self.logger.info(f"Job {job_id} scheduled")
            return job_id
        else:
            self.logger.error(f"Failed to schedule job for experiment {experiment_id}")
            return None
    
    async def get_experiment_status(self, experiment_id: str) -> Optional[dict]:
        """Get the status of an experiment.
        
        Args:
            experiment_id: ID of the experiment.
            
        Returns:
            Experiment status dict or None if not found.
        """
        if not self.storage:
            self.logger.error("Storage backend not initialized")
            return None
        
        return await self.storage.get_experiment(experiment_id)
    
    async def get_agent_status(self, agent_id: str) -> Optional[dict]:
        """Get the status of an agent.
        
        Args:
            agent_id: ID of the agent.
            
        Returns:
            Agent status dict or None if not found.
        """
        if not self.storage:
            self.logger.error("Storage backend not initialized")
            return None
        
        return await self.storage.get_agent(agent_id)
    
    async def health_check(self) -> dict:
        """Check the health of the server and its components.
        
        Returns:
            Health status information including storage and agent count.
        """
        if not self.storage:
            return {
                "status": "unhealthy",
                "storage": "not_initialized",
                "agents": 0
            }
        
        storage_healthy = await self.storage.health_check()
        agents = await self.storage.list_agents()
        
        return {
            "status": "healthy" if storage_healthy else "unhealthy",
            "storage": "healthy" if storage_healthy else "unhealthy",
            "agents": len(agents)
        }
