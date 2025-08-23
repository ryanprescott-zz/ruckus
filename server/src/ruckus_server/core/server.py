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
    
    async def register_agent(self, agent_id: str, capabilities: dict) -> bool:
        """Register a new agent with the server.
        
        Args:
            agent_id: Unique identifier for the agent.
            capabilities: Agent capabilities and hardware profile.
            
        Returns:
            True if registration successful, False otherwise.
        """
        self.logger.info(f"Registering agent {agent_id}")
        
        if not self.storage:
            self.logger.error("Storage backend not initialized")
            return False
        
        # TODO: Validate agent capabilities
        success = await self.storage.register_agent(agent_id, capabilities)
        
        if success:
            self.logger.info(f"Agent {agent_id} registered successfully")
        else:
            self.logger.error(f"Failed to register agent {agent_id}")
        
        return success
    
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
