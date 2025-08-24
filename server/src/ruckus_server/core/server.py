"""RUCKUS server core implementation."""

import asyncio
import logging
import logging.config
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from urllib.parse import urljoin

import yaml

from .config import RuckusServerSettings, StorageBackendType
from .storage.factory import StorageFactory
from .storage.base import StorageBackend
from ruckus_common.models import RegisteredAgentInfo, AgentStatus, AgentStatusEnum
from .agent import AgentProtocolUtility
from .clients.http import ConnectionError, ServiceUnavailableError
from .clients.simple_http import SimpleHttpClient


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
            existing_agent = await self.storage.get_registered_agent_info(agent_id)
            if existing_agent:
                # Agent already exists, raise conflict exception
                registered_at_str = existing_agent.registered_at.isoformat() if existing_agent.registered_at else 'unknown'
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
            
            self.logger.info(f"Agent {registered_info.agent_id} registered successfully")
            return {
                "agent_id": registered_info.agent_id,
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
    
    async def list_registered_agent_info(self) -> List[RegisteredAgentInfo]:
        """Get all registered agent information.
        
        Returns:
            List of RegisteredAgentInfo objects
        """
        self.logger.info("Retrieving all registered agent information")
        
        if not self.storage:
            raise RuntimeError("Storage backend not initialized")
        
        try:
            agents = await self.storage.list_registered_agent_info()
            self.logger.info(f"Retrieved {len(agents)} registered agents")
            return agents
        except Exception as e:
            self.logger.error(f"Failed to retrieve registered agent info: {str(e)}")
            raise RuntimeError(f"Failed to retrieve agent information: {str(e)}")
    
    async def get_registered_agent_info(self, agent_id: str) -> RegisteredAgentInfo:
        """Get registered agent information by ID.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            RegisteredAgentInfo object
            
        Raises:
            AgentNotRegisteredException: If agent is not registered
            RuntimeError: If storage backend not initialized
        """
        self.logger.info(f"Retrieving registered agent info for {agent_id}")
        
        if not self.storage:
            raise RuntimeError("Storage backend not initialized")
        
        try:
            agent_info = await self.storage.get_registered_agent_info(agent_id)
            if not agent_info:
                self.logger.warning(f"Agent {agent_id} not found")
                raise AgentNotRegisteredException(agent_id)
            
            self.logger.info(f"Retrieved registered agent info for {agent_id}")
            return agent_info
        except AgentNotRegisteredException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to retrieve agent info for {agent_id}: {str(e)}")
            raise RuntimeError(f"Failed to retrieve agent information: {str(e)}")
    
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
        
        return await self.storage.get_registered_agent_info(agent_id)
    
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
        agents = await self.storage.list_registered_agent_info()
        
        return {
            "status": "healthy" if storage_healthy else "unhealthy",
            "storage": "healthy" if storage_healthy else "unhealthy",
            "agents": len(agents)
        }
    
    async def list_registered_agent_status(self) -> List[AgentStatus]:
        """Get status of all registered agents.
        
        Fetches status from each agent's /status endpoint concurrently.
        For unreachable agents, creates AgentStatus with UNAVAILABLE status.
        
        Returns:
            List of AgentStatus objects for all registered agents
        """
        self.logger.info("Retrieving status for all registered agents")
        
        if not self.storage:
            raise RuntimeError("Storage backend not initialized")
        
        try:
            # Get all registered agents
            registered_agents = await self.storage.list_registered_agent_info()
            
            if not registered_agents:
                self.logger.info("No registered agents found")
                return []
            
            # Create simple HTTP client for status checks
            http_client = SimpleHttpClient(timeout_seconds=5.0)
            
            # Create tasks to fetch status from each agent concurrently
            tasks = []
            for agent in registered_agents:
                task = self._get_agent_status(http_client, agent)
                tasks.append(task)
            
            # Execute all status requests concurrently
            agent_statuses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle any exceptions
            results = []
            for i, status_result in enumerate(agent_statuses):
                if isinstance(status_result, Exception):
                    # Create unavailable status for failed requests
                    agent = registered_agents[i]
                    unavailable_status = AgentStatus(
                        agent_id=agent.agent_id,
                        status=AgentStatusEnum.UNAVAILABLE,
                        running_jobs=[],
                        queued_jobs=[],
                        uptime_seconds=0.0,
                        timestamp=datetime.utcnow()
                    )
                    results.append(unavailable_status)
                    self.logger.warning(f"Failed to get status for agent {agent.agent_id}: {status_result}")
                else:
                    results.append(status_result)
            
            self.logger.info(f"Retrieved status for {len(results)} agents")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve agent statuses: {str(e)}")
            raise RuntimeError(f"Failed to retrieve agent statuses: {str(e)}")
    
    async def _get_agent_status(self, http_client: SimpleHttpClient, agent: RegisteredAgentInfo) -> AgentStatus:
        """Get status from a single agent.
        
        Args:
            http_client: HTTP client for making requests
            agent: Registered agent information
            
        Returns:
            AgentStatus object from agent or UNAVAILABLE status if unreachable
        """
        try:
            # Build status endpoint URL
            status_url = urljoin(agent.agent_url.rstrip('/') + '/', 'api/v1/status')
            
            # Fetch status from agent
            response_data = await http_client.get_json(status_url)
            
            if response_data is None:
                # HTTP request failed, return unavailable status
                return AgentStatus(
                    agent_id=agent.agent_id,
                    status=AgentStatusEnum.UNAVAILABLE,
                    running_jobs=[],
                    queued_jobs=[],
                    uptime_seconds=0.0,
                    timestamp=datetime.utcnow()
                )
            
            # Parse response into AgentStatus
            agent_status = AgentStatus(**response_data)
            return agent_status
            
        except Exception as e:
            self.logger.debug(f"Error getting status for agent {agent.agent_id}: {e}")
            # Return unavailable status on any error
            return AgentStatus(
                agent_id=agent.agent_id,
                status=AgentStatusEnum.UNAVAILABLE,
                running_jobs=[],
                queued_jobs=[],
                uptime_seconds=0.0,
                timestamp=datetime.utcnow()
            )
    
    async def get_registered_agent_status(self, agent_id: str) -> AgentStatus:
        """Get status of a specific registered agent.
        
        Fetches status from the agent's /status endpoint.
        For unreachable agents, creates AgentStatus with UNAVAILABLE status.
        
        Args:
            agent_id: ID of the agent to get status for
            
        Returns:
            AgentStatus object for the specified agent
            
        Raises:
            AgentNotRegisteredException: If agent is not registered
            RuntimeError: If storage backend not initialized
        """
        self.logger.info(f"Retrieving status for agent {agent_id}")
        
        if not self.storage:
            raise RuntimeError("Storage backend not initialized")
        
        try:
            # Get the registered agent info
            agent = await self.storage.get_registered_agent_info(agent_id)
            if not agent:
                self.logger.warning(f"Agent {agent_id} not found in storage")
                raise AgentNotRegisteredException(agent_id)
            
            # Create simple HTTP client for status check
            http_client = SimpleHttpClient(timeout_seconds=5.0)
            
            # Get status from the agent
            agent_status = await self._get_agent_status(http_client, agent)
            
            self.logger.info(f"Retrieved status for agent {agent_id}: {agent_status.status}")
            return agent_status
            
        except AgentNotRegisteredException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to retrieve status for agent {agent_id}: {str(e)}")
            raise RuntimeError(f"Failed to retrieve agent status: {str(e)}")
