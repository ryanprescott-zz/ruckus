"""PostgreSQL storage backend implementation."""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from sqlalchemy import create_engine, select, update, delete
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from .base import StorageBackend, Base, Agent, Experiment, Job
from ..config import PostgreSQLSettings
from ruckus_common.models import AgentInfo, AgentType, RegisteredAgentInfo


class PostgreSQLStorageBackend(StorageBackend):
    """PostgreSQL storage backend implementation."""
    
    def __init__(self, settings: PostgreSQLSettings):
        """Initialize PostgreSQL storage backend.
        
        Args:
            settings: PostgreSQL storage settings.
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.engine = None
        self.session_factory = None
    
    async def initialize(self) -> None:
        """Initialize the PostgreSQL storage backend."""
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.settings.database_url,
                pool_size=self.settings.pool_size,
                max_overflow=self.settings.max_overflow,
                echo=self.settings.echo_sql
            )
            
            # Create session factory
            self.session_factory = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
            
            # Create tables if they don't exist
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self.logger.info("PostgreSQL storage backend initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PostgreSQL storage: {e}")
            raise
    
    async def close(self) -> None:
        """Close PostgreSQL connections."""
        if self.engine:
            await self.engine.dispose()
            self.logger.info("PostgreSQL storage backend closed")
    
    async def health_check(self) -> bool:
        """Check PostgreSQL connection health."""
        try:
            async with self.session_factory() as session:
                await session.execute(select(1))
                return True
        except Exception as e:
            self.logger.error(f"PostgreSQL health check failed: {e}")
            return False
    
    async def _retry_operation(self, operation, *args, **kwargs):
        """Retry database operations with exponential backoff."""
        import asyncio
        
        for attempt in range(self.settings.max_retries):
            try:
                return await operation(*args, **kwargs)
            except SQLAlchemyError as e:
                if attempt == self.settings.max_retries - 1:
                    self.logger.error(f"Database operation failed after {self.settings.max_retries} attempts: {e}")
                    raise
                
                wait_time = self.settings.retry_delay * (2 ** attempt)
                self.logger.warning(f"Database operation failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
    
    # Agent management
    async def register_agent(self, agent_info: RegisteredAgentInfo) -> bool:
        """Register a new agent with full information."""
        
        async def _register():
            async with self.session_factory() as session:
                agent = Agent(
                    id=agent_info.agent_id,
                    agent_name=agent_info.agent_name,
                    agent_type=agent_info.agent_type.value,  # Convert enum to string
                    agent_url=agent_info.agent_url,
                    system_info=agent_info.system_info,
                    capabilities=agent_info.capabilities,
                    status="active",
                    last_heartbeat=datetime.now(timezone.utc),
                    last_updated=agent_info.last_updated,
                    registered_at=agent_info.registered_at
                )
                session.add(agent)
                await session.commit()
                return True
        
        try:
            return await self._retry_operation(_register)
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent_info.agent_id}: {e}")
            return False
    
    async def update_agent_status(self, agent_id: str, status: str) -> bool:
        """Update agent status."""
        async def _update():
            async with self.session_factory() as session:
                stmt = update(Agent).where(Agent.id == agent_id).values(
                    status=status, updated_at=datetime.now(timezone.utc)
                )
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount > 0
        
        try:
            return await self._retry_operation(_update)
        except Exception as e:
            self.logger.error(f"Failed to update agent {agent_id} status: {e}")
            return False
    
    async def update_agent_heartbeat(self, agent_id: str) -> bool:
        """Update agent last heartbeat timestamp."""
        async def _update():
            async with self.session_factory() as session:
                stmt = update(Agent).where(Agent.id == agent_id).values(
                    last_heartbeat=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)
                )
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount > 0
        
        try:
            return await self._retry_operation(_update)
        except Exception as e:
            self.logger.error(f"Failed to update agent {agent_id} heartbeat: {e}")
            return False
    
    async def agent_exists(self, agent_id: str) -> bool:
        """Check if an agent exists by ID."""
        async def _exists():
            async with self.session_factory() as session:
                stmt = select(Agent).where(Agent.id == agent_id)
                result = await session.execute(stmt)
                agent = result.scalar_one_or_none()
                return agent is not None
        
        try:
            return await self._retry_operation(_exists)
        except Exception as e:
            self.logger.error(f"Failed to check if agent {agent_id} exists: {e}")
            return False
    
    def _agent_to_registered_agent_info(self, agent: Agent) -> RegisteredAgentInfo:
        """Convert a SQLAlchemy Agent object to RegisteredAgentInfo."""
        return RegisteredAgentInfo(
            agent_id=agent.id,
            agent_name=agent.agent_name,
            agent_type=AgentType(agent.agent_type),
            system_info=agent.system_info or {},
            capabilities=agent.capabilities or {},
            last_updated=agent.last_updated,
            agent_url=agent.agent_url,
            registered_at=agent.registered_at
        )
    
    async def get_registered_agent_info(self, agent_id: str) -> Optional[RegisteredAgentInfo]:
        """Get registered agent info by ID."""
        async def _get():
            async with self.session_factory() as session:
                stmt = select(Agent).where(Agent.id == agent_id)
                result = await session.execute(stmt)
                agent = result.scalar_one_or_none()
                
                if agent:
                    return self._agent_to_registered_agent_info(agent)
                return None
        
        try:
            return await self._retry_operation(_get)
        except Exception as e:
            self.logger.error(f"Failed to get agent {agent_id}: {e}")
            return None
    
    async def list_registered_agent_info(self) -> List[RegisteredAgentInfo]:
        """List all registered agent info."""
        async def _list():
            async with self.session_factory() as session:
                stmt = select(Agent)
                result = await session.execute(stmt)
                agents = result.scalars().all()
                
                return [self._agent_to_registered_agent_info(agent) for agent in agents]
        
        try:
            return await self._retry_operation(_list)
        except Exception as e:
            self.logger.error(f"Failed to list agents: {e}")
            return []
    
    async def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent."""
        async def _remove():
            async with self.session_factory() as session:
                stmt = delete(Agent).where(Agent.id == agent_id)
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount > 0
        
        try:
            return await self._retry_operation(_remove)
        except Exception as e:
            self.logger.error(f"Failed to remove agent {agent_id}: {e}")
            return False
    
    # Experiment management
    async def create_experiment(self, experiment_spec) -> Dict[str, Any]:
        """Create a new experiment from ExperimentSpec.
        
        Args:
            experiment_spec: ExperimentSpec object containing experiment details
            
        Returns:
            Dict containing the created experiment with 'experiment_id' and 'created_at'
            
        Raises:
            ExperimentAlreadyExistsException: If experiment with same ID already exists
        """
        from ruckus_server.core.storage.base import ExperimentAlreadyExistsException
        from sqlalchemy.exc import IntegrityError
        
        async def _create():
            async with self.session_factory() as session:
                # Check if experiment already exists
                stmt = select(Experiment).where(Experiment.id == experiment_spec.id)
                result = await session.execute(stmt)
                if result.scalar_one_or_none():
                    raise ExperimentAlreadyExistsException(experiment_spec.id)
                
                # Create new experiment
                created_at = datetime.now(timezone.utc)
                experiment = Experiment(
                    id=experiment_spec.id,
                    name=experiment_spec.name,
                    description=experiment_spec.description,
                    spec_data=experiment_spec.model_dump(mode='json'),
                    status="created",
                    created_at=created_at,
                    updated_at=created_at
                )
                session.add(experiment)
                await session.commit()
                
                return {
                    "experiment_id": experiment_spec.id,
                    "created_at": created_at
                }
        
        try:
            return await self._retry_operation(_create)
        except ExperimentAlreadyExistsException:
            # Re-raise this specific exception without wrapping
            raise
        except IntegrityError as e:
            # Handle database constraint violations
            if "unique constraint" in str(e).lower():
                raise ExperimentAlreadyExistsException(experiment_spec.id)
            else:
                self.logger.error(f"Failed to create experiment {experiment_spec.id}: {e}")
                raise
        except Exception as e:
            self.logger.error(f"Failed to create experiment {experiment_spec.id}: {e}")
            raise
    
    async def update_experiment_status(self, experiment_id: str, status: str) -> bool:
        """Update experiment status."""
        async def _update():
            async with self.session_factory() as session:
                stmt = update(Experiment).where(Experiment.id == experiment_id).values(
                    status=status, updated_at=datetime.now(timezone.utc)
                )
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount > 0
        
        try:
            return await self._retry_operation(_update)
        except Exception as e:
            self.logger.error(f"Failed to update experiment {experiment_id} status: {e}")
            return False
    
    async def get_experiment(self, experiment_id: str):
        """Get experiment by ID.
        
        Args:
            experiment_id: ID of the experiment to retrieve
            
        Returns:
            ExperimentSpec object
            
        Raises:
            ExperimentNotFoundException: If experiment with given ID doesn't exist
        """
        from ruckus_common.models import ExperimentSpec
        
        async def _get():
            async with self.session_factory() as session:
                stmt = select(Experiment).where(Experiment.id == experiment_id)
                result = await session.execute(stmt)
                experiment = result.scalar_one_or_none()
                
                if not experiment:
                    raise ExperimentNotFoundException(experiment_id)
                
                # Convert stored JSON back to ExperimentSpec
                return ExperimentSpec(**experiment.spec_data)
        
        try:
            return await self._retry_operation(_get)
        except ExperimentNotFoundException:
            # Re-raise this specific exception without wrapping
            raise
        except Exception as e:
            self.logger.error(f"Failed to get experiment {experiment_id}: {e}")
            raise
    
    async def list_experiments(self):
        """List all experiments.
        
        Returns:
            List of ExperimentSpec objects
        """
        from ruckus_common.models import ExperimentSpec
        
        async def _list():
            async with self.session_factory() as session:
                stmt = select(Experiment)
                result = await session.execute(stmt)
                experiments = result.scalars().all()
                
                # Convert each stored experiment to ExperimentSpec
                return [
                    ExperimentSpec(**experiment.spec_data)
                    for experiment in experiments
                ]
        
        try:
            return await self._retry_operation(_list)
        except Exception as e:
            self.logger.error(f"Failed to list experiments: {e}")
            raise
    
    async def delete_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Delete an experiment by ID.
        
        Args:
            experiment_id: ID of the experiment to delete
            
        Returns:
            Dict containing the deleted experiment's ID and deletion timestamp
            
        Raises:
            ExperimentNotFoundException: If experiment with given ID doesn't exist
            ExperimentHasJobsException: If experiment has associated jobs
        """
        from ruckus_server.core.storage.base import ExperimentNotFoundException, ExperimentHasJobsException
        
        async def _delete():
            async with self.session_factory() as session:
                # Check if experiment exists
                stmt = select(Experiment).where(Experiment.id == experiment_id)
                result = await session.execute(stmt)
                experiment = result.scalar_one_or_none()
                
                if not experiment:
                    raise ExperimentNotFoundException(experiment_id)
                
                # Check for associated jobs
                jobs_stmt = select(Job).where(Job.experiment_id == experiment_id)
                jobs_result = await session.execute(jobs_stmt)
                jobs = jobs_result.scalars().all()
                
                if jobs:
                    raise ExperimentHasJobsException(experiment_id, len(jobs))
                
                # Delete the experiment
                delete_stmt = delete(Experiment).where(Experiment.id == experiment_id)
                await session.execute(delete_stmt)
                await session.commit()
                
                return {
                    "experiment_id": experiment_id,
                    "deleted_at": datetime.now(timezone.utc)
                }
        
        try:
            return await self._retry_operation(_delete)
        except (ExperimentNotFoundException, ExperimentHasJobsException):
            # Re-raise these specific exceptions without wrapping
            raise
        except Exception as e:
            self.logger.error(f"Failed to delete experiment {experiment_id}: {e}")
            raise
    
    # Job management
    async def create_job(self, job_id: str, experiment_id: str, 
                        config: Dict[str, Any]) -> bool:
        """Create a new job."""
        async def _create():
            async with self.session_factory() as session:
                job = Job(
                    id=job_id,
                    experiment_id=experiment_id,
                    config=config,
                    status="scheduled"
                )
                session.add(job)
                await session.commit()
                return True
        
        try:
            return await self._retry_operation(_create)
        except Exception as e:
            self.logger.error(f"Failed to create job {job_id}: {e}")
            return False
    
    async def assign_job_to_agent(self, job_id: str, agent_id: str) -> bool:
        """Assign a job to an agent."""
        async def _assign():
            async with self.session_factory() as session:
                stmt = update(Job).where(Job.id == job_id).values(
                    agent_id=agent_id,
                    status="assigned",
                    updated_at=datetime.now(timezone.utc)
                )
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount > 0
        
        try:
            return await self._retry_operation(_assign)
        except Exception as e:
            self.logger.error(f"Failed to assign job {job_id} to agent {agent_id}: {e}")
            return False
    
    async def update_job_status(self, job_id: str, status: str, 
                               results: Optional[Dict[str, Any]] = None,
                               error_message: Optional[str] = None) -> bool:
        """Update job status and results."""
        async def _update():
            async with self.session_factory() as session:
                values = {
                    "status": status,
                    "updated_at": datetime.now(timezone.utc)
                }
                
                if results is not None:
                    values["results"] = results
                
                if error_message is not None:
                    values["error_message"] = error_message
                
                if status == "running":
                    values["started_at"] = datetime.now(timezone.utc)
                elif status in ["completed", "failed"]:
                    values["completed_at"] = datetime.now(timezone.utc)
                
                stmt = update(Job).where(Job.id == job_id).values(**values)
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount > 0
        
        try:
            return await self._retry_operation(_update)
        except Exception as e:
            self.logger.error(f"Failed to update job {job_id} status: {e}")
            return False
    
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID."""
        async def _get():
            async with self.session_factory() as session:
                stmt = select(Job).where(Job.id == job_id)
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()
                
                if job:
                    return {
                        "id": job.id,
                        "experiment_id": job.experiment_id,
                        "agent_id": job.agent_id,
                        "config": job.config,
                        "status": job.status,
                        "results": job.results,
                        "error_message": job.error_message,
                        "started_at": job.started_at,
                        "completed_at": job.completed_at,
                        "created_at": job.created_at,
                        "updated_at": job.updated_at
                    }
                return None
        
        try:
            return await self._retry_operation(_get)
        except Exception as e:
            self.logger.error(f"Failed to get job {job_id}: {e}")
            return None
    
    async def list_jobs(self, experiment_id: Optional[str] = None,
                       agent_id: Optional[str] = None,
                       status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List jobs with optional filtering."""
        async def _list():
            async with self.session_factory() as session:
                stmt = select(Job)
                
                if experiment_id:
                    stmt = stmt.where(Job.experiment_id == experiment_id)
                if agent_id:
                    stmt = stmt.where(Job.agent_id == agent_id)
                if status:
                    stmt = stmt.where(Job.status == status)
                
                result = await session.execute(stmt)
                jobs = result.scalars().all()
                
                return [
                    {
                        "id": job.id,
                        "experiment_id": job.experiment_id,
                        "agent_id": job.agent_id,
                        "config": job.config,
                        "status": job.status,
                        "results": job.results,
                        "error_message": job.error_message,
                        "started_at": job.started_at,
                        "completed_at": job.completed_at,
                        "created_at": job.created_at,
                        "updated_at": job.updated_at
                    }
                    for job in jobs
                ]
        
        try:
            return await self._retry_operation(_list)
        except Exception as e:
            self.logger.error(f"Failed to list jobs: {e}")
            return []
    
    async def get_jobs_for_agent(self, agent_id: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get jobs assigned to a specific agent."""
        return await self.list_jobs(agent_id=agent_id, status=status)
