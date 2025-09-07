"""SQLite storage backend implementation."""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, select, update, delete
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from .base import StorageBackend, Base, Agent, Experiment, Job, ExperimentAlreadyExistsException, ExperimentNotFoundException
from ..config import SQLiteSettings
from ruckus_common.models import AgentInfo, AgentType, RegisteredAgentInfo, ExperimentSpec, JobStatusEnum


class SQLiteStorageBackend(StorageBackend):
    """SQLite storage backend implementation."""
    
    def __init__(self, settings: SQLiteSettings):
        """Initialize SQLite storage backend.
        
        Args:
            settings: SQLite storage settings.
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.engine = None
        self.session_factory = None
        
        # In-memory storage for experiment results (keeping this for now)
        self._experiment_results = {} # experiment_id -> results
    
    async def initialize(self) -> None:
        """Initialize the SQLite storage backend."""
        try:
            # Ensure database directory exists
            db_path = Path(self.settings.database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create async engine for SQLite
            database_url = f"sqlite+aiosqlite:///{self.settings.database_path}"
            self.engine = create_async_engine(
                database_url,
                echo=self.settings.echo_sql,
                # SQLite specific settings
                connect_args={"check_same_thread": False}
            )
            
            # Create session factory
            self.session_factory = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
            
            # Create tables if they don't exist
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self.logger.info(f"SQLite storage backend initialized at {self.settings.database_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite storage: {e}")
            raise
    
    async def close(self) -> None:
        """Close SQLite connections."""
        if self.engine:
            await self.engine.dispose()
            self.logger.info("SQLite storage backend closed")
    
    async def health_check(self) -> bool:
        """Check SQLite connection health."""
        try:
            async with self.session_factory() as session:
                await session.execute(select(1))
                return True
        except Exception as e:
            self.logger.error(f"SQLite health check failed: {e}")
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
    async def create_experiment(self, experiment_spec: ExperimentSpec) -> Dict[str, Any]:
        """Create a new experiment from ExperimentSpec."""
        
        async def _create():
            async with self.session_factory() as session:
                # Check if experiment already exists
                existing_stmt = select(Experiment).where(Experiment.id == experiment_spec.id)
                result = await session.execute(existing_stmt)
                existing_experiment = result.scalar_one_or_none()
                
                if existing_experiment:
                    raise ExperimentAlreadyExistsException(experiment_spec.id)
                
                # Create new experiment
                experiment = Experiment(
                    id=experiment_spec.id,
                    name=experiment_spec.name,
                    description=experiment_spec.description,
                    spec_data=experiment_spec.model_dump(mode='json'),  # Store complete ExperimentSpec as JSON
                    status="created"
                )
                session.add(experiment)
                await session.commit()
                
                return {
                    "experiment_id": experiment.id,
                    "created_at": experiment.created_at
                }
        
        try:
            return await self._retry_operation(_create)
        except ExperimentAlreadyExistsException:
            # Re-raise this specific exception without wrapping
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
        """Delete an experiment by ID."""
        
        async def _delete():
            async with self.session_factory() as session:
                # First, check if experiment exists
                experiment_stmt = select(Experiment).where(Experiment.id == experiment_id)
                experiment_result = await session.execute(experiment_stmt)
                experiment = experiment_result.scalar_one_or_none()
                
                if not experiment:
                    raise ExperimentNotFoundException(experiment_id)
                
                
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
        except ExperimentNotFoundException:
            # Re-raise these specific exceptions without wrapping
            raise
        except Exception as e:
            self.logger.error(f"Failed to delete experiment {experiment_id}: {e}")
            raise
    
    async def get_agent(self, agent_id: str):
        """Get agent by ID."""
        return await self.get_registered_agent_info(agent_id)
    
    # Helper methods for JobInfo conversion
    def _job_to_job_info(self, job: Job):
        """Convert a SQLAlchemy Job object to JobInfo."""
        from ..models import JobInfo
        from ruckus_common.models import JobStatus, JobStatusEnum
        
        return JobInfo(
            job_id=job.job_id,
            experiment_id=job.experiment_id,
            agent_id=job.agent_id,
            created_time=job.created_time,
            status=JobStatus(
                status=JobStatusEnum(job.status),
                message=job.status_message or ""
            )
        )
    
    def _job_info_to_job(self, job_info) -> Job:
        """Convert a JobInfo object to SQLAlchemy Job."""
        return Job(
            job_id=job_info.job_id,
            experiment_id=job_info.experiment_id,
            agent_id=job_info.agent_id,
            status=job_info.status.status.value,
            status_message=job_info.status.message,
            created_time=job_info.created_time
        )
    
    async def _upsert_job(self, job_info):
        """Insert or update a job in the database."""
        async def _upsert():
            async with self.session_factory() as session:
                # Check if job exists
                stmt = select(Job).where(Job.job_id == job_info.job_id)
                result = await session.execute(stmt)
                existing_job = result.scalar_one_or_none()
                
                if existing_job:
                    # Update existing job
                    existing_job.status = job_info.status.status.value
                    existing_job.status_message = job_info.status.message
                    existing_job.updated_at = datetime.now(timezone.utc)
                    if job_info.status.status.value == "running" and not existing_job.started_time:
                        existing_job.started_time = datetime.now(timezone.utc)
                    # Don't set completed_time automatically - let clear_running_job handle that
                else:
                    # Create new job
                    job = self._job_info_to_job(job_info)
                    session.add(job)
                
                await session.commit()
        
        try:
            await self._retry_operation(_upsert)
        except Exception as e:
            self.logger.error(f"Failed to upsert job {job_info.job_id}: {e}")
            raise
    
    # Job management methods (database implementation)
    async def get_running_job(self, agent_id: str):
        """Get the currently running job for an agent."""
        async def _get():
            async with self.session_factory() as session:
                # Get the most recent job that is actively running or assigned
                stmt = select(Job).where(
                    Job.agent_id == agent_id,
                    Job.status.in_(["running", "assigned"])
                ).order_by(Job.updated_at.desc()).limit(1)
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()
                
                if job:
                    return self._job_to_job_info(job)
                
                # If no active job, check if there's a recently completed/failed job 
                # that hasn't been "cleared" yet (no completed_time means it's still the "current" job)
                stmt = select(Job).where(
                    Job.agent_id == agent_id,
                    Job.status.in_(["completed", "failed"]),
                    Job.completed_time.is_(None)
                ).order_by(Job.updated_at.desc()).limit(1)
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()
                
                return self._job_to_job_info(job) if job else None
        
        try:
            return await self._retry_operation(_get)
        except Exception as e:
            self.logger.error(f"Failed to get running job for agent {agent_id}: {e}")
            return None
    
    async def set_running_job(self, agent_id: str, job_info):
        """Set the running job for an agent."""
        # Update job status to running and ensure agent_id matches
        job_info_copy = job_info.model_copy()
        job_info_copy.agent_id = agent_id
        job_info_copy.status.status = JobStatusEnum.RUNNING
        await self._upsert_job(job_info_copy)
    
    async def clear_running_job(self, agent_id: str):
        """Clear the running job for an agent."""
        async def _clear():
            async with self.session_factory() as session:
                # Find the current running job (most recent non-queued job)
                stmt = select(Job).where(
                    Job.agent_id == agent_id,
                    Job.status.in_(["running", "assigned", "completed", "failed"])
                ).order_by(Job.updated_at.desc()).limit(1)
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()
                
                if job:
                    # Mark it as completed if it wasn't already
                    if job.status not in ["completed", "failed", "cancelled"]:
                        job.status = "completed"
                    if not job.completed_time:
                        job.completed_time = datetime.now(timezone.utc)
                    job.updated_at = datetime.now(timezone.utc)
                    
                    await session.commit()
        
        try:
            await self._retry_operation(_clear)
        except Exception as e:
            self.logger.error(f"Failed to clear running job for agent {agent_id}: {e}")
    
    async def update_running_job(self, agent_id: str, job_info):
        """Update the running job for an agent."""
        # Simply update the job in place - don't move it to other categories yet
        await self._upsert_job(job_info)
    
    async def get_queued_jobs(self, agent_id: str):
        """Get queued jobs for an agent."""
        async def _get():
            async with self.session_factory() as session:
                stmt = select(Job).where(
                    Job.agent_id == agent_id,
                    Job.status.in_(["queued", "assigned"])
                ).order_by(Job.created_time)
                result = await session.execute(stmt)
                jobs = result.scalars().all()
                return [self._job_to_job_info(job) for job in jobs]
        
        try:
            return await self._retry_operation(_get)
        except Exception as e:
            self.logger.error(f"Failed to get queued jobs for agent {agent_id}: {e}")
            return []
    
    async def add_queued_job(self, agent_id: str, job_info):
        """Add a job to the queue for an agent."""
        # Set status to queued and ensure agent_id matches
        job_info_copy = job_info.model_copy()
        job_info_copy.agent_id = agent_id
        job_info_copy.status.status = JobStatusEnum.QUEUED
        await self._upsert_job(job_info_copy)
    
    async def remove_queued_job(self, agent_id: str, job_id: str):
        """Remove a job from the queue for an agent."""
        async def _remove():
            async with self.session_factory() as session:
                stmt = delete(Job).where(
                    Job.agent_id == agent_id,
                    Job.job_id == job_id,
                    Job.status.in_(["queued", "assigned"])
                )
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount > 0
        
        try:
            return await self._retry_operation(_remove)
        except Exception as e:
            self.logger.error(f"Failed to remove queued job {job_id} for agent {agent_id}: {e}")
            return False
    
    async def get_completed_jobs(self, agent_id: str):
        """Get completed jobs for an agent."""
        async def _get():
            async with self.session_factory() as session:
                stmt = select(Job).where(
                    Job.agent_id == agent_id,
                    Job.status == "completed"
                ).order_by(Job.completed_time.desc())
                result = await session.execute(stmt)
                jobs = result.scalars().all()
                return [self._job_to_job_info(job) for job in jobs]
        
        try:
            return await self._retry_operation(_get)
        except Exception as e:
            self.logger.error(f"Failed to get completed jobs for agent {agent_id}: {e}")
            return []
    
    async def add_completed_job(self, agent_id: str, job_info):
        """Add a job to the completed jobs for an agent."""
        # Set status to completed and ensure agent_id matches
        job_info_copy = job_info.model_copy()
        job_info_copy.agent_id = agent_id
        job_info_copy.status.status = JobStatusEnum.COMPLETED
        await self._upsert_job(job_info_copy)
    
    async def get_failed_jobs(self, agent_id: str):
        """Get failed jobs for an agent."""
        async def _get():
            async with self.session_factory() as session:
                stmt = select(Job).where(
                    Job.agent_id == agent_id,
                    Job.status == "failed"
                ).order_by(Job.completed_time.desc())
                result = await session.execute(stmt)
                jobs = result.scalars().all()
                return [self._job_to_job_info(job) for job in jobs]
        
        try:
            return await self._retry_operation(_get)
        except Exception as e:
            self.logger.error(f"Failed to get failed jobs for agent {agent_id}: {e}")
            return []
    
    async def add_failed_job(self, agent_id: str, job_info):
        """Add a job to the failed jobs for an agent."""
        # Set status to failed and ensure agent_id matches
        job_info_copy = job_info.model_copy()
        job_info_copy.agent_id = agent_id
        job_info_copy.status.status = JobStatusEnum.FAILED
        await self._upsert_job(job_info_copy)
    
    async def save_experiment_results(self, experiment_id: str, results):
        """Save experiment results."""
        self._experiment_results[experiment_id] = results

    async def store_experiment_result(self, experiment_result):
        """Store an ExperimentResult object."""
        # Store by job_id for easy lookup
        if not hasattr(self, '_experiment_result_objects'):
            self._experiment_result_objects = {}
        self._experiment_result_objects[experiment_result.job_id] = experiment_result

    async def get_experiment_result_by_job_id(self, job_id: str):
        """Get ExperimentResult by job ID."""
        if not hasattr(self, '_experiment_result_objects'):
            self._experiment_result_objects = {}
        return self._experiment_result_objects.get(job_id)

    async def list_experiment_results(self):
        """Get all stored experiment results."""
        if not hasattr(self, '_experiment_result_objects'):
            self._experiment_result_objects = {}
        return list(self._experiment_result_objects.values())
    
