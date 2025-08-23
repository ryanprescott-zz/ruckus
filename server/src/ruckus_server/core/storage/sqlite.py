"""SQLite storage backend implementation."""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, select, update, delete
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from .base import StorageBackend, Base, Agent, Experiment, Job
from ..config import SQLiteSettings
from ruckus_common.models import AgentInfo, AgentType, RegisteredAgentInfo


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
                    last_heartbeat=datetime.utcnow(),
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
                    status=status, updated_at=datetime.utcnow()
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
                    last_heartbeat=datetime.utcnow(), updated_at=datetime.utcnow()
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
    async def create_experiment(self, experiment_id: str, name: str, 
                              description: str, config: Dict[str, Any]) -> bool:
        """Create a new experiment."""
        async def _create():
            async with self.session_factory() as session:
                experiment = Experiment(
                    id=experiment_id,
                    name=name,
                    description=description,
                    config=config,
                    status="created"
                )
                session.add(experiment)
                await session.commit()
                return True
        
        try:
            return await self._retry_operation(_create)
        except Exception as e:
            self.logger.error(f"Failed to create experiment {experiment_id}: {e}")
            return False
    
    async def update_experiment_status(self, experiment_id: str, status: str) -> bool:
        """Update experiment status."""
        async def _update():
            async with self.session_factory() as session:
                stmt = update(Experiment).where(Experiment.id == experiment_id).values(
                    status=status, updated_at=datetime.utcnow()
                )
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount > 0
        
        try:
            return await self._retry_operation(_update)
        except Exception as e:
            self.logger.error(f"Failed to update experiment {experiment_id} status: {e}")
            return False
    
    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by ID."""
        async def _get():
            async with self.session_factory() as session:
                stmt = select(Experiment).where(Experiment.id == experiment_id)
                result = await session.execute(stmt)
                experiment = result.scalar_one_or_none()
                
                if experiment:
                    return {
                        "id": experiment.id,
                        "name": experiment.name,
                        "description": experiment.description,
                        "config": experiment.config,
                        "status": experiment.status,
                        "created_at": experiment.created_at,
                        "updated_at": experiment.updated_at
                    }
                return None
        
        try:
            return await self._retry_operation(_get)
        except Exception as e:
            self.logger.error(f"Failed to get experiment {experiment_id}: {e}")
            return None
    
    async def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        async def _list():
            async with self.session_factory() as session:
                stmt = select(Experiment)
                result = await session.execute(stmt)
                experiments = result.scalars().all()
                
                return [
                    {
                        "id": experiment.id,
                        "name": experiment.name,
                        "description": experiment.description,
                        "config": experiment.config,
                        "status": experiment.status,
                        "created_at": experiment.created_at,
                        "updated_at": experiment.updated_at
                    }
                    for experiment in experiments
                ]
        
        try:
            return await self._retry_operation(_list)
        except Exception as e:
            self.logger.error(f"Failed to list experiments: {e}")
            return []
    
    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        async def _delete():
            async with self.session_factory() as session:
                stmt = delete(Experiment).where(Experiment.id == experiment_id)
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount > 0
        
        try:
            return await self._retry_operation(_delete)
        except Exception as e:
            self.logger.error(f"Failed to delete experiment {experiment_id}: {e}")
            return False
    
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
                    updated_at=datetime.utcnow()
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
                    "updated_at": datetime.utcnow()
                }
                
                if results is not None:
                    values["results"] = results
                
                if error_message is not None:
                    values["error_message"] = error_message
                
                if status == "running":
                    values["started_at"] = datetime.utcnow()
                elif status in ["completed", "failed"]:
                    values["completed_at"] = datetime.utcnow()
                
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
