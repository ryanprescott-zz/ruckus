"""
Core orchestrator logic for the Ruckus system.

This module contains the main orchestration logic for managing experiments,
jobs, and agents in the Ruckus LLM evaluation system.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload

from .models import (
    Experiment, ExperimentCreate, ExperimentUpdate,
    Job, JobCreate, JobUpdate, JobStatus,
    Agent, AgentCreate, AgentUpdate, AgentStatus
)
from .database_models import (
    ExperimentDB, JobDB, AgentDB
)


class OrchestratorService:
    """
    Core orchestrator service for managing LLM evaluation workflows.
    
    This service handles the creation, management, and coordination of
    experiments, jobs, and agents in the Ruckus system.
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize the orchestrator service.
        
        Args:
            db: Database session for data operations.
        """
        self.db = db

    async def create_experiment(self, experiment: ExperimentCreate) -> Experiment:
        """
        Create a new experiment.
        
        Args:
            experiment: Experiment creation data.
            
        Returns:
            Experiment: Created experiment instance.
        """
        db_experiment = ExperimentDB(**experiment.model_dump())
        self.db.add(db_experiment)
        await self.db.commit()
        await self.db.refresh(db_experiment)
        return Experiment.model_validate(db_experiment)

    async def get_experiment(self, experiment_id: UUID) -> Optional[Experiment]:
        """
        Get an experiment by ID.
        
        Args:
            experiment_id: Experiment identifier.
            
        Returns:
            Optional[Experiment]: Experiment instance if found.
        """
        result = await self.db.execute(
            select(ExperimentDB).where(ExperimentDB.id == experiment_id)
        )
        db_experiment = result.scalar_one_or_none()
        return Experiment.model_validate(db_experiment) if db_experiment else None

    async def list_experiments(
        self, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[Experiment]:
        """
        List experiments with pagination.
        
        Args:
            skip: Number of records to skip.
            limit: Maximum number of records to return.
            
        Returns:
            List[Experiment]: List of experiment instances.
        """
        result = await self.db.execute(
            select(ExperimentDB).offset(skip).limit(limit)
        )
        db_experiments = result.scalars().all()
        return [Experiment.model_validate(exp) for exp in db_experiments]

    async def update_experiment(
        self, 
        experiment_id: UUID, 
        experiment_update: ExperimentUpdate
    ) -> Optional[Experiment]:
        """
        Update an experiment.
        
        Args:
            experiment_id: Experiment identifier.
            experiment_update: Update data.
            
        Returns:
            Optional[Experiment]: Updated experiment instance if found.
        """
        update_data = experiment_update.model_dump(exclude_unset=True)
        if update_data:
            update_data["updated_at"] = datetime.utcnow()
            await self.db.execute(
                update(ExperimentDB)
                .where(ExperimentDB.id == experiment_id)
                .values(**update_data)
            )
            await self.db.commit()
        
        return await self.get_experiment(experiment_id)

    async def delete_experiment(self, experiment_id: UUID) -> bool:
        """
        Delete an experiment.
        
        Args:
            experiment_id: Experiment identifier.
            
        Returns:
            bool: True if experiment was deleted, False if not found.
        """
        result = await self.db.execute(
            delete(ExperimentDB).where(ExperimentDB.id == experiment_id)
        )
        await self.db.commit()
        return result.rowcount > 0

    async def create_job(self, job: JobCreate) -> Job:
        """
        Create a new job.
        
        Args:
            job: Job creation data.
            
        Returns:
            Job: Created job instance.
        """
        db_job = JobDB(**job.model_dump())
        self.db.add(db_job)
        await self.db.commit()
        await self.db.refresh(db_job)
        return Job.model_validate(db_job)

    async def get_job(self, job_id: UUID) -> Optional[Job]:
        """
        Get a job by ID.
        
        Args:
            job_id: Job identifier.
            
        Returns:
            Optional[Job]: Job instance if found.
        """
        result = await self.db.execute(
            select(JobDB).where(JobDB.id == job_id)
        )
        db_job = result.scalar_one_or_none()
        return Job.model_validate(db_job) if db_job else None

    async def list_jobs(
        self, 
        experiment_id: Optional[UUID] = None,
        status: Optional[JobStatus] = None,
        skip: int = 0, 
        limit: int = 100
    ) -> List[Job]:
        """
        List jobs with optional filtering and pagination.
        
        Args:
            experiment_id: Filter by experiment ID.
            status: Filter by job status.
            skip: Number of records to skip.
            limit: Maximum number of records to return.
            
        Returns:
            List[Job]: List of job instances.
        """
        query = select(JobDB)
        
        if experiment_id:
            query = query.where(JobDB.experiment_id == experiment_id)
        if status:
            query = query.where(JobDB.status == status)
            
        query = query.offset(skip).limit(limit)
        
        result = await self.db.execute(query)
        db_jobs = result.scalars().all()
        return [Job.model_validate(job) for job in db_jobs]

    async def update_job(
        self, 
        job_id: UUID, 
        job_update: JobUpdate
    ) -> Optional[Job]:
        """
        Update a job.
        
        Args:
            job_id: Job identifier.
            job_update: Update data.
            
        Returns:
            Optional[Job]: Updated job instance if found.
        """
        update_data = job_update.model_dump(exclude_unset=True)
        if update_data:
            update_data["updated_at"] = datetime.utcnow()
            
            # Set timestamps based on status changes
            if "status" in update_data:
                if update_data["status"] == JobStatus.RUNNING:
                    update_data["started_at"] = datetime.utcnow()
                elif update_data["status"] in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    update_data["completed_at"] = datetime.utcnow()
            
            await self.db.execute(
                update(JobDB)
                .where(JobDB.id == job_id)
                .values(**update_data)
            )
            await self.db.commit()
        
        return await self.get_job(job_id)

    async def register_agent(self, agent: AgentCreate) -> Agent:
        """
        Register a new agent.
        
        Args:
            agent: Agent registration data.
            
        Returns:
            Agent: Registered agent instance.
        """
        db_agent = AgentDB(**agent.model_dump())
        self.db.add(db_agent)
        await self.db.commit()
        await self.db.refresh(db_agent)
        return Agent.model_validate(db_agent)

    async def get_agent(self, agent_id: UUID) -> Optional[Agent]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: Agent identifier.
            
        Returns:
            Optional[Agent]: Agent instance if found.
        """
        result = await self.db.execute(
            select(AgentDB).where(AgentDB.id == agent_id)
        )
        db_agent = result.scalar_one_or_none()
        return Agent.model_validate(db_agent) if db_agent else None

    async def list_agents(
        self, 
        status: Optional[AgentStatus] = None,
        skip: int = 0, 
        limit: int = 100
    ) -> List[Agent]:
        """
        List agents with optional filtering and pagination.
        
        Args:
            status: Filter by agent status.
            skip: Number of records to skip.
            limit: Maximum number of records to return.
            
        Returns:
            List[Agent]: List of agent instances.
        """
        query = select(AgentDB)
        
        if status:
            query = query.where(AgentDB.status == status)
            
        query = query.offset(skip).limit(limit)
        
        result = await self.db.execute(query)
        db_agents = result.scalars().all()
        return [Agent.model_validate(agent) for agent in db_agents]

    async def update_agent_heartbeat(self, agent_id: UUID) -> Optional[Agent]:
        """
        Update agent heartbeat timestamp.
        
        Args:
            agent_id: Agent identifier.
            
        Returns:
            Optional[Agent]: Updated agent instance if found.
        """
        await self.db.execute(
            update(AgentDB)
            .where(AgentDB.id == agent_id)
            .values(
                last_heartbeat=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        )
        await self.db.commit()
        return await self.get_agent(agent_id)

    async def assign_job_to_agent(
        self, 
        job_id: UUID, 
        agent_id: UUID
    ) -> Optional[Job]:
        """
        Assign a job to an agent.
        
        Args:
            job_id: Job identifier.
            agent_id: Agent identifier.
            
        Returns:
            Optional[Job]: Updated job instance if found.
        """
        job_update = JobUpdate(
            agent_id=agent_id,
            status=JobStatus.RUNNING
        )
        return await self.update_job(job_id, job_update)
