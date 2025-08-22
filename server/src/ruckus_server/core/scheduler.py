"""Job scheduling service."""

import logging
from typing import List, Dict, Optional
import asyncio
from datetime import datetime, timedelta

from ruckus_common.models import JobSpec, JobStatus


class Scheduler:
    """Job scheduler for distributing work to agents."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.running_jobs: Dict[str, JobSpec] = {}
        self.agent_assignments: Dict[str, str] = {}  # job_id -> agent_id
        
        self.logger.info("Scheduler initialized")
        self.logger.debug("Job queue and tracking dictionaries created")

    async def add_job(self, job: JobSpec):
        """Add job to scheduling queue."""
        self.logger.info(f"Adding job to queue: {job.job_id}")
        self.logger.debug(f"Job details - experiment_id: {job.experiment_id}, task_type: {getattr(job, 'task_type', 'unknown')}")
        
        try:
            await self.job_queue.put(job)
            queue_size = self.job_queue.qsize()
            self.logger.info(f"Job {job.job_id} added to queue successfully. Queue size: {queue_size}")
        except Exception as e:
            self.logger.error(f"Failed to add job {job.job_id} to queue: {e}")
            raise

    async def assign_job(self, agent_id: str) -> Optional[JobSpec]:
        """Assign next available job to agent."""
        self.logger.debug(f"Agent {agent_id} requesting job assignment")
        
        if self.job_queue.empty():
            self.logger.debug(f"No jobs available for agent {agent_id}")
            return None

        try:
            job = await self.job_queue.get()
            self.running_jobs[job.job_id] = job
            self.agent_assignments[job.job_id] = agent_id
            
            self.logger.info(f"Assigned job {job.job_id} to agent {agent_id}")
            self.logger.debug(f"Running jobs count: {len(self.running_jobs)}, Queue size: {self.job_queue.qsize()}")
            
            return job
        except Exception as e:
            self.logger.error(f"Failed to assign job to agent {agent_id}: {e}")
            raise

    async def complete_job(self, job_id: str):
        """Mark job as completed."""
        self.logger.info(f"Completing job: {job_id}")
        
        agent_id = self.agent_assignments.get(job_id)
        if agent_id:
            self.logger.debug(f"Job {job_id} was assigned to agent {agent_id}")
        
        job_was_running = job_id in self.running_jobs
        assignment_existed = job_id in self.agent_assignments
        
        if job_id in self.running_jobs:
            del self.running_jobs[job_id]
            self.logger.debug(f"Removed job {job_id} from running jobs")
            
        if job_id in self.agent_assignments:
            del self.agent_assignments[job_id]
            self.logger.debug(f"Removed job {job_id} assignment")
        
        if job_was_running or assignment_existed:
            self.logger.info(f"Job {job_id} completed successfully. Running jobs: {len(self.running_jobs)}")
        else:
            self.logger.warning(f"Attempted to complete job {job_id} that was not tracked")

    def get_agent_jobs(self, agent_id: str) -> List[str]:
        """Get all jobs assigned to an agent."""
        self.logger.debug(f"Getting jobs for agent: {agent_id}")
        
        agent_jobs = [
            job_id for job_id, assigned_agent
            in self.agent_assignments.items()
            if assigned_agent == agent_id
        ]
        
        self.logger.debug(f"Agent {agent_id} has {len(agent_jobs)} assigned jobs: {agent_jobs}")
        return agent_jobs

    def get_scheduler_stats(self) -> Dict:
        """Get scheduler statistics."""
        stats = {
            "queued": self.job_queue.qsize(),
            "running": len(self.running_jobs),
            "total_assignments": len(self.agent_assignments),
        }
        
        self.logger.debug(f"Scheduler stats: {stats}")
        return stats
