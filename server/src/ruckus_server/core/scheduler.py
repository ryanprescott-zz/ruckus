"""Job scheduling service."""

from typing import List, Dict, Optional
import asyncio
from datetime import datetime, timedelta

from ruckus_common.models import JobSpec, JobStatus


class Scheduler:
    """Job scheduler for distributing work to agents."""

    def __init__(self):
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.running_jobs: Dict[str, JobSpec] = {}
        self.agent_assignments: Dict[str, str] = {}  # job_id -> agent_id

    async def add_job(self, job: JobSpec):
        """Add job to scheduling queue."""
        await self.job_queue.put(job)

    async def assign_job(self, agent_id: str) -> Optional[JobSpec]:
        """Assign next available job to agent."""
        if self.job_queue.empty():
            return None

        job = await self.job_queue.get()
        self.running_jobs[job.job_id] = job
        self.agent_assignments[job.job_id] = agent_id
        return job

    async def complete_job(self, job_id: str):
        """Mark job as completed."""
        if job_id in self.running_jobs:
            del self.running_jobs[job_id]
        if job_id in self.agent_assignments:
            del self.agent_assignments[job_id]

    def get_agent_jobs(self, agent_id: str) -> List[str]:
        """Get all jobs assigned to an agent."""
        return [
            job_id for job_id, assigned_agent
            in self.agent_assignments.items()
            if assigned_agent == agent_id
        ]

    def get_scheduler_stats(self) -> Dict:
        """Get scheduler statistics."""
        return {
            "queued": self.job_queue.qsize(),
            "running": len(self.running_jobs),
            "total_assignments": len(self.agent_assignments),
        }