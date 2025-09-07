"""Job manager for coordinating job execution across agents."""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from .config import JobManagerSettings
from .storage.base import StorageBackend
from .agent import AgentProtocolUtility
from ..utils.job_utility import JobUtility
from ..core.models import JobInfo
from ruckus_common.models import (
    JobStatus,
    JobStatusEnum,
    ExperimentSpec,
    RegisteredAgentInfo,
    AgentStatusEnum,
    JobResult,
)
from ..api.v1.models import ExperimentResult


class JobManager:
    """Manages job creation, tracking, and execution across agents."""
    
    def __init__(self, settings: JobManagerSettings, storage: StorageBackend):
        """Initialize the JobManager.
        
        Args:
            settings: Job manager configuration settings
            storage: Storage backend for persisting job data
        """
        self.settings = settings
        self.storage = storage
        self.agent_utility = AgentProtocolUtility(
            settings.agent,
            settings.http_client
        )
        self.job_utility = JobUtility()
        self.logger = logging.getLogger(__name__)
        
        # Track polling tasks for cleanup
        self.polling_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
    
    async def start(self):
        """Start the job manager."""
        self.logger.info("Starting JobManager")
        self._running = True
    
    async def stop(self):
        """Stop the job manager and clean up resources."""
        self.logger.info("Stopping JobManager")
        self._running = False
        
        # Cancel all polling tasks
        for job_id, task in self.polling_tasks.items():
            if not task.done():
                self.logger.debug(f"Cancelling polling task for job {job_id}")
                task.cancel()
        
        # Wait for all tasks to complete
        if self.polling_tasks:
            await asyncio.gather(*self.polling_tasks.values(), return_exceptions=True)
        
        self.polling_tasks.clear()
    
    async def create_job(self, experiment_id: str, agent_id: str) -> JobInfo:
        """Create a new job for an experiment on a specific agent.
        
        Args:
            experiment_id: ID of the experiment to run
            agent_id: ID of the agent to run the job on
            
        Returns:
            JobInfo object for the created job
        """
        # Generate a unique job ID
        job_id = self.job_utility.generate_job_id()
        self.logger.info(f"Creating job {job_id} for experiment {experiment_id} on agent {agent_id}")
        
        # Process the job
        job_status = await self.process_job(job_id, experiment_id, agent_id)
        
        # Create JobInfo object
        job_info = JobInfo(
            job_id=job_id,
            experiment_id=experiment_id,
            agent_id=agent_id,
            created_time=datetime.now(timezone.utc),
            status=job_status
        )
        
        # Store the job based on its status
        await self._store_job_by_status(job_info)
        
        return job_info
    
    async def process_job(self, job_id: str, experiment_id: str, agent_id: str) -> JobStatus:
        """Process a job by attempting to execute it on an agent.
        
        Args:
            job_id: The job ID
            experiment_id: The experiment ID
            agent_id: The agent ID
            
        Returns:
            JobStatus indicating the result of processing
        """
        try:
            # Retrieve the experiment
            experiment = await self.storage.get_experiment(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} does not exist")
            
            # Retrieve the agent info
            agent_info = await self.storage.get_agent(agent_id)
            if not agent_info:
                raise ValueError(f"Agent {agent_id} does not exist")
            
            # Check if agent is unavailable
            if hasattr(agent_info, 'status') and agent_info.status == AgentStatusEnum.UNAVAILABLE:
                return JobStatus(
                    status=JobStatusEnum.FAILED,
                    message="Agent is unavailable"
                )
            
            # Get agent status
            try:
                agent_status = await self.agent_utility.get_agent_status(agent_info.agent_url)
            except Exception as e:
                self.logger.error(f"Failed to get agent status: {e}")
                return JobStatus(
                    status=JobStatusEnum.FAILED,
                    message=f"Failed to get agent status: {str(e)}"
                )
            
            # Check agent status
            if agent_status.status == AgentStatusEnum.ERROR:
                return JobStatus(
                    status=JobStatusEnum.FAILED,
                    message="Agent is in error state"
                )
            
            if agent_status.status == AgentStatusEnum.ACTIVE:
                return JobStatus(
                    status=JobStatusEnum.QUEUED,
                    message="Agent is busy with another job"
                )
            
            # Agent is idle, try to execute the experiment
            if agent_status.status == AgentStatusEnum.IDLE:
                try:
                    response = await self.agent_utility.execute_experiment(
                        agent_info.agent_url,
                        experiment,
                        job_id
                    )
                    
                    # Schedule status polling
                    polling_task = asyncio.create_task(
                        self._poll_job_status(job_id, agent_id)
                    )
                    self.polling_tasks[job_id] = polling_task
                    
                    return JobStatus(
                        status=JobStatusEnum.ASSIGNED,
                        message="Job has been scheduled with the agent"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to execute experiment: {e}")
                    return JobStatus(
                        status=JobStatusEnum.FAILED,
                        message=f"Failed to execute experiment: {str(e)}"
                    )
            
            # Unexpected agent status
            return JobStatus(
                status=JobStatusEnum.FAILED,
                message=f"Unexpected agent status: {agent_status.status}"
            )
            
        except ValueError as e:
            self.logger.error(f"Error processing job: {e}")
            return JobStatus(
                status=JobStatusEnum.FAILED,
                message=str(e)
            )
        except Exception as e:
            self.logger.error(f"Unexpected error processing job: {e}")
            return JobStatus(
                status=JobStatusEnum.FAILED,
                message=f"Unexpected error: {str(e)}"
            )
    
    async def _poll_job_status(self, job_id: str, agent_id: str):
        """Poll job status periodically.
        
        Args:
            job_id: The job ID to poll status for
            agent_id: The agent ID running the job
        """
        while self._running:
            try:
                await asyncio.sleep(self.settings.job_status_polling_interval)
                
                if not self._running:
                    break
                
                await self.process_job_status(job_id, agent_id)
                
            except asyncio.CancelledError:
                self.logger.debug(f"Polling cancelled for job {job_id}")
                break
            except Exception as e:
                self.logger.error(f"Error polling status for job {job_id}: {e}")
                # Continue polling despite errors
    
    async def process_job_status(self, job_id: str, agent_id: str):
        """Process the status of a running job.
        
        Args:
            job_id: The job ID to check status for
            agent_id: The agent ID running the job
        """
        try:
            # Get agent info
            agent_info = await self.storage.get_agent(agent_id)
            if not agent_info:
                self.logger.error(f"Agent {agent_id} not found")
                return
            
            # Get job status from agent
            try:
                job_status = await self.agent_utility.get_job_status(
                    agent_info.agent_url,
                    job_id
                )
            except Exception as e:
                self.logger.error(f"Failed to get job status: {e}")
                # Treat as failure
                job_status = JobStatus(
                    status=JobStatusEnum.FAILED,
                    message=f"Failed to get status: {str(e)}"
                )
            
            # Get current job info
            job_info = await self.storage.get_running_job(agent_id)
            if not job_info or job_info.job_id != job_id:
                # Job might have already been processed
                self.logger.warning(f"Job {job_id} is not the current running job for agent {agent_id}")
                # Cancel polling for this job
                if job_id in self.polling_tasks:
                    self.polling_tasks[job_id].cancel()
                    del self.polling_tasks[job_id]
                return
            
            # Update job info with new status
            job_info.status = job_status
            
            # Handle based on status
            if job_status.status in [JobStatusEnum.FAILED, JobStatusEnum.CANCELLED, JobStatusEnum.TIMEOUT]:
                # Move to failed jobs
                await self.storage.add_failed_job(agent_id, job_info)
                await self.storage.clear_running_job(agent_id)
                
                # Cancel polling
                if job_id in self.polling_tasks:
                    del self.polling_tasks[job_id]
                
                # Process next queued job
                await self.process_next_job(agent_id)
                
            elif job_status.status == JobStatusEnum.COMPLETED:
                # Move to completed jobs
                await self.storage.add_completed_job(agent_id, job_info)
                await self.storage.clear_running_job(agent_id)
                
                # Cancel polling
                if job_id in self.polling_tasks:
                    del self.polling_tasks[job_id]
                
                # Process results
                await self.process_job_results(job_id, agent_id)
                
                # Process next queued job
                await self.process_next_job(agent_id)
                
            elif job_status.status in [JobStatusEnum.ASSIGNED, JobStatusEnum.RUNNING]:
                # Update running job status
                await self.storage.update_running_job(agent_id, job_info)
                
            elif job_status.status == JobStatusEnum.QUEUED:
                # This shouldn't happen for a job we're polling
                # Move to queued jobs
                await self.storage.add_queued_job(agent_id, job_info)
                await self.storage.clear_running_job(agent_id)
                
                # Cancel polling
                if job_id in self.polling_tasks:
                    del self.polling_tasks[job_id]
                
                # Process next queued job
                await self.process_next_job(agent_id)
                
        except Exception as e:
            self.logger.error(f"Error processing job status for {job_id}: {e}")
    
    async def process_next_job(self, agent_id: str):
        """Process the next queued job for an agent.
        
        Args:
            agent_id: The agent ID to process next job for
        """
        try:
            # Get queued jobs for this agent
            queued_jobs = await self.storage.get_queued_jobs(agent_id)
            
            if not queued_jobs:
                self.logger.debug(f"No queued jobs for agent {agent_id}")
                return
            
            # Get the first queued job
            next_job = queued_jobs[0]
            
            # Remove from queued jobs
            await self.storage.remove_queued_job(agent_id, next_job.job_id)
            
            # Process the job
            job_status = await self.process_job(
                next_job.job_id,
                next_job.experiment_id,
                next_job.agent_id
            )
            
            # Update job info with new status
            next_job.status = job_status
            
            # Store based on new status
            await self._store_job_by_status(next_job)
            
        except Exception as e:
            self.logger.error(f"Error processing next job for agent {agent_id}: {e}")
    
    async def process_job_results(self, job_id: str, agent_id: str):
        """Process and store results for a completed job.
        
        Args:
            job_id: The job ID to get results for
            agent_id: The agent ID that ran the job
        """
        try:
            # Get agent info
            agent_info = await self.storage.get_agent(agent_id)
            if not agent_info:
                raise ValueError(f"Agent {agent_id} does not exist")
            
            # Get job info
            completed_jobs = await self.storage.get_completed_jobs(agent_id)
            job_info = None
            for job in completed_jobs:
                if job.job_id == job_id:
                    job_info = job
                    break
            
            if not job_info:
                raise ValueError(f"Job {job_id} not found in completed jobs")
            
            # Get results from agent
            try:
                # Get raw results from agent (returns dict)
                raw_results = await self.agent_utility.get_experiment_results(
                    agent_info.agent_url,
                    job_id
                )
                
                # Convert dict to JobResult object, or use existing JobResult
                if isinstance(raw_results, JobResult):
                    job_result = raw_results
                else:
                    job_result = JobResult(**raw_results)
                
                # Create ExperimentResult from JobResult
                experiment_result = ExperimentResult.from_job_result(job_result, agent_id)
                
                # Store ExperimentResult object
                await self.storage.store_experiment_result(experiment_result)
                
                # Also save raw results for backward compatibility
                await self.storage.save_experiment_results(
                    job_info.experiment_id,
                    job_result.model_dump()
                )
                
                self.logger.info(f"Successfully stored results for job {job_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to get or store results for job {job_id}: {e}")
                
        except Exception as e:
            self.logger.error(f"Error processing job results: {e}")
    
    async def _store_job_by_status(self, job_info: JobInfo):
        """Store a job based on its status.
        
        Args:
            job_info: The job info to store
        """
        status = job_info.status.status
        
        if status in [JobStatusEnum.ASSIGNED, JobStatusEnum.RUNNING]:
            await self.storage.set_running_job(job_info.agent_id, job_info)
        elif status == JobStatusEnum.QUEUED:
            await self.storage.add_queued_job(job_info.agent_id, job_info)
        elif status in [JobStatusEnum.FAILED, JobStatusEnum.CANCELLED, JobStatusEnum.TIMEOUT]:
            await self.storage.add_failed_job(job_info.agent_id, job_info)
        elif status == JobStatusEnum.COMPLETED:
            await self.storage.add_completed_job(job_info.agent_id, job_info)
        else:
            self.logger.warning(f"Unknown job status: {status}")
    
    async def get_job_status(self, job_id: str) -> Optional[JobInfo]:
        """Get the current status of a job.
        
        Args:
            job_id: The job ID to get status for
            
        Returns:
            JobInfo if found, None otherwise
        """
        # Search across all agents and job states
        agents = await self.storage.list_registered_agent_info()
        
        for agent in agents:
            # Check running job
            running_job = await self.storage.get_running_job(agent.agent_id)
            if running_job and running_job.job_id == job_id:
                return running_job
            
            # Check queued jobs
            queued_jobs = await self.storage.get_queued_jobs(agent.agent_id)
            for job in queued_jobs:
                if job.job_id == job_id:
                    return job
            
            # Check completed jobs
            completed_jobs = await self.storage.get_completed_jobs(agent.agent_id)
            for job in completed_jobs:
                if job.job_id == job_id:
                    return job
            
            # Check failed jobs
            failed_jobs = await self.storage.get_failed_jobs(agent.agent_id)
            for job in failed_jobs:
                if job.job_id == job_id:
                    return job
        
        return None
    
    async def get_agent_jobs(self, agent_id: str) -> Dict[str, Any]:
        """Get all jobs for a specific agent.
        
        Args:
            agent_id: The agent ID to get jobs for
            
        Returns:
            Dictionary with running, queued, completed, and failed jobs
        """
        return {
            "running": await self.storage.get_running_job(agent_id),
            "queued": await self.storage.get_queued_jobs(agent_id),
            "completed": await self.storage.get_completed_jobs(agent_id),
            "failed": await self.storage.get_failed_jobs(agent_id)
        }
    
    async def list_job_info(self) -> Dict[str, List[JobInfo]]:
        """Get all jobs across all agents, grouped by agent_id and sorted by timestamp.
        
        Returns:
            Dictionary of JobInfo lists keyed by agent_id, sorted by created_time (newest first)
        """
        agents = await self.storage.list_registered_agent_info()
        jobs_by_agent = {}
        
        for agent in agents:
            agent_id = agent.agent_id
            all_jobs = []
            
            # Get running job
            running_job = await self.storage.get_running_job(agent_id)
            if running_job:
                all_jobs.append(running_job)
            
            # Get queued jobs
            queued_jobs = await self.storage.get_queued_jobs(agent_id)
            all_jobs.extend(queued_jobs)
            
            # Get completed jobs
            completed_jobs = await self.storage.get_completed_jobs(agent_id)
            all_jobs.extend(completed_jobs)
            
            # Get failed jobs
            failed_jobs = await self.storage.get_failed_jobs(agent_id)
            all_jobs.extend(failed_jobs)
            
            # Sort jobs by created_time, newest first
            all_jobs.sort(key=lambda job: job.created_time, reverse=True)
            
            jobs_by_agent[agent_id] = all_jobs
        
        return jobs_by_agent
    
    async def cancel_job(self, job_id: str) -> None:
        """Cancel a job by job ID.
        
        Args:
            job_id: The job ID to cancel
            
        Raises:
            ValueError: If job or agent not found
        """
        # 1. Retrieve the JobInfo object matching the provided job_id from storage
        job_info = await self.get_job_status(job_id)
        if job_info is None:
            raise ValueError(f"Job {job_id} not found")
        
        # 2. Retrieve the RegisteredAgentInfo from storage for the agent
        agent_info = await self.storage.get_agent(job_info.agent_id)
        if not agent_info:
            raise ValueError(f"Agent {job_info.agent_id} does not exist")
        
        # 3. Call cancel_experiment() on the AgentUtility
        try:
            cancellation_successful = await self.agent_utility.cancel_experiment(
                agent_info.agent_url, 
                job_id
            )
            
            if not cancellation_successful:
                self.logger.error(f"Failed to cancel job {job_id} on agent {job_info.agent_id}")
                return
                
        except Exception as e:
            self.logger.error(f"Exception while cancelling job {job_id}: {str(e)}")
            return
        
        # 4. If cancellation successful, update job status and storage
        # Store the original status before updating
        original_status = job_info.status.status
        
        cancelled_status = JobStatus(
            status=JobStatusEnum.CANCELLED,
            message="Job was cancelled"
        )
        
        # Update the JobInfo object with cancelled status
        job_info.status = cancelled_status
        
        # Remove from current location (running or queued) based on original status
        if original_status in [JobStatusEnum.ASSIGNED, JobStatusEnum.RUNNING]:
            await self.storage.clear_running_job(job_info.agent_id)
        elif original_status == JobStatusEnum.QUEUED:
            await self.storage.remove_queued_job(job_info.agent_id, job_id)
        
        # Add to failed jobs (cancelled jobs go in failed jobs list)
        await self.storage.add_failed_job(job_info.agent_id, job_info)
        
        # Cancel any polling for this job
        if job_id in self.polling_tasks:
            if not self.polling_tasks[job_id].done():
                self.polling_tasks[job_id].cancel()
            del self.polling_tasks[job_id]
        
        # Process next queued job for this agent
        await self.process_next_job(job_info.agent_id)
        
        self.logger.info(f"Successfully cancelled job {job_id}")