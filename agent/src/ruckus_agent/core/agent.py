"""Main agent implementation."""

import asyncio
import httpx
import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from ruckus_common.models import JobRequest, JobStatus, AgentType, JobUpdate, AgentStatus, AgentStatusEnum
from .config import Settings
from .models import AgentRegistration, JobErrorReport
from .detector import AgentDetector
from .storage import AgentStorage, InMemoryStorage
from ..utils.error_reporter import ErrorReporter

logger = logging.getLogger(__name__)


class Agent:
    """Main agent class coordinating job execution."""

    def __init__(self, settings: Settings, storage: Optional[AgentStorage] = None):
        self.settings = settings
        # Generate unique agent ID and name
        self.agent_id = f"agent-{uuid.uuid4().hex[:8]}"
        self.agent_name = f"{self.agent_id}-{settings.agent_type.value}"
        self.orchestrator_url = settings.orchestrator_url

        # Storage backend
        self.storage = storage or InMemoryStorage()

        # State
        self.registered = False
        self.running_jobs: Dict[str, Any] = {}
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.queued_job_ids: List[str] = []  # Track queued job IDs for status reporting
        self.startup_time = datetime.utcnow()
        self.crashed = False
        self.crash_reason: Optional[str] = None
        
        # Error reporting
        self.error_reporter = ErrorReporter(self.agent_id)
        self.error_reports: Dict[str, JobErrorReport] = {}  # Store error reports for server retrieval

        # HTTP client for orchestrator communication
        self.client = httpx.AsyncClient()

        # Background tasks
        self.tasks: List[asyncio.Task] = []
        
        logger.info(f"Agent initialized: {self.agent_id} ({self.agent_name})")
        logger.debug(f"Orchestrator URL: {self.orchestrator_url}")

    async def start(self):
        """Start the agent."""
        logger.info(f"Starting agent {self.agent_id}")

        # Detect capabilities
        await self._detect_capabilities()

        # Register with orchestrator
        if self.orchestrator_url:
            await self._register()

        # Start background tasks
        self.tasks.append(asyncio.create_task(self._heartbeat_loop()))
        self.tasks.append(asyncio.create_task(self._job_executor()))

    async def stop(self):
        """Stop the agent."""
        logger.info(f"Stopping agent {self.agent_id}")

        # Cancel background tasks
        for task in self.tasks:
            task.cancel()

        # Close HTTP client
        await self.client.aclose()

    async def _detect_capabilities(self):
        """Detect agent capabilities and system info."""
        logger.info(f"Agent {self.agent_id} starting capability detection")
        try:
            detector = AgentDetector()
            detected = await detector.detect_all()
            
            # Store system info in storage
            system_info = {
                "system": detected.get("system", {}),
                "cpu": detected.get("cpu", {}),
                "gpus": detected.get("gpus", []),
                "frameworks": detected.get("frameworks", []),
                "models": detected.get("models", []),
                "hooks": detected.get("hooks", []),
                "metrics": detected.get("metrics", [])
            }
            await self.storage.store_system_info(system_info)
            
            # Store simplified capabilities for internal use
            capabilities = {
                "agent_type": self.settings.agent_type.value,
                "gpu_count": len(detected.get("gpus", [])),
                "frameworks": [f["name"] for f in detected.get("frameworks", [])],
                "max_concurrent_jobs": self.settings.max_concurrent_jobs,
                "monitoring_available": bool(detected.get("hooks", [])),
            }
            await self.storage.store_capabilities(capabilities)
            
            logger.info(f"Agent {self.agent_id} capabilities detected successfully")
            logger.debug(f"Detected {capabilities['gpu_count']} GPUs, {len(capabilities['frameworks'])} frameworks")
        except Exception as e:
            logger.error(f"Agent {self.agent_id} capability detection failed: {e}")
            raise

    async def _register(self):
        """Register with orchestrator."""
        logger.info(f"Agent {self.agent_id} attempting registration with orchestrator")
        try:
            # Get full system info including discovered models
            system_info = await self.get_system_info()
            capabilities = await self.get_capabilities()
            
            # Create comprehensive registration
            registration = AgentRegistration(
                agent_id=self.agent_id,
                agent_name=self.agent_name,
                agent_type=self.settings.agent_type,
                
                # System information
                system=system_info.get("system"),
                cpu=system_info.get("cpu"),
                gpus=[gpu for gpu in system_info.get("gpus", [])],
                
                # Framework and model information  
                frameworks=[fw for fw in system_info.get("frameworks", [])],
                models=[model for model in system_info.get("models", [])],
                hooks=[hook for hook in system_info.get("hooks", [])],
                
                # Capabilities
                capabilities=capabilities,
                metrics_available=[metric for metric in system_info.get("metrics", [])],
                
                # Configuration
                max_concurrent_jobs=self.settings.max_concurrent_jobs,
                max_batch_size=1,  # TODO: Make configurable
            )

            response = await self.client.post(
                f"{self.orchestrator_url}/api/v1/agents/register",
                json=registration.dict(),
            )
            if response.status_code == 200:
                self.registered = True
                logger.info(f"Agent {self.agent_id} registered successfully with orchestrator")
                logger.debug(f"Registered with {len(registration.models)} models, {len(registration.frameworks)} frameworks")
            else:
                logger.error(f"Registration failed with status {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Agent {self.agent_id} registration failed: {e}")
            raise

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to orchestrator."""
        logger.debug(f"Agent {self.agent_id} starting heartbeat loop")
        while True:
            try:
                await asyncio.sleep(self.settings.heartbeat_interval)

                if self.registered:
                    status = await self.get_status()
                    response = await self.client.post(
                        f"{self.orchestrator_url}/api/v1/agents/{self.agent_id}/heartbeat",
                        json=status.dict(),
                    )
                    logger.debug(f"Heartbeat sent, status: {response.status_code}")
                else:
                    logger.debug("Skipping heartbeat - not registered")
            except asyncio.CancelledError:
                logger.debug(f"Agent {self.agent_id} heartbeat loop cancelled")
                break
            except Exception as e:
                logger.error(f"Agent {self.agent_id} heartbeat error: {e}")

    async def _job_executor(self):
        """Execute jobs from queue."""
        logger.debug(f"Agent {self.agent_id} starting job executor")
        while True:
            try:
                job = await self.job_queue.get()
                # Remove from queued job IDs when we start processing
                if job.job_id in self.queued_job_ids:
                    self.queued_job_ids.remove(job.job_id)
                logger.info(f"Agent {self.agent_id} received job from queue: {job.job_id}")
                await self._execute_job(job)
            except asyncio.CancelledError:
                logger.debug(f"Agent {self.agent_id} job executor cancelled")
                break
            except Exception as e:
                logger.error(f"Agent {self.agent_id} job execution error: {e}")

    async def _execute_job(self, job: JobRequest):
        """Execute a single job with comprehensive error handling."""
        logger.info(f"Agent {self.agent_id} executing job {job.job_id}")
        
        start_time = datetime.utcnow()
        error_report = None
        
        try:
            # Start job tracking for error reporting
            await self.error_reporter.start_job_tracking(job.job_id, "initializing")
            
            # Add to running jobs
            self.running_jobs[job.job_id] = {
                "job": job,
                "start_time": start_time,
            }

            # Send initial status update
            await self.error_reporter.update_job_stage(job.job_id, "starting")
            update = JobUpdate(
                job_id=job.job_id,
                status=JobStatus.RUNNING,
                stage="initializing",
            )
            await self._send_update(update)
            
            # TODO: Implement actual job execution stages
            await self.error_reporter.update_job_stage(job.job_id, "model_loading")
            
            # Simulate model loading stage
            await asyncio.sleep(1)  # Replace with actual model loading
            
            await self.error_reporter.update_job_stage(job.job_id, "inference")
            
            # Simulate inference stage  
            await asyncio.sleep(2)  # Replace with actual inference
            
            await self.error_reporter.update_job_stage(job.job_id, "completing")
            
            # Job completed successfully
            logger.info(f"Agent {self.agent_id} job {job.job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} job {job.job_id} execution failed: {e}")
            
            try:
                # Generate comprehensive error report
                error_report = await self.error_reporter.generate_error_report(
                    job_id=job.job_id,
                    experiment_id=job.experiment_id,
                    error=e,
                    model_name=job.model,
                    model_path=f"/ruckus/models/{job.model}",  # TODO: Get actual path
                    framework=job.framework,
                    task_type=job.task_type.value,
                    parameters=job.parameters,
                    started_at=start_time,
                    model_size_gb=None  # TODO: Get from model discovery
                )
                
                # Store error report for server retrieval
                self.error_reports[job.job_id] = error_report
                
                # Attempt cleanup
                cleanup_actions = await self._cleanup_failed_job(job, e)
                error_report.cleanup_actions = cleanup_actions
                error_report.recovery_successful = True  # If we reach here, cleanup worked
                
                logger.info(f"Generated error report for job {job.job_id}: {error_report.error_type}")
                
            except Exception as cleanup_error:
                logger.error(f"Failed to generate error report or cleanup job {job.job_id}: {cleanup_error}")
                # Mark agent as crashed if cleanup fails
                self.crashed = True
                self.crash_reason = f"Failed cleanup after job {job.job_id}: {cleanup_error}"
            
            # Send failure update to server
            failure_update = JobUpdate(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                message=str(e)
            )
            await self._send_update(failure_update)
            
        finally:
            # Clean up job tracking and running jobs
            await self.error_reporter.cleanup_job_tracking(job.job_id)
            self.running_jobs.pop(job.job_id, None)

    async def _cleanup_failed_job(self, job: JobRequest, error: Exception) -> List[str]:
        """Perform cleanup actions after a job fails."""
        cleanup_actions = []
        
        try:
            # GPU memory cleanup
            cleanup_actions.append("clearing_gpu_cache")
            # TODO: Add actual GPU cache clearing (torch.cuda.empty_cache())
            
            # Model unloading
            cleanup_actions.append("unloading_model")
            # TODO: Add actual model unloading
            
            # Temporary file cleanup
            cleanup_actions.append("cleaning_temp_files")
            # TODO: Add temp file cleanup
            
            # Process cleanup
            cleanup_actions.append("process_cleanup")
            # TODO: Add process-specific cleanup
            
            logger.info(f"Completed cleanup actions for job {job.job_id}: {cleanup_actions}")
            
        except Exception as e:
            logger.error(f"Cleanup failed for job {job.job_id}: {e}")
            cleanup_actions.append(f"cleanup_failed: {e}")
            raise
        
        return cleanup_actions

    async def _send_update(self, update: JobUpdate):
        """Send job update to orchestrator."""
        if self.orchestrator_url:
            try:
                response = await self.client.post(
                    f"{self.orchestrator_url}/api/v1/jobs/{update.job_id}/update",
                    json=update.dict(),
                )
                logger.debug(f"Job update sent for {update.job_id}: {update.status}, status: {response.status_code}")
            except Exception as e:
                logger.error(f"Agent {self.agent_id} failed to send update for job {update.job_id}: {e}")
        else:
            logger.debug(f"No orchestrator URL configured, skipping update for job {update.job_id}")
    
    async def get_capabilities(self) -> Dict:
        """Get agent capabilities."""
        return await self.storage.get_capabilities()

    async def get_system_info(self) -> Dict:
        """Get detailed system info for /info endpoint."""
        return await self.storage.get_system_info()

    async def get_status(self) -> AgentStatus:
        """Get agent status."""
        # Determine status based on current state
        if self.crashed:
            status = "crashed"
        elif self.running_jobs:
            status = "active"
        elif self.queued_job_ids:
            status = "idle"  # Has queued jobs but not running any
        else:
            status = "idle"
        
        # Calculate uptime
        uptime_seconds = (datetime.utcnow() - self.startup_time).total_seconds()
        
        return AgentStatus(
            agent_id=self.agent_id,
            status=status,
            running_jobs=list(self.running_jobs.keys()),
            queued_jobs=len(self.queued_job_ids),  # Return count instead of list
            uptime_seconds=uptime_seconds,
            timestamp=datetime.utcnow()
        )

    async def queue_job(self, job: JobRequest):
        """Queue a job for execution."""
        logger.info(f"Agent {self.agent_id} queueing job: {job.job_id}")
        try:
            await self.job_queue.put(job)
            self.queued_job_ids.append(job.job_id)  # Track queued job ID
            logger.debug(f"Job {job.job_id} added to queue, queue size: {self.job_queue.qsize()}")
            return job.job_id
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to queue job {job.job_id}: {e}")
            raise

    async def get_error_reports(self) -> List[JobErrorReport]:
        """Get all stored error reports."""
        return list(self.error_reports.values())
    
    async def get_error_report(self, job_id: str) -> Optional[JobErrorReport]:
        """Get error report for a specific job."""
        return self.error_reports.get(job_id)
    
    async def clear_error_reports(self) -> int:
        """Clear all error reports and return count cleared."""
        count = len(self.error_reports)
        self.error_reports.clear()
        self.crashed = False  # Reset crashed state when reports are cleared
        self.crash_reason = None
        logger.info(f"Cleared {count} error reports, reset crashed state")
        return count
