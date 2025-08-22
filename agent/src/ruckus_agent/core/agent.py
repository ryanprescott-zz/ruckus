"""Main agent implementation."""

import asyncio
import httpx
import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from ruckus_common.models import JobRequest, JobStatus, AgentType, JobUpdate
from .config import Settings
from .models import AgentRegistration, AgentStatus
from .detector import AgentDetector
from .storage import AgentStorage, InMemoryStorage

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
        self.startup_time = datetime.utcnow()

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
            capabilities = await self.get_capabilities()
            registration = AgentRegistration(
                agent_id=self.agent_id,
                agent_type=self.settings.agent_type,
                capabilities=capabilities,
                # TODO: Add more registration details
            )

            response = await self.client.post(
                f"{self.orchestrator_url}/api/v1/agents/register",
                json=registration.dict(),
            )
            if response.status_code == 200:
                self.registered = True
                logger.info(f"Agent {self.agent_id} registered successfully with orchestrator")
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
                        json=status,
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
                logger.info(f"Agent {self.agent_id} received job from queue: {job.job_id}")
                await self._execute_job(job)
            except asyncio.CancelledError:
                logger.debug(f"Agent {self.agent_id} job executor cancelled")
                break
            except Exception as e:
                logger.error(f"Agent {self.agent_id} job execution error: {e}")

    async def _execute_job(self, job: JobRequest):
        """Execute a single job."""
        logger.info(f"Agent {self.agent_id} executing job {job.job_id}")
        try:
            # Add to running jobs
            self.running_jobs[job.job_id] = {
                "job": job,
                "start_time": datetime.utcnow(),
            }

            # TODO: Implement actual job execution

            # Update status
            update = JobUpdate(
                job_id=job.job_id,
                status=JobStatus.RUNNING,
                stage="initializing",
            )
            await self._send_update(update)
            
            logger.info(f"Agent {self.agent_id} job {job.job_id} execution started")
        except Exception as e:
            logger.error(f"Agent {self.agent_id} job {job.job_id} execution failed: {e}")
            # Remove from running jobs on error
            self.running_jobs.pop(job.job_id, None)
            raise

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

    async def get_status(self) -> Dict:
        """Get agent status."""
        return {
            "agent_id": self.agent_id,
            "status": "idle" if not self.running_jobs else "busy",
            "running_jobs": list(self.running_jobs.keys()),
            "queued_jobs": self.job_queue.qsize(),
            "registered": self.registered,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def queue_job(self, job: JobRequest):
        """Queue a job for execution."""
        logger.info(f"Agent {self.agent_id} queueing job: {job.job_id}")
        try:
            await self.job_queue.put(job)
            logger.debug(f"Job {job.job_id} added to queue, queue size: {self.job_queue.qsize()}")
            return job.job_id
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to queue job {job.job_id}: {e}")
            raise
