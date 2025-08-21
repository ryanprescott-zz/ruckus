"""Main agent implementation."""

import asyncio
import httpx
from typing import Dict, Any, Optional, List
from datetime import datetime

from ruckus_common.models import JobRequest, JobStatus, AgentType
from .config import Settings
from .models import AgentRegistration, AgentStatus
from .detector import AgentDetector


class Agent:
    """Main agent class coordinating job execution."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.agent_id = settings.agent_id
        self.orchestrator_url = settings.orchestrator_url

        # State
        self.registered = False
        self.capabilities = {}
        self.running_jobs: Dict[str, Any] = {}
        self.job_queue: asyncio.Queue = asyncio.Queue()

        # HTTP client for orchestrator communication
        self.client = httpx.AsyncClient()

        # Background tasks
        self.tasks: List[asyncio.Task] = []

    async def start(self):
        """Start the agent."""
        print(f"Starting agent {self.agent_id}")

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
        print(f"Stopping agent {self.agent_id}")

        # Cancel background tasks
        for task in self.tasks:
            task.cancel()

        # Close HTTP client
        await self.client.aclose()

    async def _detect_capabilities(self):
        """Detect agent capabilities."""
        detector = AgentDetector()
        self.capabilities = await detector.detect_all()
        print(f"Detected capabilities: {self.capabilities}")

    async def _register(self):
        """Register with orchestrator."""
        registration = AgentRegistration(
            agent_id=self.agent_id,
            agent_type=self.settings.agent_type,
            capabilities=self.capabilities,
            # TODO: Add more registration details
        )

        try:
            response = await self.client.post(
                f"{self.orchestrator_url}/api/v1/agents/register",
                json=registration.dict(),
            )
            if response.status_code == 200:
                self.registered = True
                print(f"Registered with orchestrator")
        except Exception as e:
            print(f"Failed to register: {e}")

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to orchestrator."""
        while True:
            try:
                await asyncio.sleep(self.settings.heartbeat_interval)

                if self.registered:
                    status = await self.get_status()
                    await self.client.post(
                        f"{self.orchestrator_url}/api/v1/agents/{self.agent_id}/heartbeat",
                        json=status,
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Heartbeat error: {e}")

    async def _job_executor(self):
        """Execute jobs from queue."""
        while True:
            try:
                job = await self.job_queue.get()
                await self._execute_job(job)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Job execution error: {e}")

    async def _execute_job(self, job: JobRequest):
        """Execute a single job."""
        print(f"Executing job {job.job_id}")
        # TODO: Implement actual job execution

        # Update status
        update = JobUpdate(
            job_id=job.job_id,
            status=JobStatus.RUNNING,
            stage="initializing",
        )
        await self._send_update(update)

    async def _send_update(self, update: JobUpdate):
        """Send job update to orchestrator."""
        if self.orchestrator_url:
            try:
                await self.client.post(
                    f"{self.orchestrator_url}/api/v1/jobs/{update.job_id}/update",
                    json=update.dict(),
                )
            except Exception as e:
                print(f"Failed to send update: {e}")
    
    async def get_capabilities(self) -> Dict:
        """Get agent capabilities."""
        return self.capabilities

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
        await self.job_queue.put(job)
        return job.job_id