"""Main agent implementation."""

import asyncio
import httpx
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from ruckus_common.models import JobRequest, JobStatus, AgentType, JobUpdate
from .config import Settings
from .models import AgentRegistration, AgentStatus
from .detector import AgentDetector
from .storage import AgentStorage, InMemoryStorage


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
        """Detect agent capabilities and system info."""
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
        
        print(f"Agent {self.agent_id} ({self.agent_name}) capabilities detected")

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
        await self.job_queue.put(job)
        return job.job_id