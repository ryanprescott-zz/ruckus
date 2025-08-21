"""Main API router for agent v1."""

from fastapi import APIRouter, Request
from typing import Dict, Any

from ruckus_common.models import JobRequest, JobUpdate, JobResult

router = APIRouter()


@router.get("/")
async def api_info():
    """API information endpoint."""
    return {
        "version": "v1",
        "type": "agent",
        "endpoints": [
            "/capabilities",
            "/execute",
            "/status",
        ]
    }


@router.get("/capabilities")
async def get_capabilities(request: Request):
    """Get agent capabilities."""
    agent = request.app.state.agent
    return await agent.get_capabilities()


@router.post("/execute")
async def execute_job(job_request: JobRequest, request: Request):
    """Execute a benchmark job."""
    agent = request.app.state.agent
    # TODO: Queue job for execution
    return {
        "job_id": job_request.job_id,
        "status": "accepted",
    }


@router.get("/status")
async def get_status(request: Request):
    """Get agent status."""
    agent = request.app.state.agent
    return await agent.get_status()


@router.post("/cancel/{job_id}")
async def cancel_job(job_id: str, request: Request):
    """Cancel a running job."""
    agent = request.app.state.agent
    # TODO: Implement job cancellation
    return {"cancelled": False, "reason": "Not implemented"}