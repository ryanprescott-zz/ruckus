"""Main API router for agent v1."""

from fastapi import APIRouter, Request
from typing import Dict, Any, List, Optional

from ruckus_common.models import (
    JobRequest, JobUpdate, JobResult, AgentType,
    AgentInfoResponse, AgentInfo, AgentStatus
)
from ruckus_agent.core.models import JobErrorReport

router = APIRouter()


@router.get("/")
async def api_info():
    """API information endpoint."""
    return {
        "version": "v1",
        "type": "agent",
        "endpoints": [
            "/info",
            "/capabilities",
            "/execute",
            "/status",
            "/errors",
            "/errors/{job_id}",
            "/errors/clear",
        ]
    }


@router.get("/info", response_model=AgentInfoResponse)
async def get_agent_info(request: Request):
    """Get detailed agent system information."""
    agent = request.app.state.agent
    system_info = await agent.get_system_info()
    capabilities = await agent.get_capabilities()
    
    agent_info = AgentInfo(
        agent_id=agent.agent_id,
        agent_name=agent.agent_name,
        agent_type=agent.settings.agent_type,
        system_info=system_info,
        capabilities=capabilities
    )
    
    response = AgentInfoResponse(agent_info=agent_info)
    return response


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


@router.get("/status", response_model=AgentStatus)
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


@router.get("/errors", response_model=List[JobErrorReport])
async def get_error_reports(request: Request):
    """Get all error reports from failed jobs."""
    agent = request.app.state.agent
    return await agent.get_error_reports()


@router.get("/errors/{job_id}", response_model=Optional[JobErrorReport])
async def get_error_report(job_id: str, request: Request):
    """Get error report for a specific job."""
    agent = request.app.state.agent
    return await agent.get_error_report(job_id)


@router.delete("/errors/clear")
async def clear_error_reports(request: Request):
    """Clear all error reports and reset crashed state."""
    agent = request.app.state.agent
    count = await agent.clear_error_reports()
    return {
        "message": f"Cleared {count} error reports",
        "cleared_count": count,
        "agent_reset": True
    }