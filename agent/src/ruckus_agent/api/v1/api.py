"""Main API router for agent v1."""

from fastapi import APIRouter, Request
from typing import Dict, Any

from ruckus_common.models import (
    JobRequest, JobUpdate, JobResult, AgentType,
    AgentRegistrationResponse, AgentInfoResponse, AgentInfo, AgentStatus
)

router = APIRouter()


@router.get("/")
async def api_info():
    """API information endpoint."""
    return {
        "version": "v1",
        "type": "agent",
        "endpoints": [
            "/register",
            "/info",
            "/capabilities",
            "/execute",
            "/status",
        ]
    }


@router.get("/register", response_model=AgentRegistrationResponse)
async def register_agent(request: Request):
    """Register agent with server - announces agent ID and name."""
    agent = request.app.state.agent
    
    response = AgentRegistrationResponse(
        agent_id=agent.agent_id,
        agent_name=agent.agent_name,
        message="Agent registered successfully"
    )
    return response


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