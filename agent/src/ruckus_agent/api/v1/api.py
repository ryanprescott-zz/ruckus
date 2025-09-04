"""Main API router for agent v1."""

from fastapi import APIRouter, Request
from typing import Dict, Any, List, Optional

from ruckus_common.models import (
    JobRequest, JobUpdate, JobResult, AgentType,
    AgentInfoResponse, AgentInfo, AgentStatus
)

router = APIRouter()


@router.get("/")
async def api_info():
    """API information endpoint."""
    return {
        "version": "v1",
        "type": "agent",
        "endpoints": [
            "/info",
            "/execute", 
            "/status",
            "/results/{job_id}",
            "/jobs/{job_id}",  # DELETE for cancellation
        ]
    }


@router.get("/info", response_model=AgentInfoResponse)
async def get_agent_info(request: Request):
    """Get detailed agent system information."""
    agent = request.app.state.agent
    system_info = await agent.get_system_info()
    
    agent_info = AgentInfo(
        agent_id=agent.agent_id,
        agent_name=agent.agent_name,
        agent_type=agent.settings.agent_type,
        system_info=system_info
    )
    
    response = AgentInfoResponse(agent_info=agent_info)
    return response


@router.post("/execute")
async def execute_job(job_request: JobRequest, request: Request):
    """Execute a benchmark job."""
    agent = request.app.state.agent
    
    # Actually execute the job using the agent
    try:
        await agent.execute_job(job_request)
        return {
            "job_id": job_request.job_id,
            "status": "accepted",
        }
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute job {job_request.job_id}: {str(e)}"
        )


@router.get("/status", response_model=AgentStatus)
async def get_status(request: Request):
    """Get agent status."""
    agent = request.app.state.agent
    return await agent.get_status()


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, request: Request):
    """Cancel a running job."""
    agent = request.app.state.agent
    
    try:
        success, reason = await agent.cancel_job(job_id)
        
        if success:
            return {
                "cancelled": True, 
                "job_id": job_id,
                "message": f"Job {job_id} cancelled successfully",
                "reason": reason
            }
        else:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to cancel job {job_id}: {reason}"
            )
            
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail=f"Error cancelling job {job_id}: {str(e)}"
        )




@router.get("/results/{job_id}")
async def get_job_results(job_id: str, request: Request):
    """Get results for a completed job."""
    agent = request.app.state.agent
    
    # Get result from cache
    result = agent.result_cache.get(job_id)
    
    if result is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Results not found for job {job_id}")
    
    return result