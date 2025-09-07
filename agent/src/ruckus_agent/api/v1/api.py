"""Main API router for agent v1."""

from fastapi import APIRouter, Request
from typing import Dict, Any, List, Optional

from ruckus_common.models import (
    JobRequest, JobUpdate, JobResult, AgentType,
    AgentInfoResponse, AgentInfo, AgentStatus, ExecuteJobRequest,
    JobStatusEnum
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
            "/jobs",  # POST for execution 
            "/status",
            "/status/{job_id}",  # GET for job status
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


@router.post("/jobs")
async def execute_job(execute_request: ExecuteJobRequest, request: Request):
    """Execute a benchmark job with experiment specification."""
    agent = request.app.state.agent
    
    # Convert ExecuteJobRequest to JobRequest for backward compatibility
    # Extract task config from the experiment spec
    task_config = {}
    if execute_request.experiment_spec.task.params:
        if hasattr(execute_request.experiment_spec.task.params, 'dict'):
            task_config = execute_request.experiment_spec.task.params.dict()
        else:
            task_config = execute_request.experiment_spec.task.params
    
    # Extract framework params
    framework_params = {}
    if execute_request.experiment_spec.framework.params:
        if hasattr(execute_request.experiment_spec.framework.params, 'dict'):
            framework_params = execute_request.experiment_spec.framework.params.dict()
        else:
            framework_params = execute_request.experiment_spec.framework.params
    
    # Create JobRequest from ExecuteJobRequest
    job_request = JobRequest(
        job_id=execute_request.job_id,
        experiment_id=execute_request.experiment_spec.id,
        model=execute_request.experiment_spec.model,
        framework=execute_request.experiment_spec.framework.name.value,
        task_type=execute_request.experiment_spec.task.type,
        task_config=task_config,
        parameters=framework_params,
        required_metrics=list(execute_request.experiment_spec.metrics.metrics.keys()) if execute_request.experiment_spec.metrics.metrics else [],
        optional_metrics=[],
        timeout_seconds=3600,  # Default timeout
        runs_per_job=1  # Default to single run
    )
    
    # Execute the job using the agent
    try:
        await agent.execute_job(job_request)
        return {
            "job_id": execute_request.job_id,
            "status": "accepted",
        }
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute job {execute_request.job_id}: {str(e)}"
        )


@router.get("/status", response_model=AgentStatus)
async def get_status(request: Request):
    """Get agent status."""
    agent = request.app.state.agent
    return await agent.get_status()


@router.get("/status/{job_id}")
async def get_job_status(job_id: str, request: Request):
    """Get status for a specific job."""
    agent = request.app.state.agent
    
    # Check if job is currently running
    if job_id in agent.running_jobs:
        job_info = agent.running_jobs[job_id]
        return {
            "job_id": job_id,
            "status": JobStatusEnum.RUNNING,
            "started_at": job_info.get("start_time"),
            "message": "Job is currently running"
        }
    
    # Check if job is in the queue
    if job_id in agent.queued_job_ids:
        return {
            "job_id": job_id,
            "status": JobStatusEnum.QUEUED,
            "message": "Job is queued for execution"
        }
    
    # Check if job result is cached
    result = agent.result_cache.get(job_id)
    if result:
        status = JobStatusEnum.COMPLETED
        if result.get("status") == "failed":
            status = JobStatusEnum.FAILED
        elif result.get("status") == "cancelled":
            status = JobStatusEnum.CANCELLED
        elif result.get("status") == "timeout":
            status = JobStatusEnum.TIMEOUT
            
        return {
            "job_id": job_id,
            "status": status,
            "completed_at": result.get("completed_at"),
            "message": f"Job {status.value}"
        }
    
    # Job not found
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


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