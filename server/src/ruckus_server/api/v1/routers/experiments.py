"""Experiment management endpoints."""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Optional
from datetime import datetime

from ruckus_common.models import (
    ExperimentSpec, ExperimentSubmission, ExperimentSubmissionResponse,
    ExperimentExecution
)
from ....core.orchestrator import OrchestrationError

router = APIRouter()


@router.post("/")
async def create_experiment(request: Request, submission: ExperimentSubmission) -> ExperimentSubmissionResponse:
    """Submit a new experiment for execution."""
    server = request.app.state.server
    if not server or not server.orchestrator:
        raise HTTPException(status_code=503, detail="Server not properly initialized")
    
    try:
        response = await server.orchestrator.submit_experiment(submission)
        return response
    except OrchestrationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/")
async def list_experiments(
    limit: int = 100,
    offset: int = 0,
    status: Optional[str] = None,
):
    """List experiments."""
    # TODO: Implement experiment listing
    return {
        "experiments": [],
        "total": 0,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get experiment details."""
    # TODO: Implement experiment retrieval
    raise HTTPException(status_code=404, detail="Experiment not found")


@router.get("/{experiment_id}/status")
async def get_experiment_status(request: Request, experiment_id: str):
    """Get experiment execution status."""
    server = request.app.state.server
    if not server or not server.orchestrator:
        raise HTTPException(status_code=503, detail="Server not properly initialized")
    
    try:
        execution = await server.orchestrator.get_experiment_execution(experiment_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return {
            "experiment_id": experiment_id,
            "status": execution.status,
            "progress": execution.progress_percent,
            "jobs_total": execution.total_jobs,
            "jobs_completed": len(execution.completed_jobs),
            "jobs_failed": len(execution.failed_jobs),
            "jobs_running": len(execution.running_jobs),
            "jobs_queued": len(execution.queued_jobs),
            "started_at": execution.started_at,
            "completed_at": execution.completed_at,
            "error_summary": execution.error_summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.delete("/{experiment_id}")
async def cancel_experiment(experiment_id: str):
    """Cancel an experiment."""
    # TODO: Implement experiment cancellation
    return {"cancelled": True}


@router.get("/{experiment_id}/results")
async def get_experiment_results(request: Request, experiment_id: str):
    """Get aggregated experiment results."""
    server = request.app.state.server
    if not server or not server.orchestrator:
        raise HTTPException(status_code=503, detail="Server not properly initialized")
    
    try:
        results = await server.orchestrator.get_experiment_results(experiment_id)
        if not results:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return {
            "experiment_id": experiment_id,
            **results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")