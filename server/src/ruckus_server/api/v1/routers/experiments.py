"""Experiment management endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime

from ruckus_common.models import ExperimentSpec

router = APIRouter()


@router.post("/")
async def create_experiment(experiment: ExperimentSpec):
    """Create a new experiment."""
    # TODO: Implement experiment creation
    return {
        "experiment_id": experiment.experiment_id,
        "status": "created",
        "jobs_queued": 0,
    }


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
async def get_experiment_status(experiment_id: str):
    """Get experiment execution status."""
    # TODO: Implement status retrieval
    return {
        "experiment_id": experiment_id,
        "status": "running",
        "progress": 0.0,
        "jobs_total": 0,
        "jobs_completed": 0,
        "jobs_failed": 0,
    }


@router.delete("/{experiment_id}")
async def cancel_experiment(experiment_id: str):
    """Cancel an experiment."""
    # TODO: Implement experiment cancellation
    return {"cancelled": True}


@router.get("/{experiment_id}/results")
async def get_experiment_results(experiment_id: str):
    """Get aggregated experiment results."""
    # TODO: Implement results aggregation
    return {
        "experiment_id": experiment_id,
        "results": {},
        "summary": {},
    }