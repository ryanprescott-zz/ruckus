"""Experiment management endpoints."""

from fastapi import APIRouter, HTTPException, Request
from typing import List, Optional
from datetime import datetime

from ..models import CreateExperimentRequest, CreateExperimentResponse, DeleteExperimentResponse, ListExperimentsResponse, ExperimentSummary, GetExperimentResponse
from ruckus_server.core.storage.base import ExperimentAlreadyExistsException, ExperimentNotFoundException, ExperimentHasJobsException
from ruckus_common.models import ExperimentSpec

router = APIRouter()


@router.post("/", response_model=CreateExperimentResponse)
async def create_experiment(request_data: CreateExperimentRequest, request: Request):
    """Create and store a new experiment.
    
    Args:
        request_data: CreateExperimentRequest containing ExperimentSpec
        request: FastAPI request object to access app state
        
    Returns:
        CreateExperimentResponse containing experiment ID and creation time
        
    Raises:
        HTTPException: 409 for experiment conflicts, 503 for server issues, 500 for other errors
    """
    experiment_manager = getattr(request.app.state, 'experiment_manager', None)
    if not experiment_manager:
        raise HTTPException(status_code=503, detail="Experiment manager not initialized")
    
    try:
        # Call the ExperimentManager create_experiment method
        result = await experiment_manager.create_experiment(request_data.experiment_spec)
        
        # Return success response
        return CreateExperimentResponse(
            experiment_id=result["experiment_id"],
            created_at=result["created_at"]
        )
        
    except ExperimentAlreadyExistsException as e:
        # Experiment already exists - return 409 conflict
        raise HTTPException(
            status_code=409,
            detail=f"Experiment {e.experiment_id} already exists"
        )
    except ValueError as e:
        # Invalid experiment data or other validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/", response_model=ListExperimentsResponse)
async def list_experiments(request: Request):
    """List all existing experiments.
    
    Args:
        request: FastAPI request object to access app state
        
    Returns:
        ListExperimentsResponse containing list of ExperimentSpec objects
        
    Raises:
        HTTPException: 503 for server issues, 500 for other errors
    """
    experiment_manager = getattr(request.app.state, 'experiment_manager', None)
    if not experiment_manager:
        raise HTTPException(status_code=503, detail="Experiment manager not initialized")
    
    try:
        # Call the ExperimentManager list_experiments method
        experiments = await experiment_manager.list_experiments()
        
        # Return response with list of ExperimentSpec objects
        return ListExperimentsResponse(experiments=experiments)
        
    except ValueError as e:
        # Invalid request data or other validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{experiment_id}", response_model=GetExperimentResponse)
async def get_experiment(experiment_id: str, request: Request):
    """Get a specific experiment by ID.
    
    Args:
        experiment_id: ID of the experiment to retrieve
        request: FastAPI request object to access app state
        
    Returns:
        GetExperimentResponse containing the complete ExperimentSpec
        
    Raises:
        HTTPException: 404 if experiment not found, 503 for server issues, 500 for other errors
    """
    experiment_manager = getattr(request.app.state, 'experiment_manager', None)
    if not experiment_manager:
        raise HTTPException(status_code=503, detail="Experiment manager not initialized")
    
    try:
        # Call the ExperimentManager get_experiment method
        experiment_spec = await experiment_manager.get_experiment(experiment_id)
        
        # Return response with the ExperimentSpec
        return GetExperimentResponse(experiment=experiment_spec)
        
    except ExperimentNotFoundException as e:
        # Experiment not found - return 404
        raise HTTPException(
            status_code=404,
            detail=f"Experiment {e.experiment_id} not found"
        )
    except ValueError as e:
        # Invalid request data or other validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/{experiment_id}", response_model=DeleteExperimentResponse)
async def delete_experiment(experiment_id: str, request: Request):
    """Delete an existing experiment by ID.
    
    Args:
        experiment_id: ID of the experiment to delete
        request: FastAPI request object to access app state
        
    Returns:
        DeleteExperimentResponse containing experiment ID and deletion timestamp
        
    Raises:
        HTTPException: 404 if experiment not found, 409 if experiment has jobs, 503 for server issues, 500 for other errors
    """
    experiment_manager = getattr(request.app.state, 'experiment_manager', None)
    if not experiment_manager:
        raise HTTPException(status_code=503, detail="Experiment manager not initialized")
    
    try:
        # Call the ExperimentManager delete_experiment method
        result = await experiment_manager.delete_experiment(experiment_id)
        
        # Return success response
        return DeleteExperimentResponse(
            experiment_id=result["experiment_id"],
            deleted_at=result["deleted_at"]
        )
        
    except ExperimentNotFoundException as e:
        # Experiment not found - return 404
        raise HTTPException(
            status_code=404,
            detail=f"Experiment {e.experiment_id} not found"
        )
    except ExperimentHasJobsException as e:
        # Experiment has jobs - return 409 conflict
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete experiment {e.experiment_id}: it has {e.job_count} associated job(s)"
        )
    except ValueError as e:
        # Invalid request data or other validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")