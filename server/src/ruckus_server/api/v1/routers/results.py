"""Experiment results API endpoints."""

import logging
from fastapi import APIRouter, HTTPException, Request, status
from typing import Optional

from ..models import ListExperimentResultsResponse, GetExperimentResultResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["results"],
    responses={
        404: {"description": "Resource not found"},
        500: {"description": "Internal server error"},
    }
)


@router.get("", response_model=ListExperimentResultsResponse)
async def list_experiment_results(
    request: Request
) -> ListExperimentResultsResponse:
    """Get all completed experiment results.
    
    Args:
        request: FastAPI request object containing app state
        
    Returns:
        ListExperimentResultsResponse containing list of completed job results from experiments
        
    Raises:
        HTTPException: 500 for errors
    """
    storage = request.app.state.storage_backend
    
    try:
        # Get all experiment results from storage
        results = await storage.list_experiment_results()
        return ListExperimentResultsResponse(results=results)
        
    except Exception as e:
        logger.error(f"Unexpected error listing experiment results: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list experiment results: {str(e)}"
        )


@router.get("/{job_id}", response_model=GetExperimentResultResponse)
async def get_experiment_result(
    request: Request,
    job_id: str
) -> GetExperimentResultResponse:
    """Get experiment result for a specific job.
    
    Args:
        request: FastAPI request object containing app state
        job_id: ID of the job to get results for
        
    Returns:
        GetExperimentResultResponse containing the experiment result for the specified job
        
    Raises:
        HTTPException: 404 if job result not found
        HTTPException: 500 for other errors
    """
    storage = request.app.state.storage_backend
    
    try:
        # Get experiment result by job ID
        result = await storage.get_experiment_result_by_job_id(job_id)
        
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment result for job {job_id} not found"
            )
        
        return GetExperimentResultResponse(result=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting experiment result: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get experiment result: {str(e)}"
        )