"""Job management API endpoints."""

import logging
from fastapi import APIRouter, HTTPException, Request, status
from typing import Optional

from ..models import CreateJobRequest, CreateJobResponse, ListJobsResponse
from ....core.models import JobInfo

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["jobs"],
    responses={
        404: {"description": "Resource not found"},
        500: {"description": "Internal server error"},
    }
)


@router.post("", response_model=CreateJobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    request: Request,
    job_request: CreateJobRequest
) -> CreateJobResponse:
    """Create a new job for an experiment on a specific agent.
    
    Args:
        request: FastAPI request object containing app state
        job_request: Request containing experiment_id and agent_id
        
    Returns:
        CreateJobResponse containing the new job ID
        
    Raises:
        HTTPException: 404 if experiment or agent not found
        HTTPException: 500 for other errors
    """
    job_manager = request.app.state.job_manager
    
    try:
        # Create the job
        job_info: JobInfo = await job_manager.create_job(
            experiment_id=job_request.experiment_id,
            agent_id=job_request.agent_id
        )
        
        logger.info(f"Created job {job_info.job_id} for experiment {job_request.experiment_id} on agent {job_request.agent_id}")
        
        # Return the response
        return CreateJobResponse(job_id=job_info.job_id)
        
    except ValueError as e:
        # Handle not found errors
        error_message = str(e)
        if "does not exist" in error_message:
            logger.warning(f"Resource not found: {error_message}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_message
            )
        else:
            logger.error(f"Value error creating job: {error_message}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_message
            )
    
    except Exception as e:
        logger.error(f"Unexpected error creating job: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {str(e)}"
        )


@router.get("/{job_id}", response_model=JobInfo)
async def get_job_status(
    request: Request,
    job_id: str
) -> JobInfo:
    """Get the status of a specific job.
    
    Args:
        request: FastAPI request object containing app state
        job_id: ID of the job to get status for
        
    Returns:
        JobInfo object with current job status
        
    Raises:
        HTTPException: 404 if job not found
        HTTPException: 500 for other errors
    """
    job_manager = request.app.state.job_manager
    
    try:
        # Get job status
        job_info = await job_manager.get_job_status(job_id)
        
        if job_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        return job_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting job status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )


@router.get("", response_model=ListJobsResponse)
async def list_jobs(
    request: Request
) -> ListJobsResponse:
    """Get all jobs across all agents, grouped by agent_id.
    
    Args:
        request: FastAPI request object containing app state
        
    Returns:
        ListJobsResponse containing dictionary of JobInfo lists keyed by agent_id,
        sorted by timestamp (newest first)
        
    Raises:
        HTTPException: 500 for errors
    """
    job_manager = request.app.state.job_manager
    
    try:
        # Get all jobs grouped by agent
        jobs_by_agent = await job_manager.list_job_info()
        
        return ListJobsResponse(jobs=jobs_by_agent)
        
    except Exception as e:
        logger.error(f"Unexpected error listing jobs: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {str(e)}"
        )


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_job(
    request: Request,
    job_id: str
) -> None:
    """Cancel a specific job.
    
    Args:
        request: FastAPI request object containing app state
        job_id: ID of the job to cancel
        
    Raises:
        HTTPException: 404 if job not found
        HTTPException: 500 for other errors
    """
    job_manager = request.app.state.job_manager
    
    try:
        # Cancel the job
        await job_manager.cancel_job(job_id)
        
        logger.info(f"Successfully cancelled job {job_id}")
        
    except ValueError as e:
        # Handle not found errors
        error_message = str(e)
        if "not found" in error_message or "does not exist" in error_message:
            logger.warning(f"Resource not found: {error_message}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_message
            )
        else:
            logger.error(f"Value error cancelling job: {error_message}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_message
            )
    
    except Exception as e:
        logger.error(f"Unexpected error cancelling job: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}"
        )