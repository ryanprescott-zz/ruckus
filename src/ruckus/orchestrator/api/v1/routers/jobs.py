"""
Jobs API router for the Ruckus Orchestrator.

This module defines the FastAPI routes for managing jobs,
including CRUD operations and job-specific functionality.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_db
from ...core.orchestrator import OrchestratorService
from ...core.models import (
    Job, JobCreate, JobUpdate, JobList, JobStatus
)

router = APIRouter()


@router.post("/", response_model=Job, status_code=status.HTTP_201_CREATED)
async def create_job(
    job: JobCreate,
    db: AsyncSession = Depends(get_db)
) -> Job:
    """
    Create a new job.
    
    Args:
        job: Job creation data.
        db: Database session.
        
    Returns:
        Job: Created job instance.
    """
    orchestrator = OrchestratorService(db)
    return await orchestrator.create_job(job)


@router.get("/{job_id}", response_model=Job)
async def get_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> Job:
    """
    Get a job by ID.
    
    Args:
        job_id: Job identifier.
        db: Database session.
        
    Returns:
        Job: Job instance.
        
    Raises:
        HTTPException: If job not found.
    """
    orchestrator = OrchestratorService(db)
    job = await orchestrator.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return job


@router.get("/", response_model=List[Job])
async def list_jobs(
    experiment_id: Optional[UUID] = Query(None, description="Filter by experiment ID"),
    status: Optional[JobStatus] = Query(None, description="Filter by job status"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    db: AsyncSession = Depends(get_db)
) -> List[Job]:
    """
    List jobs with optional filtering and pagination.
    
    Args:
        experiment_id: Filter by experiment ID.
        status: Filter by job status.
        skip: Number of records to skip.
        limit: Maximum number of records to return.
        db: Database session.
        
    Returns:
        List[Job]: List of job instances.
    """
    orchestrator = OrchestratorService(db)
    return await orchestrator.list_jobs(
        experiment_id=experiment_id,
        status=status,
        skip=skip,
        limit=limit
    )


@router.put("/{job_id}", response_model=Job)
async def update_job(
    job_id: UUID,
    job_update: JobUpdate,
    db: AsyncSession = Depends(get_db)
) -> Job:
    """
    Update a job.
    
    Args:
        job_id: Job identifier.
        job_update: Update data.
        db: Database session.
        
    Returns:
        Job: Updated job instance.
        
    Raises:
        HTTPException: If job not found.
    """
    orchestrator = OrchestratorService(db)
    job = await orchestrator.update_job(job_id, job_update)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return job


@router.post("/{job_id}/assign/{agent_id}", response_model=Job)
async def assign_job_to_agent(
    job_id: UUID,
    agent_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> Job:
    """
    Assign a job to an agent.
    
    Args:
        job_id: Job identifier.
        agent_id: Agent identifier.
        db: Database session.
        
    Returns:
        Job: Updated job instance.
        
    Raises:
        HTTPException: If job not found.
    """
    orchestrator = OrchestratorService(db)
    job = await orchestrator.assign_job_to_agent(job_id, agent_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return job
