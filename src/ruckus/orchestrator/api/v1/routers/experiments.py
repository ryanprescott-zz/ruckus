"""
Experiments API router for the Ruckus Orchestrator.

This module defines the FastAPI routes for managing experiments,
including CRUD operations and experiment-specific functionality.
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_db
from ...core.orchestrator import OrchestratorService
from ...core.models import (
    Experiment, ExperimentCreate, ExperimentUpdate, ExperimentList
)

router = APIRouter()


@router.post("/", response_model=Experiment, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    experiment: ExperimentCreate,
    db: AsyncSession = Depends(get_db)
) -> Experiment:
    """
    Create a new experiment.
    
    Args:
        experiment: Experiment creation data.
        db: Database session.
        
    Returns:
        Experiment: Created experiment instance.
    """
    orchestrator = OrchestratorService(db)
    return await orchestrator.create_experiment(experiment)


@router.get("/{experiment_id}", response_model=Experiment)
async def get_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> Experiment:
    """
    Get an experiment by ID.
    
    Args:
        experiment_id: Experiment identifier.
        db: Database session.
        
    Returns:
        Experiment: Experiment instance.
        
    Raises:
        HTTPException: If experiment not found.
    """
    orchestrator = OrchestratorService(db)
    experiment = await orchestrator.get_experiment(experiment_id)
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found"
        )
    
    return experiment


@router.get("/", response_model=List[Experiment])
async def list_experiments(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
) -> List[Experiment]:
    """
    List experiments with pagination.
    
    Args:
        skip: Number of records to skip.
        limit: Maximum number of records to return.
        db: Database session.
        
    Returns:
        List[Experiment]: List of experiment instances.
    """
    orchestrator = OrchestratorService(db)
    return await orchestrator.list_experiments(skip=skip, limit=limit)


@router.put("/{experiment_id}", response_model=Experiment)
async def update_experiment(
    experiment_id: UUID,
    experiment_update: ExperimentUpdate,
    db: AsyncSession = Depends(get_db)
) -> Experiment:
    """
    Update an experiment.
    
    Args:
        experiment_id: Experiment identifier.
        experiment_update: Update data.
        db: Database session.
        
    Returns:
        Experiment: Updated experiment instance.
        
    Raises:
        HTTPException: If experiment not found.
    """
    orchestrator = OrchestratorService(db)
    experiment = await orchestrator.update_experiment(experiment_id, experiment_update)
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found"
        )
    
    return experiment


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> None:
    """
    Delete an experiment.
    
    Args:
        experiment_id: Experiment identifier.
        db: Database session.
        
    Raises:
        HTTPException: If experiment not found.
    """
    orchestrator = OrchestratorService(db)
    deleted = await orchestrator.delete_experiment(experiment_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found"
        )
