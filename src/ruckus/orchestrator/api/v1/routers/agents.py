"""
Agents API router for the Ruckus Orchestrator.

This module defines the FastAPI routes for managing agents,
including registration, status updates, and agent-specific functionality.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_db
from ...core.orchestrator import OrchestratorService
from ...core.models import (
    Agent, AgentCreate, AgentUpdate, AgentList, AgentStatus
)

router = APIRouter()


@router.post("/", response_model=Agent, status_code=status.HTTP_201_CREATED)
async def register_agent(
    agent: AgentCreate,
    db: AsyncSession = Depends(get_db)
) -> Agent:
    """
    Register a new agent.
    
    Args:
        agent: Agent registration data.
        db: Database session.
        
    Returns:
        Agent: Registered agent instance.
    """
    orchestrator = OrchestratorService(db)
    return await orchestrator.register_agent(agent)


@router.get("/{agent_id}", response_model=Agent)
async def get_agent(
    agent_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> Agent:
    """
    Get an agent by ID.
    
    Args:
        agent_id: Agent identifier.
        db: Database session.
        
    Returns:
        Agent: Agent instance.
        
    Raises:
        HTTPException: If agent not found.
    """
    orchestrator = OrchestratorService(db)
    agent = await orchestrator.get_agent(agent_id)
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    
    return agent


@router.get("/", response_model=List[Agent])
async def list_agents(
    status: Optional[AgentStatus] = Query(None, description="Filter by agent status"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    db: AsyncSession = Depends(get_db)
) -> List[Agent]:
    """
    List agents with optional filtering and pagination.
    
    Args:
        status: Filter by agent status.
        skip: Number of records to skip.
        limit: Maximum number of records to return.
        db: Database session.
        
    Returns:
        List[Agent]: List of agent instances.
    """
    orchestrator = OrchestratorService(db)
    return await orchestrator.list_agents(
        status=status,
        skip=skip,
        limit=limit
    )


@router.post("/{agent_id}/heartbeat", response_model=Agent)
async def agent_heartbeat(
    agent_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> Agent:
    """
    Update agent heartbeat timestamp.
    
    Args:
        agent_id: Agent identifier.
        db: Database session.
        
    Returns:
        Agent: Updated agent instance.
        
    Raises:
        HTTPException: If agent not found.
    """
    orchestrator = OrchestratorService(db)
    agent = await orchestrator.update_agent_heartbeat(agent_id)
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    
    return agent
