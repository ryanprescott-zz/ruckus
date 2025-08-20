"""
Main API router for Ruckus Agent v1.

This module creates the main API router for the agent service
and includes health check endpoints.
"""

from fastapi import APIRouter

from ...core.agent import agent_service
from ...core.models import HealthCheck, AgentStatus

api_router = APIRouter()


@api_router.get("/health", response_model=HealthCheck)
async def health_check() -> HealthCheck:
    """
    Health check endpoint.
    
    Returns:
        HealthCheck: Agent health information.
    """
    return agent_service.get_health()


@api_router.get("/status", response_model=AgentStatus)
async def get_status() -> AgentStatus:
    """
    Get agent status.
    
    Returns:
        AgentStatus: Current agent status information.
    """
    return agent_service.get_status()
