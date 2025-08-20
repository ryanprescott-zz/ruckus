"""
Main API router for Ruckus Orchestrator v1.

This module aggregates all the v1 API routers and creates the main
API router for the orchestrator service.
"""

from fastapi import APIRouter

from .routers import experiments, jobs, agents

api_router = APIRouter()

# Include all routers
api_router.include_router(
    experiments.router,
    prefix="/experiments",
    tags=["experiments"]
)

api_router.include_router(
    jobs.router,
    prefix="/jobs", 
    tags=["jobs"]
)

api_router.include_router(
    agents.router,
    prefix="/agents",
    tags=["agents"]
)


@api_router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Health status information.
    """
    return {"status": "healthy", "service": "ruckus-orchestrator"}
