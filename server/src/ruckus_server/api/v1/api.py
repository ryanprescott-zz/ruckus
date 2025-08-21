"""Main API router for v1."""

from fastapi import APIRouter

from .routers import agents, experiments, jobs

api_router = APIRouter()

# Include routers
api_router.include_router(agents.router, prefix="/agents", tags=["agents"])
api_router.include_router(experiments.router, prefix="/experiments", tags=["experiments"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])


@api_router.get("/")
async def api_info():
    """API information endpoint."""
    return {
        "version": "v1",
        "endpoints": [
            "/agents",
            "/experiments",
            "/jobs",
        ]
    }