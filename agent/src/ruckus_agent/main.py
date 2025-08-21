"""Main entry point for RUCKUS agent."""

import asyncio
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from .api.v1.api import router as api_router
from .core.config import settings
from .core.agent import Agent
from . import __version__


# Global agent instance
agent: Agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global agent

    # Startup
    print(f"Starting RUCKUS Agent v{__version__}")
    agent = Agent(settings)
    await agent.start()
    app.state.agent = agent

    yield

    # Shutdown
    print("Shutting down RUCKUS Agent")
    await agent.stop()


app = FastAPI(
    title="RUCKUS Agent",
    description="Worker agent for model benchmarking",
    version=__version__,
    lifespan=lifespan,
)

# Include API router
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": __version__,
        "agent_id": settings.agent_id,
    }


def main():
    """Run the agent."""
    uvicorn.run(
        "ruckus_agent.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()