"""Main entry point for RUCKUS server."""

import asyncio
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from . import __version__
from .api.v1.api import api_router
from .core.config import settings
from .core.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    print(f"Starting RUCKUS Server v{__version__}")
    await init_db()
    yield
    # Shutdown
    print("Shutting down RUCKUS Server")


app = FastAPI(
    title="RUCKUS Server",
    description="Orchestrator for distributed model benchmarking",
    version=__version__,
    lifespan=lifespan,
)

# Include API router
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": __version__}


def main():
    """Run the server."""
    uvicorn.run(
        "ruckus_server.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
