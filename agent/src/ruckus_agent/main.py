"""Main entry point for RUCKUS agent."""

import asyncio
import logging
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles

from .api.v1.api import router as api_router
from .core.config import settings
from .core.agent import Agent
from . import __version__

logger = logging.getLogger(__name__)


# Global agent instance
agent: Agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global agent

    # Startup
    logger.info(f"Starting RUCKUS Agent v{__version__}")
    agent = Agent(settings)
    await agent.start()
    app.state.agent = agent

    yield

    # Shutdown
    logger.info("Shutting down RUCKUS Agent")
    await agent.stop()


app = FastAPI(
    title="RUCKUS Agent",
    description="Worker agent for model benchmarking",
    version=__version__,
    lifespan=lifespan,
)

# Include API router
app.include_router(api_router, prefix=settings.api_prefix, tags=["API"])

# Mount /static to serve self-hosted swagger UI assets
app.mount(settings.openapi_prefix, StaticFiles(directory="static"), name="static")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": __version__,
        "agent_id": settings.agent_id,
    }


@app.get(f"{settings.api_prefix}/docs", include_in_schema=False)
async def self_hosted_swagger_ui_html():
    """
    Serve the self-hosted Swagger UI HTML.
    
    Returns:
        HTML: The Swagger UI HTML page
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title,
        swagger_favicon_url="/icons/favicon.ico",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url=f"{settings.openapi_prefix}/swagger-ui-bundle.js",
        swagger_css_url=f"{settings.openapi_prefix}/swagger-ui.css",
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    """
    Serve the Swagger UI OAuth2 redirect HTML.
    
    Returns:
        HTML: The Swagger UI OAuth2 redirect page
    """
    return get_swagger_ui_oauth2_redirect_html()


# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)