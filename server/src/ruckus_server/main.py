"""Main entry point for RUCKUS server."""

import logging
import asyncio
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles


from . import __version__
from .api.v1.api import api_router
from .core.config import Settings
from .core.agent_manager import AgentManager
from .core.experiment_manager import ExperimentManager

logger = logging.getLogger(__name__)

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info(f"Starting RUCKUS Server v{__version__}")
    
    # Initialize AgentManager with settings
    agent_manager = AgentManager(settings.agent_manager)
    await agent_manager.start()
    
    # Initialize ExperimentManager with settings
    experiment_manager = ExperimentManager(settings.experiment_manager)
    await experiment_manager.start()
    
    # Make managers available to the app
    app.state.agent_manager = agent_manager
    app.state.experiment_manager = experiment_manager
    
    yield
    
    # Shutdown
    logger.info("Shutting down RUCKUS Server")
    if hasattr(app.state, 'agent_manager') and app.state.agent_manager:
        await app.state.agent_manager.stop()
    if hasattr(app.state, 'experiment_manager') and app.state.experiment_manager:
        await app.state.experiment_manager.stop()


app = FastAPI(
    title="RUCKUS Server",
    description="Server for managing agents and coordinating model benchmarking tasks",
    version=__version__,
    lifespan=lifespan,
)

# Include API router
app.include_router(api_router, prefix=settings.app.api_prefix, tags=["API"])

# Mount /static to serve self-hosted swagger UI assets
app.mount(settings.app.openapi_prefix, StaticFiles(directory="static"), name="static")


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        dict: Health status information
    """
    if hasattr(app.state, 'agent_manager') and app.state.agent_manager:
        return await app.state.agent_manager.health_check()
    return {"status": "unhealthy", "reason": "agent_manager_not_initialized"}


@app.get(f"{settings.app.api_prefix}/docs", include_in_schema=False)
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
        swagger_js_url=f"{settings.app.openapi_prefix}/swagger-ui-bundle.js",
        swagger_css_url=f"{settings.app.openapi_prefix}/swagger-ui.css",
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
