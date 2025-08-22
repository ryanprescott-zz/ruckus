"""Main entry point for RUCKUS server."""

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
app.include_router(api_router, prefix=settings.base_router_path)

# Mount /static to serve self-hosted swagger UI assets
app.mount(settings.openapi_prefix, StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    """
    Root endpoint providing basic API information.
    
    Returns:
        dict: Basic API information and status
    """
    return {
        "message": "Ruckus server app",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        dict: Health status information
    """
    return {"status": "healthy"}


@app.get(f"{uri_prefix}/docs", include_in_schema=False)
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
