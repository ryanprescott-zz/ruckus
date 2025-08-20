"""
Pytest configuration and fixtures for orchestrator tests.

This module provides common test fixtures and configuration
for the orchestrator test suite.
"""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from ..core.database import Base, get_db
from ..core.orchestrator import OrchestratorService
from ..main import create_app


@pytest_asyncio.fixture
async def test_db():
    """
    Create a test database session.
    
    Yields:
        AsyncSession: Test database session.
    """
    # Create in-memory SQLite database for testing
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False}
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session factory
    TestSessionLocal = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with TestSessionLocal() as session:
        yield session


@pytest.fixture
def orchestrator_service(test_db):
    """
    Create an orchestrator service instance for testing.
    
    Args:
        test_db: Test database session.
        
    Returns:
        OrchestratorService: Orchestrator service instance.
    """
    return OrchestratorService(test_db)


@pytest.fixture
def test_app(test_db):
    """
    Create a test FastAPI application.
    
    Args:
        test_db: Test database session.
        
    Returns:
        FastAPI: Test application instance.
    """
    app = create_app()
    
    # Override database dependency
    async def override_get_db():
        yield test_db
    
    app.dependency_overrides[get_db] = override_get_db
    return app
