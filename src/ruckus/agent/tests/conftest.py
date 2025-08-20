"""
Pytest configuration and fixtures for agent tests.

This module provides common test fixtures and configuration
for the agent test suite.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ..core.agent import AgentService
from ..main import create_app


@pytest.fixture
def mock_orchestrator_client():
    """
    Create a mock orchestrator client.
    
    Returns:
        AsyncMock: Mock orchestrator client.
    """
    client = AsyncMock()
    client.post = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def agent_service(mock_orchestrator_client):
    """
    Create an agent service instance for testing.
    
    Args:
        mock_orchestrator_client: Mock orchestrator client.
        
    Returns:
        AgentService: Agent service instance.
    """
    service = AgentService()
    service.orchestrator_client = mock_orchestrator_client
    return service


@pytest.fixture
def test_app():
    """
    Create a test FastAPI application.
    
    Returns:
        FastAPI: Test application instance.
    """
    return create_app()
