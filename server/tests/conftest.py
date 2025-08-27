"""Shared test fixtures and configuration for ruckus_server tests."""

import asyncio
import pytest
import pytest_asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, AsyncGenerator
from unittest.mock import AsyncMock, Mock

import httpx
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine

from ruckus_server.main import app
from ruckus_server.core.server import RuckusServer
from ruckus_server.core.config import RuckusServerSettings, SQLiteSettings, AgentSettings, HttpClientSettings
from ruckus_server.core.storage.sqlite import SQLiteStorageBackend
from ruckus_server.core.clients.http import HttpClient
from ruckus_server.core.clients.simple_http import SimpleHttpClient
from ruckus_server.core.agent import AgentProtocolUtility
from ruckus_common.models import AgentInfo, AgentType, AgentInfoResponse, RegisteredAgentInfo, AgentStatus, AgentStatusEnum


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_db_path():
    """Create a temporary database file path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def sqlite_settings(temp_db_path):
    """Create SQLite settings with temporary database."""
    return SQLiteSettings(
        database_path=temp_db_path,
        echo_sql=False,
        max_retries=1,
        retry_delay=0.1
    )


@pytest.fixture
def agent_settings():
    """Create agent settings for testing."""
    return AgentSettings(
        info_endpoint_path="/api/v1/info"
    )


@pytest.fixture
def http_client_settings():
    """Create HTTP client settings for testing."""
    return HttpClientSettings(
        connection_timeout=1.0,
        read_timeout=1.0,
        max_retries=1,
        initial_backoff=0.1,
        max_backoff=0.2
    )


@pytest.fixture
def ruckus_server_settings(sqlite_settings, agent_settings, http_client_settings):
    """Create RuckusServer settings for testing."""
    return RuckusServerSettings(
        log_level="DEBUG",
        log_config_file="logging.yml",
        storage_backend="sqlite",
        sqlite=sqlite_settings,
        agent=agent_settings,
        http_client=http_client_settings
    )


@pytest_asyncio.fixture
async def sqlite_storage(sqlite_settings):
    """Create and initialize SQLite storage backend."""
    storage = SQLiteStorageBackend(sqlite_settings)
    await storage.initialize()
    try:
        yield storage
    finally:
        await storage.close()


@pytest_asyncio.fixture
async def ruckus_server(ruckus_server_settings):
    """Create and start RuckusServer instance."""
    server = RuckusServer(ruckus_server_settings)
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


@pytest.fixture
def http_client(http_client_settings):
    """Create HttpClient instance."""
    return HttpClient(http_client_settings)


@pytest.fixture
def simple_http_client():
    """Create SimpleHttpClient instance."""
    return SimpleHttpClient(timeout_seconds=1.0)


@pytest.fixture
def agent_protocol_utility(agent_settings, http_client_settings):
    """Create AgentProtocolUtility instance."""
    return AgentProtocolUtility(agent_settings, http_client_settings)


@pytest.fixture
def sample_agent_info():
    """Create sample AgentInfo for testing."""
    return AgentInfo(
        agent_id="test-agent-123",
        agent_name="Test Agent",
        agent_type=AgentType.WHITE_BOX,
        system_info={
            "hostname": "test-host",
            "os": "linux",
            "python_version": "3.9.0"
        },
        capabilities={
            "frameworks": ["pytorch", "transformers"],
            "gpu_count": 2
        },
        last_updated=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_agent_info_response(sample_agent_info):
    """Create sample AgentInfoResponse for testing."""
    return AgentInfoResponse(agent_info=sample_agent_info)


@pytest.fixture
def sample_registered_agent_info(sample_agent_info):
    """Create sample RegisteredAgentInfo for testing."""
    return RegisteredAgentInfo(
        agent_id=sample_agent_info.agent_id,
        agent_name=sample_agent_info.agent_name,
        agent_type=sample_agent_info.agent_type,
        system_info=sample_agent_info.system_info,
        capabilities=sample_agent_info.capabilities,
        last_updated=sample_agent_info.last_updated,
        agent_url="http://localhost:8001",
        registered_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_agent_status():
    """Create sample AgentStatus for testing."""
    return AgentStatus(
        agent_id="test-agent-123",
        status=AgentStatusEnum.IDLE,
        running_jobs=[],
        queued_jobs=["job-1", "job-2"],
        uptime_seconds=3600.0,
        timestamp=datetime.now(timezone.utc)
    )


@pytest.fixture
def mock_httpx_client():
    """Create mock httpx client."""
    client = Mock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.fixture
def mock_agent_server_responses():
    """Create mock responses for agent server endpoints."""
    return {
        "info": {
            "agent_info": {
                "agent_id": "test-agent-123",
                "agent_name": "Test Agent",
                "agent_type": "white_box",
                "system_info": {
                    "hostname": "test-host",
                    "os": "linux",
                    "python_version": "3.9.0"
                },
                "capabilities": {
                    "frameworks": ["pytorch", "transformers"],
                    "gpu_count": 2
                },
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
        },
        "status": {
            "agent_id": "test-agent-123",
            "status": "idle",
            "running_jobs": [],
            "queued_jobs": ["job-1", "job-2"],
            "uptime_seconds": 3600.0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }


@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def test_client_with_server(ruckus_server):
    """Create FastAPI test client with initialized server."""
    # Set the server instance in app state
    if not hasattr(app.state, 'server'):
        app.state.server = None
    original_server = app.state.server
    app.state.server = ruckus_server
    
    try:
        with TestClient(app) as client:
            yield client
    finally:
        # Cleanup
        app.state.server = original_server


@pytest.fixture
def agent_urls():
    """Create sample agent URLs for testing."""
    return [
        "http://agent1:8001",
        "http://agent2:8002",
        "http://agent3:8003"
    ]


@pytest.fixture
def invalid_agent_urls():
    """Create invalid agent URLs for testing."""
    return [
        "not-a-url",
        "ftp://invalid-scheme.com",
        "http://",
        "https://no-hostname",
        ""
    ]


# Test data generators
@pytest.fixture
def agent_info_factory():
    """Factory function to create AgentInfo instances."""
    def _create_agent_info(
        agent_id: str = None,
        agent_name: str = None,
        agent_type: AgentType = AgentType.WHITE_BOX,
        **kwargs
    ):
        return AgentInfo(
            agent_id=agent_id or f"agent-{datetime.now(timezone.utc).timestamp()}",
            agent_name=agent_name,
            agent_type=agent_type,
            system_info=kwargs.get("system_info", {}),
            capabilities=kwargs.get("capabilities", {}),
            last_updated=kwargs.get("last_updated", datetime.now(timezone.utc))
        )
    return _create_agent_info


@pytest.fixture
def registered_agent_info_factory():
    """Factory function to create RegisteredAgentInfo instances."""
    def _create_registered_agent_info(
        agent_id: str = None,
        agent_url: str = None,
        **kwargs
    ):
        base_agent_id = agent_id or f"agent-{datetime.now(timezone.utc).timestamp()}"
        return RegisteredAgentInfo(
            agent_id=base_agent_id,
            agent_name=kwargs.get("agent_name"),
            agent_type=kwargs.get("agent_type", AgentType.WHITE_BOX),
            system_info=kwargs.get("system_info", {}),
            capabilities=kwargs.get("capabilities", {}),
            last_updated=kwargs.get("last_updated", datetime.now(timezone.utc)),
            agent_url=agent_url or f"http://agent-{base_agent_id}:8001",
            registered_at=kwargs.get("registered_at", datetime.now(timezone.utc))
        )
    return _create_registered_agent_info