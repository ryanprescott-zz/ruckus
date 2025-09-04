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
from ruckus_server.core.agent_manager import AgentManager
from ruckus_server.core.experiment_manager import ExperimentManager
from ruckus_server.core.config import AgentManagerSettings, ExperimentManagerSettings, StorageSettings, StorageBackendType, SQLiteSettings, AgentSettings, HttpClientSettings
from ruckus_server.core.storage.sqlite import SQLiteStorageBackend
from ruckus_server.core.clients.http import HttpClient
from ruckus_server.core.clients.simple_http import SimpleHttpClient
from ruckus_server.core.agent import AgentProtocolUtility
from ruckus_common.models import AgentInfo, AgentType, AgentInfoResponse, RegisteredAgentInfo, AgentStatus, AgentStatusEnum, ExperimentSpec, TaskType


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
def storage_settings(sqlite_settings):
    """Create storage settings for testing."""
    return StorageSettings(
        storage_backend=StorageBackendType.SQLITE,
        sqlite=sqlite_settings
    )


@pytest.fixture
def agent_manager_settings(storage_settings, agent_settings, http_client_settings):
    """Create AgentManager settings for testing."""
    return AgentManagerSettings(
        log_level="DEBUG",
        log_config_file="logging.yml",
        storage=storage_settings,
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
async def agent_manager(agent_manager_settings):
    """Create and start AgentManager instance."""
    manager = AgentManager(agent_manager_settings)
    await manager.start()
    try:
        yield manager
    finally:
        await manager.stop()


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
def test_client_with_server(agent_manager):
    """Create FastAPI test client with initialized agent manager."""
    # Set the agent manager instance in app state
    if not hasattr(app.state, 'agent_manager'):
        app.state.agent_manager = None
    original_agent_manager = app.state.agent_manager
    app.state.agent_manager = agent_manager
    
    try:
        with TestClient(app) as client:
            yield client
    finally:
        # Cleanup
        app.state.agent_manager = original_agent_manager


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


# Experiment-related fixtures
@pytest.fixture
def experiment_manager_settings(storage_settings):
    """Create ExperimentManager settings for testing."""
    return ExperimentManagerSettings(
        log_level="DEBUG",
        log_config_file="logging.yml",
        storage=storage_settings
    )


@pytest_asyncio.fixture
async def experiment_manager(experiment_manager_settings):
    """Create and start ExperimentManager instance."""
    manager = ExperimentManager(experiment_manager_settings)
    await manager.start()
    try:
        yield manager
    finally:
        await manager.stop()


@pytest.fixture
def sample_experiment_spec():
    """Create sample ExperimentSpec for testing."""
    return ExperimentSpec(
        experiment_id="test-experiment-123",
        name="Test Experiment",
        description="A test experiment for unit testing",
        models=["test-model"],
        task_type=TaskType.SUMMARIZATION,
        priority=5,
        timeout_seconds=7200,
        owner="test-user",
        tags=["test", "unit-test"]
    )


@pytest.fixture
def test_client_with_experiment_manager(experiment_manager):
    """Create FastAPI test client with initialized experiment manager."""
    # Set the experiment manager instance in app state
    if not hasattr(app.state, 'experiment_manager'):
        app.state.experiment_manager = None
    original_experiment_manager = app.state.experiment_manager
    app.state.experiment_manager = experiment_manager
    
    try:
        with TestClient(app) as client:
            yield client
    finally:
        # Cleanup
        app.state.experiment_manager = original_experiment_manager


@pytest.fixture
def test_client_with_both_managers(agent_manager, experiment_manager):
    """Create FastAPI test client with both agent and experiment managers."""
    # Set both managers in app state
    if not hasattr(app.state, 'agent_manager'):
        app.state.agent_manager = None
    if not hasattr(app.state, 'experiment_manager'):
        app.state.experiment_manager = None
        
    original_agent_manager = app.state.agent_manager
    original_experiment_manager = app.state.experiment_manager
    
    app.state.agent_manager = agent_manager
    app.state.experiment_manager = experiment_manager
    
    try:
        with TestClient(app) as client:
            yield client
    finally:
        # Cleanup
        app.state.agent_manager = original_agent_manager
        app.state.experiment_manager = original_experiment_manager


@pytest.fixture
def experiment_spec_factory():
    """Factory function to create ExperimentSpec instances."""
    def _create_experiment_spec(
        experiment_id: str = None,
        name: str = None,
        models: list = None,
        task_type: TaskType = TaskType.SUMMARIZATION,
        **kwargs
    ):
        return ExperimentSpec(
            experiment_id=experiment_id or f"experiment-{datetime.now(timezone.utc).timestamp()}",
            name=name or "Test Experiment",
            description=kwargs.get("description", "Test experiment description"),
            models=models or ["test-model"],
            task_type=task_type,
            priority=kwargs.get("priority", 0),
            timeout_seconds=kwargs.get("timeout_seconds", 3600),
            owner=kwargs.get("owner"),
            tags=kwargs.get("tags", [])
        )
    return _create_experiment_spec