"""Tests for agent API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock
from ruckus_agent.main import app
from ruckus_agent.core.agent import Agent
from ruckus_agent.core.config import Settings
from ruckus_agent.core.storage import InMemoryStorage
from ruckus_common.models import AgentType


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(agent_type=AgentType.WHITE_BOX)


@pytest.fixture
def storage():
    """Create test storage."""
    return InMemoryStorage()


@pytest.fixture
def test_agent(settings, storage):
    """Create and configure test agent."""
    import asyncio
    
    async def setup_agent():
        agent = Agent(settings, storage)
        
        # Pre-populate with test data
        test_system_info = {
            "system": {"hostname": "test-agent", "os": "Linux"},
            "cpu": {"cores": 4, "model": "Intel"},
            "gpus": [{"name": "Tesla", "memory_mb": 8000}],
            "frameworks": [{"name": "pytorch", "version": "2.0"}],
            "models": [],
            "hooks": [{"name": "nvidia-smi"}],
            "metrics": [{"name": "latency"}, {"name": "throughput"}]
        }
        
        await storage.store_system_info(test_system_info)
        
        return agent
    
    # Run the async setup
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        agent = loop.run_until_complete(setup_agent())
    finally:
        loop.close()
    
    return agent


@pytest.fixture
def client(test_agent):
    """Create test client with mocked agent."""
    # Mock the agent in app state
    app.state.agent = test_agent
    return TestClient(app)


class TestAgentAPI:
    """Test agent API endpoints."""

    def test_api_info_endpoint(self, client):
        """Test the API info endpoint."""
        response = client.get("/api/v1/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["version"] == "v1"
        assert data["type"] == "agent"
        assert "/info" in data["endpoints"]

    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "agent_id" in data

    def test_info_endpoint(self, client, test_agent):
        """Test the info endpoint."""
        response = client.get("/api/v1/info")
        assert response.status_code == 200
        
        data = response.json()
        agent_info = data["agent_info"]
        assert agent_info["agent_id"] == test_agent.agent_id
        assert agent_info["agent_name"] == test_agent.agent_name
        assert agent_info["agent_type"] == "white_box"
        
        # Check system info structure
        system_info = agent_info["system_info"]
        assert system_info["system"]["hostname"] == "test-agent"
        assert system_info["cpu"]["cores"] == 4
        assert len(system_info["gpus"]) == 1
        assert system_info["gpus"][0]["name"] == "Tesla"
        
        # Verify frameworks are present in system_info
        assert len(system_info["frameworks"]) == 1
        assert system_info["frameworks"][0]["name"] == "pytorch"

    def test_status_endpoint(self, client, test_agent):
        """Test the status endpoint."""
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["agent_id"] == test_agent.agent_id
        assert data["status"] == "idle"
        assert data["running_jobs"] == []
        assert data["queued_jobs"] == []

    def test_api_response_models(self, client):
        """Test that API responses match expected Pydantic models."""
        # Test info endpoint response model
        info_response = client.get("/api/v1/info")
        assert info_response.status_code == 200
        info_data = info_response.json()
        
        # Info endpoint returns AgentInfoResponse with agent_info field
        assert "agent_info" in info_data
        agent_info = info_data["agent_info"]
        
        required_info_fields = ["agent_id", "agent_type", "system_info", "last_updated"]
        for field in required_info_fields:
            assert field in agent_info

    def test_concurrent_requests(self, client):
        """Test handling multiple sequential requests to endpoints."""
        # Test that multiple sequential requests work
        responses = []
        for _ in range(5):
            response = client.get("/api/v1/info")
            responses.append(response)
        
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "agent_info" in data