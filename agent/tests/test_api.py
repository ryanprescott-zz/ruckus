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
        
        test_capabilities = {
            "agent_type": "white_box",
            "gpu_count": 1,
            "frameworks": ["pytorch"],
            "max_concurrent_jobs": 1,
            "monitoring_available": True
        }
        
        await storage.store_system_info(test_system_info)
        await storage.store_capabilities(test_capabilities)
        
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
        assert "/register" in data["endpoints"]
        assert "/info" in data["endpoints"]

    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "agent_id" in data

    def test_register_endpoint(self, client, test_agent):
        """Test the registration endpoint."""
        response = client.get("/api/v1/register")
        assert response.status_code == 200
        
        data = response.json()
        assert data["agent_id"] == test_agent.agent_id
        assert data["agent_name"] == test_agent.agent_name
        assert data["message"] == "Agent registered successfully"
        assert "server_time" in data

    def test_info_endpoint(self, client, test_agent):
        """Test the info endpoint."""
        response = client.get("/api/v1/info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["agent_id"] == test_agent.agent_id
        assert data["agent_name"] == test_agent.agent_name
        assert data["agent_type"] == "white_box"
        
        # Check system info structure
        system_info = data["system_info"]
        assert system_info["system"]["hostname"] == "test-agent"
        assert system_info["cpu"]["cores"] == 4
        assert len(system_info["gpus"]) == 1
        assert system_info["gpus"][0]["name"] == "Tesla"
        
        # Check capabilities structure
        capabilities = data["capabilities"]
        assert capabilities["agent_type"] == "white_box"
        assert capabilities["gpu_count"] == 1
        assert "pytorch" in capabilities["frameworks"]
        assert capabilities["monitoring_available"] is True

    def test_capabilities_endpoint(self, client):
        """Test the capabilities endpoint (legacy)."""
        response = client.get("/api/v1/capabilities")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, dict)

    def test_status_endpoint(self, client, test_agent):
        """Test the status endpoint."""
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["agent_id"] == test_agent.agent_id
        assert data["status"] == "idle"
        assert data["running_jobs"] == []
        assert data["queued_jobs"] == 0

    def test_api_response_models(self, client):
        """Test that API responses match expected Pydantic models."""
        # Test register endpoint response model
        register_response = client.get("/api/v1/register")
        assert register_response.status_code == 200
        register_data = register_response.json()
        
        required_register_fields = ["agent_id", "server_time"]
        for field in required_register_fields:
            assert field in register_data
        
        # Test info endpoint response model
        info_response = client.get("/api/v1/info")
        assert info_response.status_code == 200
        info_data = info_response.json()
        
        required_info_fields = ["agent_id", "agent_type", "system_info", "capabilities", "last_updated"]
        for field in required_info_fields:
            assert field in info_data

    def test_concurrent_requests(self, client):
        """Test handling concurrent requests to endpoints."""
        import concurrent.futures
        import requests
        
        # Note: This would need to be adapted for async testing in a real scenario
        # For now, just test that multiple sequential requests work
        responses = []
        for _ in range(5):
            response = client.get("/api/v1/register")
            responses.append(response)
        
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "agent_id" in data