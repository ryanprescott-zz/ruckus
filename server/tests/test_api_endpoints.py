"""Tests for API endpoints in ruckus_server.api.v1.routers.agents."""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient
from ruckus_server.core.agent_manager import AgentAlreadyRegisteredException, AgentNotRegisteredException
from ruckus_server.core.clients.http import ConnectionError, ServiceUnavailableError
from ruckus_common.models import AgentStatus, AgentStatusEnum


class TestAgentEndpoints:
    """Tests for agent-related API endpoints."""

    def test_register_agent_success(self, test_client_with_server, sample_registered_agent_info):
        """Test successful agent registration."""
        # Mock the server method
        test_client_with_server.app.state.agent_manager.register_agent = AsyncMock(
            return_value={
                "agent_id": sample_registered_agent_info.agent_id,
                "registered_at": sample_registered_agent_info.registered_at
            }
        )
        
        response = test_client_with_server.post(
            "/api/v1/agents/register",
            json={"agent_url": "http://localhost:8001"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "agent_id" in data
        assert "registered_at" in data
        assert data["agent_id"] == sample_registered_agent_info.agent_id

    def test_register_agent_invalid_url(self, test_client_with_server):
        """Test agent registration with invalid URL."""
        response = test_client_with_server.post(
            "/api/v1/agents/register",
            json={"agent_url": "not-a-valid-url"}
        )
        
        assert response.status_code == 422  # Validation error

    def test_register_agent_empty_url(self, test_client_with_server):
        """Test agent registration with empty URL."""
        response = test_client_with_server.post(
            "/api/v1/agents/register",
            json={"agent_url": ""}
        )
        
        assert response.status_code == 422  # Validation error

    def test_register_agent_missing_scheme(self, test_client_with_server):
        """Test agent registration with missing scheme."""
        response = test_client_with_server.post(
            "/api/v1/agents/register",
            json={"agent_url": "localhost:8001"}
        )
        
        assert response.status_code == 422  # Validation error

    def test_register_agent_already_registered(self, test_client_with_server):
        """Test registering agent that's already registered."""
        test_client_with_server.app.state.agent_manager.register_agent = AsyncMock(
            side_effect=AgentAlreadyRegisteredException("test-agent", "2023-01-01T00:00:00")
        )
        
        response = test_client_with_server.post(
            "/api/v1/agents/register",
            json={"agent_url": "http://localhost:8001"}
        )
        
        assert response.status_code == 409
        assert "already registered" in response.json()["detail"]

    def test_register_agent_connection_error(self, test_client_with_server):
        """Test agent registration with connection error."""
        test_client_with_server.app.state.agent_manager.register_agent = AsyncMock(
            side_effect=ConnectionError("Connection failed")
        )
        
        response = test_client_with_server.post(
            "/api/v1/agents/register",
            json={"agent_url": "http://unreachable:8001"}
        )
        
        assert response.status_code == 404
        assert "not found or unreachable" in response.json()["detail"]

    def test_register_agent_service_unavailable(self, test_client_with_server):
        """Test agent registration with service unavailable."""
        test_client_with_server.app.state.agent_manager.register_agent = AsyncMock(
            side_effect=ServiceUnavailableError("Service unavailable")
        )
        
        response = test_client_with_server.post(
            "/api/v1/agents/register",
            json={"agent_url": "http://busy:8001"}
        )
        
        assert response.status_code == 503
        assert "temporarily unavailable" in response.json()["detail"]

    def test_register_agent_server_not_initialized(self, test_client):
        """Test agent registration when server not initialized."""
        # test_client doesn't have server in app state
        response = test_client.post(
            "/api/v1/agents/register",
            json={"agent_url": "http://localhost:8001"}
        )
        
        assert response.status_code == 503
        assert "Agent manager not initialized" in response.json()["detail"]

    def test_unregister_agent_success(self, test_client_with_server):
        """Test successful agent unregistration."""
        unregistered_at = datetime.now(timezone.utc)
        test_client_with_server.app.state.agent_manager.unregister_agent = AsyncMock(
            return_value={
                "agent_id": "test-agent-123",
                "unregistered_at": unregistered_at
            }
        )
        
        response = test_client_with_server.post(
            "/api/v1/agents/unregister",
            json={"agent_id": "test-agent-123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == "test-agent-123"
        assert "unregistered_at" in data

    def test_unregister_agent_not_found(self, test_client_with_server):
        """Test unregistering non-existent agent."""
        test_client_with_server.app.state.agent_manager.unregister_agent = AsyncMock(
            side_effect=AgentNotRegisteredException("non-existent-agent")
        )
        
        response = test_client_with_server.post(
            "/api/v1/agents/unregister",
            json={"agent_id": "non-existent-agent"}
        )
        
        assert response.status_code == 404
        assert "No agent with ID" in response.json()["detail"]

    def test_unregister_agent_empty_id(self, test_client_with_server):
        """Test unregistering with empty agent ID."""
        response = test_client_with_server.post(
            "/api/v1/agents/unregister",
            json={"agent_id": ""}
        )
        
        assert response.status_code == 422  # Validation error

    def test_list_agents_success(self, test_client_with_server, sample_registered_agent_info):
        """Test successful agent listing."""
        test_client_with_server.app.state.agent_manager.list_registered_agent_info = AsyncMock(
            return_value=[sample_registered_agent_info]
        )
        
        response = test_client_with_server.get("/api/v1/agents/")
        
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert len(data["agents"]) == 1
        assert data["agents"][0]["agent_id"] == sample_registered_agent_info.agent_id

    def test_list_agents_empty(self, test_client_with_server):
        """Test agent listing when no agents registered."""
        test_client_with_server.app.state.agent_manager.list_registered_agent_info = AsyncMock(
            return_value=[]
        )
        
        response = test_client_with_server.get("/api/v1/agents/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["agents"] == []

    def test_get_agent_info_success(self, test_client_with_server, sample_registered_agent_info):
        """Test successful agent info retrieval."""
        agent_id = sample_registered_agent_info.agent_id
        test_client_with_server.app.state.agent_manager.get_registered_agent_info = AsyncMock(
            return_value=sample_registered_agent_info
        )
        
        response = test_client_with_server.get(f"/api/v1/agents/{agent_id}/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "agent" in data
        assert data["agent"]["agent_id"] == agent_id

    def test_get_agent_info_not_found(self, test_client_with_server):
        """Test agent info retrieval for non-existent agent."""
        agent_id = "non-existent-agent"
        test_client_with_server.app.state.agent_manager.get_registered_agent_info = AsyncMock(
            side_effect=AgentNotRegisteredException(agent_id)
        )
        
        response = test_client_with_server.get(f"/api/v1/agents/{agent_id}/info")
        
        assert response.status_code == 404
        assert "No agent with ID" in response.json()["detail"]

    def test_list_agent_status_success(self, test_client_with_server, sample_agent_status):
        """Test successful agent status listing."""
        test_client_with_server.app.state.agent_manager.list_registered_agent_status = AsyncMock(
            return_value=[sample_agent_status]
        )
        
        response = test_client_with_server.get("/api/v1/agents/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert len(data["agents"]) == 1
        assert data["agents"][0]["agent_id"] == sample_agent_status.agent_id
        assert data["agents"][0]["status"] == sample_agent_status.status.value

    def test_list_agent_status_empty(self, test_client_with_server):
        """Test agent status listing when no agents registered."""
        test_client_with_server.app.state.agent_manager.list_registered_agent_status = AsyncMock(
            return_value=[]
        )
        
        response = test_client_with_server.get("/api/v1/agents/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["agents"] == []

    def test_list_agent_status_with_unavailable_agents(self, test_client_with_server):
        """Test agent status listing with unavailable agents."""
        unavailable_status = AgentStatus(
            agent_id="unavailable-agent",
            status=AgentStatusEnum.UNAVAILABLE,
            running_jobs=[],
            queued_jobs=[],
            uptime_seconds=0.0,
            timestamp=datetime.now(timezone.utc)
        )
        
        test_client_with_server.app.state.agent_manager.list_registered_agent_status = AsyncMock(
            return_value=[unavailable_status]
        )
        
        response = test_client_with_server.get("/api/v1/agents/status")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["agents"]) == 1
        assert data["agents"][0]["status"] == "unavailable"
        assert data["agents"][0]["uptime_seconds"] == 0.0

    def test_get_agent_status_success(self, test_client_with_server, sample_agent_status):
        """Test successful individual agent status retrieval."""
        agent_id = sample_agent_status.agent_id
        test_client_with_server.app.state.agent_manager.get_registered_agent_status = AsyncMock(
            return_value=sample_agent_status
        )
        
        response = test_client_with_server.get(f"/api/v1/agents/{agent_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "agent" in data
        assert data["agent"]["agent_id"] == agent_id
        assert data["agent"]["status"] == sample_agent_status.status.value

    def test_get_agent_status_not_found(self, test_client_with_server):
        """Test individual agent status retrieval for non-existent agent."""
        agent_id = "non-existent-agent"
        test_client_with_server.app.state.agent_manager.get_registered_agent_status = AsyncMock(
            side_effect=AgentNotRegisteredException(agent_id)
        )
        
        response = test_client_with_server.get(f"/api/v1/agents/{agent_id}/status")
        
        assert response.status_code == 404
        assert "No agent with ID" in response.json()["detail"]

    def test_get_agent_status_unavailable(self, test_client_with_server):
        """Test individual agent status retrieval for unavailable agent."""
        agent_id = "unavailable-agent"
        unavailable_status = AgentStatus(
            agent_id=agent_id,
            status=AgentStatusEnum.UNAVAILABLE,
            running_jobs=[],
            queued_jobs=[],
            uptime_seconds=0.0,
            timestamp=datetime.now(timezone.utc)
        )
        
        test_client_with_server.app.state.agent_manager.get_registered_agent_status = AsyncMock(
            return_value=unavailable_status
        )
        
        response = test_client_with_server.get(f"/api/v1/agents/{agent_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["agent"]["status"] == "unavailable"
        assert data["agent"]["uptime_seconds"] == 0.0

    def test_api_endpoints_content_type(self, test_client_with_server):
        """Test that API endpoints return proper content type."""
        # Mock server methods to avoid actual calls
        test_client_with_server.app.state.agent_manager.list_registered_agent_info = AsyncMock(return_value=[])
        test_client_with_server.app.state.agent_manager.list_registered_agent_status = AsyncMock(return_value=[])
        
        # Test various endpoints
        endpoints = [
            "/api/v1/agents/",
            "/api/v1/agents/status"
        ]
        
        for endpoint in endpoints:
            response = test_client_with_server.get(endpoint)
            assert response.status_code == 200
            assert "application/json" in response.headers.get("content-type", "")

    def test_api_error_responses_format(self, test_client_with_server):
        """Test that API error responses have consistent format."""
        test_client_with_server.app.state.agent_manager.register_agent = AsyncMock(
            side_effect=ValueError("Test error")
        )
        
        response = test_client_with_server.post(
            "/api/v1/agents/register",
            json={"agent_url": "http://localhost:8001"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], str)

    def test_concurrent_requests_handling(self, test_client_with_server):
        """Test handling of concurrent API requests."""
        import threading
        import time
        
        test_client_with_server.app.state.agent_manager.list_registered_agent_info = AsyncMock(
            return_value=[]
        )
        
        results = []
        errors = []
        
        def make_request():
            try:
                response = test_client_with_server.get("/api/v1/agents/")
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Make multiple concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5)
        
        # All requests should succeed
        assert len(results) == 5
        assert all(status == 200 for status in results)
        assert len(errors) == 0

    def test_request_validation_edge_cases(self, test_client_with_server):
        """Test request validation with edge cases."""
        # Test with missing request body
        response = test_client_with_server.post("/api/v1/agents/register")
        assert response.status_code == 422
        
        # Test with malformed JSON
        response = test_client_with_server.post(
            "/api/v1/agents/register",
            data="invalid json",
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
        
        # Test with wrong field types
        response = test_client_with_server.post(
            "/api/v1/agents/register",
            json={"agent_url": 12345}  # Should be string
        )
        assert response.status_code == 422

    def test_response_serialization(self, test_client_with_server, sample_registered_agent_info):
        """Test that complex objects are properly serialized in responses."""
        test_client_with_server.app.state.agent_manager.list_registered_agent_info = AsyncMock(
            return_value=[sample_registered_agent_info]
        )
        
        response = test_client_with_server.get("/api/v1/agents/")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify datetime fields are properly serialized
        agent = data["agents"][0]
        assert isinstance(agent["registered_at"], str)
        assert isinstance(agent["last_updated"], str)
        
        # Verify complex nested objects are serialized
        assert isinstance(agent["system_info"], dict)
        assert isinstance(agent["capabilities"], dict)