"""Tests for pydantic models in ruckus_server.api.v1.models."""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from ruckus_server.api.v1.models import (
    RegisterAgentRequest,
    RegisterAgentResponse,
    UnregisterAgentRequest,
    UnregisterAgentResponse,
    ListAgentInfoResponse,
    GetAgentInfoResponse,
    ListAgentStatusResponse,
    GetAgentStatusResponse
)
from ruckus_common.models import AgentStatus, AgentStatusEnum


class TestRegisterAgentRequest:
    """Tests for RegisterAgentRequest model."""

    def test_valid_http_url(self):
        """Test valid HTTP URL."""
        request = RegisterAgentRequest(agent_url="http://localhost:8001")
        assert request.agent_url == "http://localhost:8001"

    def test_valid_https_url(self):
        """Test valid HTTPS URL."""
        request = RegisterAgentRequest(agent_url="https://agent.example.com")
        assert request.agent_url == "https://agent.example.com"

    def test_valid_url_with_port(self):
        """Test valid URL with port."""
        request = RegisterAgentRequest(agent_url="http://192.168.1.100:8080")
        assert request.agent_url == "http://192.168.1.100:8080"

    def test_valid_localhost_url(self):
        """Test valid localhost URL."""
        request = RegisterAgentRequest(agent_url="http://localhost:3000")
        assert request.agent_url == "http://localhost:3000"

    def test_empty_url_raises_validation_error(self):
        """Test empty URL raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            RegisterAgentRequest(agent_url="")
        
        error = exc_info.value.errors()[0]
        assert error["type"] == "value_error"
        assert "cannot be empty" in str(error["msg"])

    def test_no_scheme_raises_validation_error(self):
        """Test URL without scheme raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            RegisterAgentRequest(agent_url="localhost:8001")
        
        error = exc_info.value.errors()[0]
        assert error["type"] == "value_error"
        # The error message might vary, so check for key parts
        error_msg = str(error["msg"]).lower()
        assert "scheme" in error_msg or "http" in error_msg

    def test_invalid_scheme_raises_validation_error(self):
        """Test URL with invalid scheme raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            RegisterAgentRequest(agent_url="ftp://agent.example.com")
        
        error = exc_info.value.errors()[0]
        assert error["type"] == "value_error"
        assert "must be http or https" in str(error["msg"])

    def test_no_hostname_raises_validation_error(self):
        """Test URL without hostname raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            RegisterAgentRequest(agent_url="http://")
        
        error = exc_info.value.errors()[0]
        assert error["type"] == "value_error"
        assert "must include a hostname" in str(error["msg"])

    def test_invalid_hostname_raises_validation_error(self):
        """Test URL with invalid hostname raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            RegisterAgentRequest(agent_url="http://invalid..hostname")
        
        error = exc_info.value.errors()[0]
        assert error["type"] == "value_error"
        assert "Invalid hostname format" in str(error["msg"])

    def test_invalid_url_format_raises_validation_error(self):
        """Test completely invalid URL format raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            RegisterAgentRequest(agent_url="not-a-url-at-all")
        
        error = exc_info.value.errors()[0]
        assert error["type"] == "value_error"


class TestRegisterAgentResponse:
    """Tests for RegisterAgentResponse model."""

    def test_valid_response(self):
        """Test valid response creation."""
        now = datetime.now(timezone.utc)
        response = RegisterAgentResponse(
            agent_id="test-agent-123",
            registered_at=now
        )
        assert response.agent_id == "test-agent-123"
        assert response.registered_at == now

    def test_response_serialization(self):
        """Test response can be serialized to dict."""
        now = datetime.now(timezone.utc)
        response = RegisterAgentResponse(
            agent_id="test-agent-123",
            registered_at=now
        )
        data = response.model_dump()
        assert data["agent_id"] == "test-agent-123"
        assert data["registered_at"] == now


class TestUnregisterAgentRequest:
    """Tests for UnregisterAgentRequest model."""

    def test_valid_agent_id(self):
        """Test valid agent ID."""
        request = UnregisterAgentRequest(agent_id="test-agent-123")
        assert request.agent_id == "test-agent-123"

    def test_empty_agent_id_raises_validation_error(self):
        """Test empty agent ID raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            UnregisterAgentRequest(agent_id="")
        
        error = exc_info.value.errors()[0]
        assert error["type"] == "string_too_short"


class TestUnregisterAgentResponse:
    """Tests for UnregisterAgentResponse model."""

    def test_valid_response(self):
        """Test valid response creation."""
        now = datetime.now(timezone.utc)
        response = UnregisterAgentResponse(
            agent_id="test-agent-123",
            unregistered_at=now
        )
        assert response.agent_id == "test-agent-123"
        assert response.unregistered_at == now


class TestListAgentInfoResponse:
    """Tests for ListAgentInfoResponse model."""

    def test_empty_agent_list(self):
        """Test response with empty agent list."""
        response = ListAgentInfoResponse(agents=[])
        assert response.agents == []

    def test_agent_list_with_agents(self, sample_registered_agent_info):
        """Test response with agent list."""
        response = ListAgentInfoResponse(agents=[sample_registered_agent_info])
        assert len(response.agents) == 1
        assert response.agents[0].agent_id == sample_registered_agent_info.agent_id


class TestGetAgentInfoResponse:
    """Tests for GetAgentInfoResponse model."""

    def test_valid_response(self, sample_registered_agent_info):
        """Test valid response creation."""
        response = GetAgentInfoResponse(agent=sample_registered_agent_info)
        assert response.agent.agent_id == sample_registered_agent_info.agent_id


class TestListAgentStatusResponse:
    """Tests for ListAgentStatusResponse model."""

    def test_empty_status_list(self):
        """Test response with empty status list."""
        response = ListAgentStatusResponse(agents=[])
        assert response.agents == []

    def test_status_list_with_agents(self, sample_agent_status):
        """Test response with status list."""
        response = ListAgentStatusResponse(agents=[sample_agent_status])
        assert len(response.agents) == 1
        assert response.agents[0].agent_id == sample_agent_status.agent_id


class TestGetAgentStatusResponse:
    """Tests for GetAgentStatusResponse model."""

    def test_valid_response(self, sample_agent_status):
        """Test valid response creation."""
        response = GetAgentStatusResponse(agent=sample_agent_status)
        assert response.agent.agent_id == sample_agent_status.agent_id
        assert response.agent.status == sample_agent_status.status

    def test_unavailable_agent_response(self):
        """Test response with unavailable agent status."""
        unavailable_status = AgentStatus(
            agent_id="unreachable-agent",
            status=AgentStatusEnum.UNAVAILABLE,
            running_jobs=[],
            queued_jobs=[],
            uptime_seconds=0.0,
            timestamp=datetime.now(timezone.utc)
        )
        response = GetAgentStatusResponse(agent=unavailable_status)
        assert response.agent.status == AgentStatusEnum.UNAVAILABLE
        assert response.agent.uptime_seconds == 0.0