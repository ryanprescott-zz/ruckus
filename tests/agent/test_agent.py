"""
Unit tests for the agent service.

This module contains tests for the core agent functionality
including job execution, status reporting, and orchestrator communication.
"""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock
import httpx

from ..core.models import (
    JobRequest, JobExecutionStatus, AgentCapabilities
)


class TestAgentService:
    """Test cases for the AgentService class."""

    def test_detect_capabilities(self, agent_service):
        """Test capability detection."""
        capabilities = agent_service._detect_capabilities()
        
        assert isinstance(capabilities, AgentCapabilities)
        assert capabilities.runtime == "transformers"
        assert capabilities.platform == "cuda"

    @pytest.mark.asyncio
    async def test_register_with_orchestrator_success(self, agent_service, mock_orchestrator_client):
        """Test successful registration with orchestrator."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": str(uuid4())}
        mock_response.raise_for_status = MagicMock()
        mock_orchestrator_client.post.return_value = mock_response
        
        result = await agent_service.register_with_orchestrator()
        
        assert result is True
        mock_orchestrator_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_with_orchestrator_failure(self, agent_service, mock_orchestrator_client):
        """Test failed registration with orchestrator."""
        # Mock failed response
        mock_orchestrator_client.post.side_effect = httpx.HTTPError("Connection failed")
        
        result = await agent_service.register_with_orchestrator()
        
        assert result is False

    @pytest.mark.asyncio
    async def test_send_heartbeat_success(self, agent_service, mock_orchestrator_client):
        """Test successful heartbeat."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_orchestrator_client.post.return_value = mock_response
        
        result = await agent_service.send_heartbeat()
        
        assert result is True
        mock_orchestrator_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_heartbeat_failure(self, agent_service, mock_orchestrator_client):
        """Test failed heartbeat."""
        # Mock failed response
        mock_orchestrator_client.post.side_effect = httpx.HTTPError("Connection failed")
        
        result = await agent_service.send_heartbeat()
        
        assert result is False

    @pytest.mark.asyncio
    async def test_execute_job_success(self, agent_service):
        """Test successful job execution."""
        job_request = JobRequest(
            job_id=uuid4(),
            experiment_id=uuid4(),
            config={"batch_size": 32},
            model_name="test-model",
            runtime="transformers",
            platform="cuda",
            task_config={"task": "summarization"},
            data_config={"dataset": "test-data"}
        )
        
        result = await agent_service.execute_job(job_request)
        
        assert result.job_id == job_request.job_id
        assert result.status == JobExecutionStatus.COMPLETED
        assert result.results is not None
        assert "model_output" in result.results

    @pytest.mark.asyncio
    async def test_execute_job_with_error(self, agent_service, monkeypatch):
        """Test job execution with error."""
        # Mock the _execute_llm_task to raise an exception
        async def mock_execute_llm_task(job_request):
            raise ValueError("Mock execution error")
        
        monkeypatch.setattr(agent_service, "_execute_llm_task", mock_execute_llm_task)
        
        job_request = JobRequest(
            job_id=uuid4(),
            experiment_id=uuid4(),
            config={"batch_size": 32},
            model_name="test-model",
            runtime="transformers",
            platform="cuda",
            task_config={"task": "summarization"},
            data_config={"dataset": "test-data"}
        )
        
        result = await agent_service.execute_job(job_request)
        
        assert result.job_id == job_request.job_id
        assert result.status == JobExecutionStatus.FAILED
        assert result.error_message == "Mock execution error"

    def test_get_status(self, agent_service):
        """Test getting agent status."""
        status = agent_service.get_status()
        
        assert status.agent_id == agent_service.agent_id
        assert status.capabilities == agent_service.capabilities
        assert status.current_jobs == 0
        assert status.total_jobs_completed == 0

    def test_get_health(self, agent_service):
        """Test getting agent health."""
        health = agent_service.get_health()
        
        assert health.status == "healthy"
        assert health.agent_id == agent_service.agent_id
        assert health.uptime >= 0
        assert health.current_jobs == 0

    @pytest.mark.asyncio
    async def test_job_tracking(self, agent_service):
        """Test that jobs are properly tracked during execution."""
        job_request = JobRequest(
            job_id=uuid4(),
            experiment_id=uuid4(),
            config={"batch_size": 32},
            model_name="test-model",
            runtime="transformers",
            platform="cuda",
            task_config={"task": "summarization"},
            data_config={"dataset": "test-data"}
        )
        
        # Check initial state
        assert len(agent_service.current_jobs) == 0
        assert agent_service.completed_jobs == 0
        
        # Execute job
        result = await agent_service.execute_job(job_request)
        
        # Check final state
        assert len(agent_service.current_jobs) == 0  # Job should be removed after completion
        assert agent_service.completed_jobs == 1
        assert result.status == JobExecutionStatus.COMPLETED
