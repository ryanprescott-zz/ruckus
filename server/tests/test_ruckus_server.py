"""Tests for RuckusServer core functionality."""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

from ruckus_server.core.server import (
    RuckusServer,
    AgentAlreadyRegisteredException,
    AgentNotRegisteredException
)
from ruckus_server.core.clients.http import ConnectionError, ServiceUnavailableError
from ruckus_common.models import AgentStatus, AgentStatusEnum, RegisteredAgentInfo


class TestRuckusServer:
    """Tests for RuckusServer core functionality."""

    def test_initialization(self, ruckus_server_settings):
        """Test RuckusServer initialization."""
        server = RuckusServer(ruckus_server_settings)
        
        assert server.settings == ruckus_server_settings
        assert server.logger is not None
        assert server.storage is None  # Not started yet

    def test_initialization_with_default_settings(self):
        """Test RuckusServer initialization with default settings."""
        server = RuckusServer()
        
        assert server.settings is not None
        assert server.logger is not None

    @pytest.mark.asyncio
    async def test_start_and_stop(self, ruckus_server):
        """Test server start and stop lifecycle."""
        # Server fixture already starts the server
        assert ruckus_server.storage is not None
        
        # Test that we can stop it
        await ruckus_server.stop()

    @pytest.mark.asyncio
    async def test_register_agent_success(self, ruckus_server, mock_agent_server_responses):
        """Test successful agent registration."""
        agent_url = "http://localhost:8001"
        mock_response = mock_agent_server_responses["info"]
        
        with patch('ruckus_server.core.server.AgentProtocolUtility') as mock_utility_class:
            # Create a mixed mock - sync methods use regular Mock, async methods use AsyncMock
            mock_utility = Mock()
            mock_utility.get_agent_info = AsyncMock(return_value=Mock(
                agent_info=Mock(agent_id="test-agent-123")
            ))
            mock_utility.create_registered_agent_info.return_value = Mock(
                agent_id="test-agent-123",
                registered_at=datetime.now(timezone.utc)
            )
            mock_utility_class.return_value = mock_utility
            
            # Mock storage to return None for existing agent check
            ruckus_server.storage.get_registered_agent_info = AsyncMock(return_value=None)
            ruckus_server.storage.register_agent = AsyncMock(return_value=True)
            
            result = await ruckus_server.register_agent(agent_url)
            
            assert "agent_id" in result
            assert "registered_at" in result
            assert result["agent_id"] == "test-agent-123"

    @pytest.mark.asyncio
    async def test_register_agent_already_registered(self, ruckus_server, sample_registered_agent_info):
        """Test registering agent that's already registered raises exception."""
        agent_url = "http://localhost:8001"
        
        with patch('ruckus_server.core.server.AgentProtocolUtility') as mock_utility_class:
            mock_utility = Mock()
            mock_utility.get_agent_info = AsyncMock(return_value=Mock(
                agent_info=Mock(agent_id=sample_registered_agent_info.agent_id)
            ))
            mock_utility_class.return_value = mock_utility
            
            # Mock storage to return existing agent
            ruckus_server.storage.get_registered_agent_info = AsyncMock(
                return_value=sample_registered_agent_info
            )
            
            with pytest.raises(AgentAlreadyRegisteredException) as exc_info:
                await ruckus_server.register_agent(agent_url)
            
            assert exc_info.value.agent_id == sample_registered_agent_info.agent_id

    @pytest.mark.asyncio
    async def test_register_agent_connection_error(self, ruckus_server):
        """Test agent registration with connection error."""
        agent_url = "http://unreachable:8001"
        
        with patch('ruckus_server.core.server.AgentProtocolUtility') as mock_utility_class:
            mock_utility = Mock()
            mock_utility.get_agent_info = AsyncMock(side_effect=ConnectionError("Connection failed"))
            mock_utility_class.return_value = mock_utility
            
            with pytest.raises(ConnectionError) as exc_info:
                await ruckus_server.register_agent(agent_url)
            
            assert "Connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_register_agent_service_unavailable(self, ruckus_server):
        """Test agent registration with service unavailable error."""
        agent_url = "http://busy:8001"
        
        with patch('ruckus_server.core.server.AgentProtocolUtility') as mock_utility_class:
            mock_utility = Mock()
            mock_utility.get_agent_info = AsyncMock(side_effect=ServiceUnavailableError("Service unavailable"))
            mock_utility_class.return_value = mock_utility
            
            with pytest.raises(ServiceUnavailableError) as exc_info:
                await ruckus_server.register_agent(agent_url)
            
            assert "Service unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_register_agent_storage_failure(self, ruckus_server):
        """Test agent registration with storage failure."""
        agent_url = "http://localhost:8001"
        
        with patch('ruckus_server.core.server.AgentProtocolUtility') as mock_utility_class:
            mock_utility = Mock()
            mock_utility.get_agent_info = AsyncMock(return_value=Mock(
                agent_info=Mock(agent_id="test-agent-123")
            ))
            mock_utility.create_registered_agent_info.return_value = Mock(
                agent_id="test-agent-123"
            )
            mock_utility_class.return_value = mock_utility
            
            # Mock storage failure
            ruckus_server.storage.get_registered_agent_info = AsyncMock(return_value=None)
            ruckus_server.storage.register_agent = AsyncMock(return_value=False)
            
            with pytest.raises(RuntimeError) as exc_info:
                await ruckus_server.register_agent(agent_url)
            
            assert "Failed to store agent information" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unregister_agent_success(self, ruckus_server):
        """Test successful agent unregistration."""
        agent_id = "test-agent-123"
        
        # Mock storage methods
        ruckus_server.storage.agent_exists = AsyncMock(return_value=True)
        ruckus_server.storage.remove_agent = AsyncMock(return_value=True)
        
        result = await ruckus_server.unregister_agent(agent_id)
        
        assert "agent_id" in result
        assert "unregistered_at" in result
        assert result["agent_id"] == agent_id

    @pytest.mark.asyncio
    async def test_unregister_agent_not_registered(self, ruckus_server):
        """Test unregistering non-existent agent raises exception."""
        agent_id = "non-existent-agent"
        
        ruckus_server.storage.agent_exists = AsyncMock(return_value=False)
        
        with pytest.raises(AgentNotRegisteredException) as exc_info:
            await ruckus_server.unregister_agent(agent_id)
        
        assert exc_info.value.agent_id == agent_id

    @pytest.mark.asyncio
    async def test_unregister_agent_storage_failure(self, ruckus_server):
        """Test agent unregistration with storage failure."""
        agent_id = "test-agent-123"
        
        ruckus_server.storage.agent_exists = AsyncMock(return_value=True)
        ruckus_server.storage.remove_agent = AsyncMock(return_value=False)
        
        with pytest.raises(RuntimeError) as exc_info:
            await ruckus_server.unregister_agent(agent_id)
        
        assert "Failed to remove agent from database" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_registered_agent_info(self, ruckus_server, sample_registered_agent_info):
        """Test listing registered agent information."""
        expected_agents = [sample_registered_agent_info]
        
        ruckus_server.storage.list_registered_agent_info = AsyncMock(return_value=expected_agents)
        
        result = await ruckus_server.list_registered_agent_info()
        
        assert result == expected_agents
        assert len(result) == 1
        assert result[0].agent_id == sample_registered_agent_info.agent_id

    @pytest.mark.asyncio
    async def test_list_registered_agent_info_empty(self, ruckus_server):
        """Test listing when no agents are registered."""
        ruckus_server.storage.list_registered_agent_info = AsyncMock(return_value=[])
        
        result = await ruckus_server.list_registered_agent_info()
        
        assert result == []

    @pytest.mark.asyncio
    async def test_get_registered_agent_info(self, ruckus_server, sample_registered_agent_info):
        """Test getting specific agent information."""
        agent_id = sample_registered_agent_info.agent_id
        
        ruckus_server.storage.get_registered_agent_info = AsyncMock(
            return_value=sample_registered_agent_info
        )
        
        result = await ruckus_server.get_registered_agent_info(agent_id)
        
        assert result == sample_registered_agent_info
        assert result.agent_id == agent_id

    @pytest.mark.asyncio
    async def test_get_registered_agent_info_not_found(self, ruckus_server):
        """Test getting non-existent agent raises exception."""
        agent_id = "non-existent-agent"
        
        ruckus_server.storage.get_registered_agent_info = AsyncMock(return_value=None)
        
        with pytest.raises(AgentNotRegisteredException) as exc_info:
            await ruckus_server.get_registered_agent_info(agent_id)
        
        assert exc_info.value.agent_id == agent_id

    @pytest.mark.asyncio
    async def test_list_registered_agent_status_success(self, ruckus_server, sample_registered_agent_info):
        """Test listing all agent statuses successfully."""
        mock_status = AgentStatus(
            agent_id=sample_registered_agent_info.agent_id,
            status=AgentStatusEnum.IDLE,
            running_jobs=[],
            queued_jobs=[],
            uptime_seconds=3600.0,
            timestamp=datetime.now(timezone.utc)
        )
        
        ruckus_server.storage.list_registered_agent_info = AsyncMock(
            return_value=[sample_registered_agent_info]
        )
        
        with patch('ruckus_server.core.server.SimpleHttpClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_json.return_value = mock_status.model_dump()
            mock_client_class.return_value = mock_client
            
            result = await ruckus_server.list_registered_agent_status()
            
            assert len(result) == 1
            assert result[0].agent_id == sample_registered_agent_info.agent_id
            assert result[0].status == AgentStatusEnum.IDLE

    @pytest.mark.asyncio
    async def test_list_registered_agent_status_unavailable_agent(self, ruckus_server, sample_registered_agent_info):
        """Test listing agent statuses when agent is unavailable."""
        ruckus_server.storage.list_registered_agent_info = AsyncMock(
            return_value=[sample_registered_agent_info]
        )
        
        with patch('ruckus_server.core.server.SimpleHttpClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_json.return_value = None  # Simulate connection failure
            mock_client_class.return_value = mock_client
            
            result = await ruckus_server.list_registered_agent_status()
            
            assert len(result) == 1
            assert result[0].agent_id == sample_registered_agent_info.agent_id
            assert result[0].status == AgentStatusEnum.UNAVAILABLE
            assert result[0].uptime_seconds == 0.0
            assert result[0].running_jobs == []
            assert result[0].queued_jobs == []

    @pytest.mark.asyncio
    async def test_list_registered_agent_status_empty(self, ruckus_server):
        """Test listing agent statuses when no agents are registered."""
        ruckus_server.storage.list_registered_agent_info = AsyncMock(return_value=[])
        
        result = await ruckus_server.list_registered_agent_status()
        
        assert result == []

    @pytest.mark.asyncio
    async def test_list_registered_agent_status_concurrent_calls(self, ruckus_server, registered_agent_info_factory):
        """Test concurrent status calls for multiple agents."""
        # Create multiple agents
        agents = [
            registered_agent_info_factory(agent_id=f"agent-{i}", agent_url=f"http://agent{i}:8001")
            for i in range(3)
        ]
        
        ruckus_server.storage.list_registered_agent_info = AsyncMock(return_value=agents)
        
        # Mock responses for each agent
        mock_responses = [
            {"agent_id": f"agent-{i}", "status": "idle", "running_jobs": [], "queued_jobs": [], 
             "uptime_seconds": 3600.0, "timestamp": datetime.now(timezone.utc).isoformat()}
            for i in range(3)
        ]
        
        with patch('ruckus_server.core.server.SimpleHttpClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_json.side_effect = mock_responses
            mock_client_class.return_value = mock_client
            
            result = await ruckus_server.list_registered_agent_status()
            
            assert len(result) == 3
            assert all(status.status == AgentStatusEnum.IDLE for status in result)
            # Verify all agents were called concurrently
            assert mock_client.get_json.call_count == 3

    @pytest.mark.asyncio
    async def test_get_registered_agent_status_success(self, ruckus_server, sample_registered_agent_info):
        """Test getting specific agent status successfully."""
        mock_status_data = {
            "agent_id": sample_registered_agent_info.agent_id,
            "status": "active",
            "running_jobs": ["job-1"],
            "queued_jobs": [],
            "uptime_seconds": 7200.0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        ruckus_server.storage.get_registered_agent_info = AsyncMock(
            return_value=sample_registered_agent_info
        )
        
        with patch('ruckus_server.core.server.SimpleHttpClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_json.return_value = mock_status_data
            mock_client_class.return_value = mock_client
            
            result = await ruckus_server.get_registered_agent_status(sample_registered_agent_info.agent_id)
            
            assert result.agent_id == sample_registered_agent_info.agent_id
            assert result.status == AgentStatusEnum.ACTIVE
            assert result.running_jobs == ["job-1"]
            assert result.uptime_seconds == 7200.0

    @pytest.mark.asyncio
    async def test_get_registered_agent_status_not_found(self, ruckus_server):
        """Test getting status for non-existent agent."""
        agent_id = "non-existent-agent"
        
        ruckus_server.storage.get_registered_agent_info = AsyncMock(return_value=None)
        
        with pytest.raises(AgentNotRegisteredException) as exc_info:
            await ruckus_server.get_registered_agent_status(agent_id)
        
        assert exc_info.value.agent_id == agent_id

    @pytest.mark.asyncio
    async def test_get_registered_agent_status_unavailable(self, ruckus_server, sample_registered_agent_info):
        """Test getting status when agent is unavailable."""
        ruckus_server.storage.get_registered_agent_info = AsyncMock(
            return_value=sample_registered_agent_info
        )
        
        with patch('ruckus_server.core.server.SimpleHttpClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_json.return_value = None  # Simulate connection failure
            mock_client_class.return_value = mock_client
            
            result = await ruckus_server.get_registered_agent_status(sample_registered_agent_info.agent_id)
            
            assert result.agent_id == sample_registered_agent_info.agent_id
            assert result.status == AgentStatusEnum.UNAVAILABLE
            assert result.uptime_seconds == 0.0

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, ruckus_server, sample_registered_agent_info):
        """Test health check when system is healthy."""
        ruckus_server.storage.health_check = AsyncMock(return_value=True)
        ruckus_server.storage.list_registered_agent_info = AsyncMock(
            return_value=[sample_registered_agent_info]
        )
        
        result = await ruckus_server.health_check()
        
        assert result["status"] == "healthy"
        assert result["storage"] == "healthy"
        assert result["agents"] == 1

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_storage(self, ruckus_server):
        """Test health check when storage is unhealthy."""
        ruckus_server.storage.health_check = AsyncMock(return_value=False)
        ruckus_server.storage.list_registered_agent_info = AsyncMock(return_value=[])
        
        result = await ruckus_server.health_check()
        
        assert result["status"] == "unhealthy"
        assert result["storage"] == "unhealthy"
        assert result["agents"] == 0

    @pytest.mark.asyncio
    async def test_health_check_no_storage(self, ruckus_server_settings):
        """Test health check when storage is not initialized."""
        server = RuckusServer(ruckus_server_settings)
        # Don't start the server, so storage remains None
        
        result = await server.health_check()
        
        assert result["status"] == "unhealthy"
        assert result["storage"] == "not_initialized"
        assert result["agents"] == 0

    @pytest.mark.asyncio
    async def test_storage_backend_not_initialized_errors(self, ruckus_server_settings):
        """Test operations fail gracefully when storage not initialized."""
        server = RuckusServer(ruckus_server_settings)
        # Don't start the server
        
        with pytest.raises(RuntimeError, match="Storage backend not initialized"):
            await server.register_agent("http://test.com")
        
        with pytest.raises(RuntimeError, match="Storage backend not initialized"):
            await server.unregister_agent("test-agent")
        
        with pytest.raises(RuntimeError, match="Storage backend not initialized"):
            await server.list_registered_agent_info()
        
        with pytest.raises(RuntimeError, match="Storage backend not initialized"):
            await server.get_registered_agent_info("test-agent")
        
        with pytest.raises(RuntimeError, match="Storage backend not initialized"):
            await server.list_registered_agent_status()
        
        with pytest.raises(RuntimeError, match="Storage backend not initialized"):
            await server.get_registered_agent_status("test-agent")