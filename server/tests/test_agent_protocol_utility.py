"""Tests for AgentProtocolUtility."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

from ruckus_server.core.agent import AgentProtocolUtility
from ruckus_server.core.clients.http import ConnectionError, ServiceUnavailableError
from ruckus_common.models import AgentInfo, AgentType, AgentInfoResponse, RegisteredAgentInfo


class TestAgentProtocolUtility:
    """Tests for AgentProtocolUtility class."""

    def test_initialization(self, agent_settings, http_client_settings):
        """Test AgentProtocolUtility initialization."""
        utility = AgentProtocolUtility(agent_settings, http_client_settings)
        
        assert utility.agent_settings == agent_settings
        assert utility.http_client_settings == http_client_settings
        assert utility.logger is not None

    def test_build_info_url_basic(self, agent_protocol_utility):
        """Test building info URL with basic agent URL."""
        agent_url = "http://localhost:8001"
        info_url = agent_protocol_utility.build_info_url(agent_url)
        
        expected = "http://localhost:8001/api/v1/info"
        assert info_url == expected

    def test_build_info_url_with_trailing_slash(self, agent_protocol_utility):
        """Test building info URL when agent URL has trailing slash."""
        agent_url = "http://localhost:8001/"
        info_url = agent_protocol_utility.build_info_url(agent_url)
        
        expected = "http://localhost:8001/api/v1/info"
        assert info_url == expected

    def test_build_info_url_with_path(self, agent_protocol_utility):
        """Test building info URL when agent URL has existing path."""
        agent_url = "http://localhost:8001/agent"
        info_url = agent_protocol_utility.build_info_url(agent_url)
        
        expected = "http://localhost:8001/agent/api/v1/info"
        assert info_url == expected

    def test_build_info_url_custom_endpoint_path(self, agent_settings, http_client_settings):
        """Test building info URL with custom endpoint path."""
        agent_settings.info_endpoint_path = "/custom/info"
        utility = AgentProtocolUtility(agent_settings, http_client_settings)
        
        agent_url = "http://localhost:8001"
        info_url = utility.build_info_url(agent_url)
        
        expected = "http://localhost:8001/custom/info"
        assert info_url == expected

    def test_build_info_url_endpoint_with_leading_slash(self, agent_settings, http_client_settings):
        """Test building info URL when endpoint path has leading slash."""
        agent_settings.info_endpoint_path = "/api/v1/info"
        utility = AgentProtocolUtility(agent_settings, http_client_settings)
        
        agent_url = "http://localhost:8001/"
        info_url = utility.build_info_url(agent_url)
        
        expected = "http://localhost:8001/api/v1/info"
        assert info_url == expected

    @pytest.mark.asyncio
    async def test_get_agent_info_success(self, agent_protocol_utility, mock_agent_server_responses):
        """Test successful agent info retrieval."""
        mock_response_data = mock_agent_server_responses["info"]
        
        with patch('ruckus_server.core.agent.HttpClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_with_retry.return_value = mock_response_data
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            agent_url = "http://localhost:8001"
            result = await agent_protocol_utility.get_agent_info(agent_url)

            assert isinstance(result, AgentInfoResponse)
            assert result.agent_info.agent_id == "test-agent-123"
            assert result.agent_info.agent_type == AgentType.WHITE_BOX
            
            # Verify correct URL was called
            expected_url = "http://localhost:8001/api/v1/info"
            mock_client.get_with_retry.assert_called_once_with(expected_url)

    @pytest.mark.asyncio
    async def test_get_agent_info_connection_error(self, agent_protocol_utility):
        """Test agent info retrieval with connection error."""
        with patch('ruckus_server.core.agent.HttpClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_with_retry.side_effect = ConnectionError("Connection failed")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            agent_url = "http://localhost:8001"
            
            with pytest.raises(ConnectionError) as exc_info:
                await agent_protocol_utility.get_agent_info(agent_url)
            
            assert "Connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_agent_info_service_unavailable(self, agent_protocol_utility):
        """Test agent info retrieval with service unavailable error."""
        with patch('ruckus_server.core.agent.HttpClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_with_retry.side_effect = ServiceUnavailableError("Service unavailable")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            agent_url = "http://localhost:8001"
            
            with pytest.raises(ServiceUnavailableError) as exc_info:
                await agent_protocol_utility.get_agent_info(agent_url)
            
            assert "Service unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_agent_info_invalid_response(self, agent_protocol_utility):
        """Test agent info retrieval with invalid response format."""
        invalid_response = {"invalid": "data"}
        
        with patch('ruckus_server.core.agent.HttpClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_with_retry.return_value = invalid_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            agent_url = "http://localhost:8001"
            
            with pytest.raises(ConnectionError) as exc_info:
                await agent_protocol_utility.get_agent_info(agent_url)
            
            assert "Failed to get agent info" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_agent_info_unexpected_exception(self, agent_protocol_utility):
        """Test agent info retrieval with unexpected exception."""
        with patch('ruckus_server.core.agent.HttpClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_with_retry.side_effect = ValueError("Unexpected error")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            agent_url = "http://localhost:8001"
            
            with pytest.raises(ConnectionError) as exc_info:
                await agent_protocol_utility.get_agent_info(agent_url)
            
            assert "Failed to get agent info" in str(exc_info.value)

    def test_create_registered_agent_info(self, agent_protocol_utility, sample_agent_info_response):
        """Test creating RegisteredAgentInfo from AgentInfoResponse."""
        agent_url = "http://localhost:8001"
        
        result = agent_protocol_utility.create_registered_agent_info(
            sample_agent_info_response, 
            agent_url
        )
        
        assert isinstance(result, RegisteredAgentInfo)
        assert result.agent_id == sample_agent_info_response.agent_info.agent_id
        assert result.agent_name == sample_agent_info_response.agent_info.agent_name
        assert result.agent_type == sample_agent_info_response.agent_info.agent_type
        assert result.system_info == sample_agent_info_response.agent_info.system_info
        assert result.capabilities == sample_agent_info_response.agent_info.capabilities
        assert result.last_updated == sample_agent_info_response.agent_info.last_updated
        assert result.agent_url == agent_url
        assert isinstance(result.registered_at, datetime)

    def test_create_registered_agent_info_with_none_fields(self, agent_protocol_utility):
        """Test creating RegisteredAgentInfo with None optional fields."""
        agent_info = AgentInfo(
            agent_id="test-agent",
            agent_name=None,  # Optional field
            agent_type=AgentType.BLACK_BOX,
            system_info={},
            capabilities={},
            last_updated=datetime.now(timezone.utc)
        )
        
        agent_info_response = AgentInfoResponse(agent_info=agent_info)
        agent_url = "http://localhost:8001"
        
        result = agent_protocol_utility.create_registered_agent_info(
            agent_info_response, 
            agent_url
        )
        
        assert result.agent_id == "test-agent"
        assert result.agent_name is None
        assert result.agent_type == AgentType.BLACK_BOX

    def test_create_registered_agent_info_preserves_all_data(self, agent_protocol_utility):
        """Test that all data is preserved when creating RegisteredAgentInfo."""
        complex_system_info = {
            "hostname": "complex-host",
            "os": "ubuntu",
            "version": "20.04",
            "memory": "32GB",
            "cpu": {"cores": 8, "model": "Intel i7"}
        }
        
        complex_capabilities = {
            "frameworks": ["pytorch", "tensorflow", "transformers"],
            "gpu_count": 4,
            "models": ["llama", "bert"],
            "batch_sizes": [1, 2, 4, 8]
        }
        
        agent_info = AgentInfo(
            agent_id="complex-agent",
            agent_name="Complex Test Agent",
            agent_type=AgentType.GRAY_BOX,
            system_info=complex_system_info,
            capabilities=complex_capabilities,
            last_updated=datetime.now(timezone.utc)
        )
        
        agent_info_response = AgentInfoResponse(agent_info=agent_info)
        agent_url = "https://complex-agent.example.com:8443/api"
        
        result = agent_protocol_utility.create_registered_agent_info(
            agent_info_response, 
            agent_url
        )
        
        # Verify all complex data is preserved
        assert result.system_info == complex_system_info
        assert result.capabilities == complex_capabilities
        assert result.agent_url == agent_url
        assert result.system_info["cpu"]["cores"] == 8
        assert "pytorch" in result.capabilities["frameworks"]

    @pytest.mark.asyncio
    async def test_http_client_context_manager_usage(self, agent_protocol_utility):
        """Test that HttpClient is properly used as context manager."""
        with patch('ruckus_server.core.agent.HttpClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_with_retry.return_value = {"agent_info": {"agent_id": "test"}}
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            agent_url = "http://localhost:8001"
            
            try:
                await agent_protocol_utility.get_agent_info(agent_url)
            except Exception:
                pass  # We expect this to fail due to invalid response format
            
            # Verify context manager methods were called
            mock_client.__aenter__.assert_called_once()
            mock_client.__aexit__.assert_called_once()

    def test_logger_initialization(self, agent_protocol_utility):
        """Test that logger is properly initialized."""
        assert agent_protocol_utility.logger is not None
        assert agent_protocol_utility.logger.name == "ruckus_server.core.agent"