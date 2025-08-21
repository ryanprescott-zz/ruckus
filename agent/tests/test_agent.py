"""Tests for agent core functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from ruckus_agent.core.agent import Agent
from ruckus_agent.core.config import Settings
from ruckus_agent.core.storage import InMemoryStorage
from ruckus_common.models import AgentType


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        agent_type=AgentType.WHITE_BOX,
        max_concurrent_jobs=2,
        orchestrator_url=None  # No orchestrator for tests
    )


@pytest.fixture
def storage():
    """Create test storage."""
    return InMemoryStorage()


@pytest.fixture
def agent(settings, storage):
    """Create test agent."""
    return Agent(settings, storage)


class TestAgent:
    """Test agent functionality."""

    def test_agent_initialization(self, agent, settings):
        """Test agent initialization."""
        assert agent.settings == settings
        assert agent.agent_id.startswith("agent-")
        assert agent.agent_name.endswith("-white_box")
        assert not agent.registered
        assert len(agent.running_jobs) == 0
        assert agent.startup_time is not None

    def test_agent_id_uniqueness(self, settings, storage):
        """Test that each agent gets a unique ID."""
        agent1 = Agent(settings, storage)
        agent2 = Agent(settings, storage)
        
        assert agent1.agent_id != agent2.agent_id
        assert agent1.agent_name != agent2.agent_name

    @pytest.mark.asyncio
    async def test_get_capabilities_empty(self, agent):
        """Test getting capabilities when empty."""
        capabilities = await agent.get_capabilities()
        assert capabilities == {}

    @pytest.mark.asyncio
    async def test_get_system_info_empty(self, agent):
        """Test getting system info when empty."""
        system_info = await agent.get_system_info()
        assert system_info == {}

    @pytest.mark.asyncio
    async def test_get_status(self, agent):
        """Test getting agent status."""
        status = await agent.get_status()
        
        assert status["agent_id"] == agent.agent_id
        assert status["status"] == "idle"
        assert status["running_jobs"] == []
        assert status["queued_jobs"] == 0
        assert not status["registered"]
        assert "timestamp" in status

    @pytest.mark.asyncio
    @patch('ruckus_agent.core.agent.AgentDetector')
    async def test_detect_capabilities(self, mock_detector_class, agent):
        """Test capability detection."""
        # Mock the detector
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        
        mock_detected_data = {
            "system": {"hostname": "test", "os": "Linux"},
            "cpu": {"cores": 4},
            "gpus": [{"name": "Tesla", "memory_mb": 8000}],
            "frameworks": [{"name": "pytorch", "version": "2.0"}],
            "models": [],
            "hooks": [{"name": "nvidia-smi"}],
            "metrics": [{"name": "latency"}]
        }
        mock_detector.detect_all = AsyncMock(return_value=mock_detected_data)
        
        # Call the method
        await agent._detect_capabilities()
        
        # Verify storage was updated
        system_info = await agent.get_system_info()
        capabilities = await agent.get_capabilities()
        
        assert system_info["system"]["hostname"] == "test"
        assert system_info["cpu"]["cores"] == 4
        assert len(system_info["gpus"]) == 1
        
        assert capabilities["agent_type"] == "white_box"
        assert capabilities["gpu_count"] == 1
        assert "pytorch" in capabilities["frameworks"]
        assert capabilities["monitoring_available"] is True

    @pytest.mark.asyncio
    async def test_storage_integration(self, agent):
        """Test agent storage integration."""
        # Store some data directly in storage
        test_system_info = {
            "system": {"hostname": "storage-test"},
            "cpu": {"cores": 8},
            "gpus": [],
            "frameworks": [],
            "models": [],
            "hooks": [],
            "metrics": []
        }
        
        test_capabilities = {
            "agent_type": "gray_box",
            "gpu_count": 0,
            "frameworks": [],
            "max_concurrent_jobs": 1,
            "monitoring_available": False
        }
        
        await agent.storage.store_system_info(test_system_info)
        await agent.storage.store_capabilities(test_capabilities)
        
        # Verify agent can retrieve it
        retrieved_system = await agent.get_system_info()
        retrieved_capabilities = await agent.get_capabilities()
        
        assert retrieved_system == test_system_info
        assert retrieved_capabilities == test_capabilities