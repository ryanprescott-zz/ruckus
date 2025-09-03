"""Tests for agent core functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from ruckus_agent.core.agent import Agent
from ruckus_agent.core.config import Settings
from ruckus_agent.core.storage import InMemoryStorage
from ruckus_common.models import AgentType, AgentStatusEnum


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        agent_type=AgentType.WHITE_BOX,
        max_concurrent_jobs=2
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
        assert len(agent.running_jobs) == 0
        assert agent.startup_time is not None

    def test_agent_id_uniqueness(self, settings, storage):
        """Test that each agent gets a unique ID."""
        agent1 = Agent(settings, storage)
        agent2 = Agent(settings, storage)
        
        assert agent1.agent_id != agent2.agent_id
        assert agent1.agent_name != agent2.agent_name

    @pytest.mark.asyncio
    async def test_get_system_info_empty(self, agent):
        """Test getting system info when empty."""
        system_info = await agent.get_system_info()
        assert system_info == {}

    @pytest.mark.asyncio
    async def test_get_status(self, agent):
        """Test getting agent status."""
        status = await agent.get_status()
        
        assert status.agent_id == agent.agent_id
        assert status.status == AgentStatusEnum.IDLE
        assert status.running_jobs == []
        assert status.queued_jobs == []
        assert status.uptime_seconds >= 0
        assert status.timestamp is not None

    @pytest.mark.asyncio
    @patch('ruckus_agent.core.agent.AgentDetector')
    async def test_storage_integration(self, mock_detector, agent):
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
        
        await agent.storage.store_system_info(test_system_info)
        
        # Verify agent can retrieve it
        retrieved_system = await agent.get_system_info()
        
        assert retrieved_system == test_system_info