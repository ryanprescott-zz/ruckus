"""Tests for storage backend implementations."""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from ruckus_server.core.storage.sqlite import SQLiteStorageBackend
from ruckus_server.core.storage.postgresql import PostgreSQLStorageBackend
from ruckus_server.core.storage.base import Agent
from ruckus_server.core.config import SQLiteSettings, PostgreSQLSettings
from ruckus_common.models import AgentType, RegisteredAgentInfo


class TestSQLiteStorageBackend:
    """Tests for SQLite storage backend."""

    @pytest.mark.asyncio
    async def test_initialization_creates_tables(self, sqlite_settings):
        """Test that initialization creates database tables."""
        storage = SQLiteStorageBackend(sqlite_settings)
        await storage.initialize()
        
        # Verify database file was created
        assert Path(sqlite_settings.database_path).exists()
        
        await storage.close()

    @pytest.mark.asyncio
    async def test_health_check_success(self, sqlite_storage):
        """Test successful health check."""
        result = await sqlite_storage.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_register_agent_success(self, sqlite_storage, sample_registered_agent_info):
        """Test successful agent registration."""
        result = await sqlite_storage.register_agent(sample_registered_agent_info)
        assert result is True

    @pytest.mark.asyncio
    async def test_agent_exists_after_registration(self, sqlite_storage, sample_registered_agent_info):
        """Test agent exists check after registration."""
        await sqlite_storage.register_agent(sample_registered_agent_info)
        
        exists = await sqlite_storage.agent_exists(sample_registered_agent_info.agent_id)
        assert exists is True

    @pytest.mark.asyncio
    async def test_agent_does_not_exist_initially(self, sqlite_storage):
        """Test agent does not exist initially."""
        exists = await sqlite_storage.agent_exists("non-existent-agent")
        assert exists is False

    @pytest.mark.asyncio
    async def test_get_registered_agent_info(self, sqlite_storage, sample_registered_agent_info):
        """Test retrieving registered agent info."""
        await sqlite_storage.register_agent(sample_registered_agent_info)
        
        retrieved = await sqlite_storage.get_registered_agent_info(sample_registered_agent_info.agent_id)
        
        assert retrieved is not None
        assert retrieved.agent_id == sample_registered_agent_info.agent_id
        assert retrieved.agent_name == sample_registered_agent_info.agent_name
        assert retrieved.agent_type == sample_registered_agent_info.agent_type
        assert retrieved.agent_url == sample_registered_agent_info.agent_url

    @pytest.mark.asyncio
    async def test_get_nonexistent_agent_returns_none(self, sqlite_storage):
        """Test retrieving non-existent agent returns None."""
        retrieved = await sqlite_storage.get_registered_agent_info("non-existent-agent")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_list_registered_agent_info_empty(self, sqlite_storage):
        """Test listing agents when none are registered."""
        agents = await sqlite_storage.list_registered_agent_info()
        assert agents == []

    @pytest.mark.asyncio
    async def test_list_registered_agent_info_with_agents(self, sqlite_storage, registered_agent_info_factory):
        """Test listing agents when some are registered."""
        # Register multiple agents
        agent1 = registered_agent_info_factory(agent_id="agent-1", agent_url="http://agent1:8001")
        agent2 = registered_agent_info_factory(agent_id="agent-2", agent_url="http://agent2:8002")
        
        await sqlite_storage.register_agent(agent1)
        await sqlite_storage.register_agent(agent2)
        
        agents = await sqlite_storage.list_registered_agent_info()
        
        assert len(agents) == 2
        agent_ids = {agent.agent_id for agent in agents}
        assert agent_ids == {"agent-1", "agent-2"}

    @pytest.mark.asyncio
    async def test_remove_agent_success(self, sqlite_storage, sample_registered_agent_info):
        """Test successful agent removal."""
        await sqlite_storage.register_agent(sample_registered_agent_info)
        
        # Verify agent exists
        exists_before = await sqlite_storage.agent_exists(sample_registered_agent_info.agent_id)
        assert exists_before is True
        
        # Remove agent
        result = await sqlite_storage.remove_agent(sample_registered_agent_info.agent_id)
        assert result is True
        
        # Verify agent no longer exists
        exists_after = await sqlite_storage.agent_exists(sample_registered_agent_info.agent_id)
        assert exists_after is False

    @pytest.mark.asyncio
    async def test_remove_nonexistent_agent(self, sqlite_storage):
        """Test removing non-existent agent returns False."""
        result = await sqlite_storage.remove_agent("non-existent-agent")
        assert result is False

    @pytest.mark.asyncio
    async def test_update_agent_status(self, sqlite_storage, sample_registered_agent_info):
        """Test updating agent status."""
        await sqlite_storage.register_agent(sample_registered_agent_info)
        
        result = await sqlite_storage.update_agent_status(sample_registered_agent_info.agent_id, "inactive")
        assert result is True

    @pytest.mark.asyncio
    async def test_update_agent_heartbeat(self, sqlite_storage, sample_registered_agent_info):
        """Test updating agent heartbeat."""
        await sqlite_storage.register_agent(sample_registered_agent_info)
        
        result = await sqlite_storage.update_agent_heartbeat(sample_registered_agent_info.agent_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_create_experiment(self, sqlite_storage):
        """Test creating an experiment."""
        result = await sqlite_storage.create_experiment(
            experiment_id="exp-1",
            name="Test Experiment",
            description="A test experiment",
            config={"param1": "value1"}
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_create_job(self, sqlite_storage):
        """Test creating a job."""
        result = await sqlite_storage.create_job(
            job_id="job-1",
            experiment_id="exp-1",
            config={"model": "test-model"}
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_database_error_handling(self, sqlite_settings):
        """Test error handling when database operations fail."""
        # Use an invalid database path to trigger errors
        invalid_settings = SQLiteSettings(
            database_path="/invalid/path/database.db",
            echo_sql=False
        )
        
        storage = SQLiteStorageBackend(invalid_settings)
        
        with pytest.raises(Exception):
            await storage.initialize()

    @pytest.mark.asyncio
    async def test_concurrent_agent_registration(self, sqlite_storage, registered_agent_info_factory):
        """Test concurrent agent registration doesn't cause conflicts."""
        import asyncio
        
        # Create multiple agents
        agents = [
            registered_agent_info_factory(agent_id=f"concurrent-agent-{i}")
            for i in range(5)
        ]
        
        # Register all agents concurrently
        tasks = [sqlite_storage.register_agent(agent) for agent in agents]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(results)
        
        # Verify all agents were registered
        registered_agents = await sqlite_storage.list_registered_agent_info()
        assert len(registered_agents) == 5

    @pytest.mark.asyncio
    async def test_agent_to_registered_agent_info_conversion(self, sqlite_storage, sample_registered_agent_info):
        """Test conversion from database Agent model to RegisteredAgentInfo."""
        await sqlite_storage.register_agent(sample_registered_agent_info)
        
        retrieved = await sqlite_storage.get_registered_agent_info(sample_registered_agent_info.agent_id)
        
        # Verify all fields are correctly converted
        assert isinstance(retrieved, RegisteredAgentInfo)
        assert retrieved.agent_id == sample_registered_agent_info.agent_id
        assert retrieved.agent_type == sample_registered_agent_info.agent_type
        assert isinstance(retrieved.system_info, dict)
        assert isinstance(retrieved.capabilities, dict)
        assert isinstance(retrieved.registered_at, datetime)


class TestPostgreSQLStorageBackend:
    """Tests for PostgreSQL storage backend."""
    
    # Note: These tests would require a test PostgreSQL database
    # For now, we'll test the initialization and basic functionality
    
    def test_initialization_with_settings(self):
        """Test PostgreSQL backend initialization."""
        settings = PostgreSQLSettings(
            database_url="postgresql+asyncpg://test:test@localhost/test_db",
            pool_size=5,
            max_overflow=10,
            echo_sql=False
        )
        
        storage = PostgreSQLStorageBackend(settings)
        assert storage.settings == settings
        assert storage.engine is None  # Not initialized yet
        assert storage.session_factory is None

    @pytest.mark.asyncio
    async def test_health_check_without_initialization(self):
        """Test health check fails when not initialized."""
        settings = PostgreSQLSettings(
            database_url="postgresql+asyncpg://invalid:invalid@localhost/invalid_db"
        )
        
        storage = PostgreSQLStorageBackend(settings)
        
        # Health check should return False if not initialized
        result = await storage.health_check()
        assert result is False

    def test_retry_operation_configuration(self):
        """Test retry operation is configured correctly."""
        settings = PostgreSQLSettings(
            database_url="postgresql+asyncpg://test:test@localhost/test_db",
            max_retries=3,
            retry_delay=0.5
        )
        
        storage = PostgreSQLStorageBackend(settings)
        assert storage.settings.max_retries == 3
        assert storage.settings.retry_delay == 0.5


class TestStorageBackendErrorHandling:
    """Tests for storage backend error handling scenarios."""

    @pytest.mark.asyncio
    async def test_sqlite_connection_error_during_operation(self, sqlite_settings):
        """Test handling of connection errors during operations."""
        storage = SQLiteStorageBackend(sqlite_settings)
        await storage.initialize()
        
        # Close the connection to simulate error
        await storage.close()
        
        # Operations should fail gracefully
        result = await storage.agent_exists("any-agent")
        assert result is False

    @pytest.mark.asyncio
    async def test_sqlite_register_agent_with_invalid_data(self, sqlite_storage):
        """Test registering agent with invalid data."""
        # First register a valid agent
        valid_agent = RegisteredAgentInfo(
            agent_id="test-agent-1",
            agent_type=AgentType.WHITE_BOX,
            agent_url="http://test.com"
        )
        await sqlite_storage.register_agent(valid_agent)
        
        # Now try to register another agent with the same ID (duplicate primary key)
        duplicate_agent = RegisteredAgentInfo(
            agent_id="test-agent-1",  # Duplicate ID should cause primary key violation
            agent_type=AgentType.BLACK_BOX,
            agent_url="http://test2.com"
        )
        
        result = await sqlite_storage.register_agent(duplicate_agent)
        # Should handle gracefully and return False due to primary key violation
        assert result is False

    @pytest.mark.asyncio
    async def test_database_file_permissions(self):
        """Test handling of database file permission errors."""
        # Create a read-only directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            temp_path.chmod(0o444)  # Read-only
            
            db_path = temp_path / "test.db"
            settings = SQLiteSettings(database_path=str(db_path))
            
            storage = SQLiteStorageBackend(settings)
            
            # Should raise exception due to permissions
            with pytest.raises(Exception):
                await storage.initialize()