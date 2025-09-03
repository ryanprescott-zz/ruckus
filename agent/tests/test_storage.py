"""Tests for storage backends."""

import pytest
from datetime import datetime, timezone
from ruckus_agent.core.storage import InMemoryStorage


@pytest.mark.asyncio
class TestInMemoryStorage:
    """Test in-memory storage implementation."""

    async def test_system_info_storage(self):
        """Test storing and retrieving system info."""
        storage = InMemoryStorage()
        
        system_info = {
            "system": {"hostname": "test-host", "os": "Linux"},
            "cpu": {"cores": 4, "model": "Intel"},
            "gpus": [],
            "frameworks": [],
            "models": [],
            "hooks": [],
            "metrics": []
        }
        
        await storage.store_system_info(system_info)
        retrieved = await storage.get_system_info()
        
        assert retrieved == system_info
        assert retrieved is not system_info  # Should be a copy

    async def test_last_seen_tracking(self):
        """Test last seen timestamp tracking."""
        storage = InMemoryStorage()
        
        # Initially no last seen
        assert await storage.get_last_seen() is None
        
        # Update last seen
        before = datetime.now(timezone.utc)
        await storage.update_last_seen()
        after = datetime.now(timezone.utc)
        
        last_seen = await storage.get_last_seen()
        assert last_seen is not None
        assert before <= last_seen <= after

    @pytest.mark.asyncio
    async def test_empty_storage_returns_empty_dicts(self):
        """Test that empty storage returns empty dicts."""
        storage = InMemoryStorage()
        
        assert await storage.get_system_info() == {}

    async def test_storage_isolation(self):
        """Test that multiple storage instances are isolated."""
        storage1 = InMemoryStorage()
        storage2 = InMemoryStorage()
        
        await storage1.store_system_info({"test": "data1"})
        await storage2.store_system_info({"test": "data2"})
        
        assert await storage1.get_system_info() == {"test": "data1"}
        assert await storage2.get_system_info() == {"test": "data2"}