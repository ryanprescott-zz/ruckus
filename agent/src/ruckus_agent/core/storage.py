"""Storage backends for agent data."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime


class AgentStorage(ABC):
    """Abstract base class for agent data storage."""
    
    @abstractmethod
    async def store_system_info(self, system_info: Dict[str, Any]) -> None:
        """Store system information."""
        pass
    
    @abstractmethod
    async def get_system_info(self) -> Dict[str, Any]:
        """Retrieve system information."""
        pass
    
    @abstractmethod
    async def store_capabilities(self, capabilities: Dict[str, Any]) -> None:
        """Store agent capabilities."""
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> Dict[str, Any]:
        """Retrieve agent capabilities."""
        pass
    
    @abstractmethod
    async def update_last_seen(self) -> None:
        """Update last seen timestamp."""
        pass
    
    @abstractmethod
    async def get_last_seen(self) -> Optional[datetime]:
        """Get last seen timestamp."""
        pass


class InMemoryStorage(AgentStorage):
    """In-memory storage implementation."""
    
    def __init__(self):
        self._system_info: Dict[str, Any] = {}
        self._capabilities: Dict[str, Any] = {}
        self._last_seen: Optional[datetime] = None
    
    async def store_system_info(self, system_info: Dict[str, Any]) -> None:
        """Store system information in memory."""
        self._system_info = system_info.copy()
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Retrieve system information from memory."""
        return self._system_info.copy()
    
    async def store_capabilities(self, capabilities: Dict[str, Any]) -> None:
        """Store agent capabilities in memory."""
        self._capabilities = capabilities.copy()
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Retrieve agent capabilities from memory."""
        return self._capabilities.copy()
    
    async def update_last_seen(self) -> None:
        """Update last seen timestamp."""
        self._last_seen = datetime.utcnow()
    
    async def get_last_seen(self) -> Optional[datetime]:
        """Get last seen timestamp."""
        return self._last_seen