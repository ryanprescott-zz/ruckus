"""Base task interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class TaskResult(BaseModel):
    """Result from task execution."""
    output: Any
    metrics: Dict[str, Any]
    artifacts: List[str] = []
    metadata: Dict[str, Any] = {}


class BaseTask(ABC):
    """Abstract base class for benchmark tasks."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    async def prepare(self) -> None:
        """Prepare task for execution."""
        pass

    @abstractmethod
    async def run(self, model_adapter, parameters: Dict[str, Any]) -> TaskResult:
        """Run the task."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up after task."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get task configuration."""
        return self.config