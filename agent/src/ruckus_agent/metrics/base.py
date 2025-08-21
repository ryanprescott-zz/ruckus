"""Base metric collector."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime


class MetricCollector(ABC):
    """Abstract base class for metric collectors."""

    def __init__(self, name: str):
        self.name = name
        self.metrics = {}
        self.start_time = None

    def start(self):
        """Start metric collection."""
        self.start_time = datetime.utcnow()
        self.metrics = {}

    def stop(self):
        """Stop metric collection."""
        if self.start_time:
            duration = (datetime.utcnow() - self.start_time).total_seconds()
            self.metrics["duration_seconds"] = duration

    @abstractmethod
    async def collect(self) -> Dict[str, Any]:
        """Collect metrics."""
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return self.metrics

    def reset(self):
        """Reset metrics."""
        self.metrics = {}
        self.start_time = None