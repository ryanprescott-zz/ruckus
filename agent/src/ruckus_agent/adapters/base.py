"""Base adapter interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""

    @abstractmethod
    async def load_model(self, model_name: str, **kwargs) -> None:
        """Load a model."""
        pass

    @abstractmethod
    async def unload_model(self) -> None:
        """Unload the current model."""
        pass

    @abstractmethod
    async def generate(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Generate output from prompt."""
        pass

    @abstractmethod
    async def generate_batch(self, prompts: List[str], parameters: Dict[str, Any]) -> List[str]:
        """Generate outputs for multiple prompts."""
        pass

    @abstractmethod
    async def tokenize(self, text: str) -> List[int]:
        """Tokenize text."""
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, bool]:
        """Get adapter capabilities."""
        pass

    @abstractmethod
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        pass