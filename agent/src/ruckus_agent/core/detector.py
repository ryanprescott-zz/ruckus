"""System detection utilities."""

import psutil
from typing import Dict, Any


class SystemDetector:
    """Detects system capabilities and hardware."""
    
    def detect_capabilities(self) -> Dict[str, Any]:
        """Detect system capabilities."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total // (1024**3),
            "platform": "unknown",
        }
