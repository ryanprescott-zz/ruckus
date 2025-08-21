"""System capability detection."""

import platform
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from .models import (
    GPUInfo, CPUInfo, SystemInfo,
    FrameworkInfo, ModelInfo, HookInfo,
    MetricCapability
)


class AgentDetector:
    """Detect agent capabilities."""

    async def detect_all(self) -> Dict[str, Any]:
        """Detect all capabilities."""
        return {
            "system": await self.detect_system(),
            "cpu": await self.detect_cpu(),
            "gpus": await self.detect_gpus(),
            "frameworks": await self.detect_frameworks(),
            "models": await self.detect_models(),
            "hooks": await self.detect_hooks(),
            "metrics": await self.detect_metrics(),
        }

    async def detect_system(self) -> Dict:
        """Detect system information."""
        import psutil

        return {
            "hostname": platform.node(),
            "os": platform.system(),
            "os_version": platform.version(),
            "kernel": platform.release(),
            "python_version": platform.python_version(),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "disk_total_gb": psutil.disk_usage('/').total / (1024**3),
            "disk_available_gb": psutil.disk_usage('/').free / (1024**3),
        }

    async def detect_cpu(self) -> Dict:
        """Detect CPU information."""
        import psutil

        return {
            "model": platform.processor() or "Unknown",
            "cores_physical": psutil.cpu_count(logical=False) or 0,
            "cores_logical": psutil.cpu_count(logical=True) or 0,
            "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            "architecture": platform.machine(),
        }

    async def detect_gpus(self) -> List[Dict]:
        """Detect GPU information."""
        gpus = []

        # Try nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        gpus.append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "uuid": parts[2],
                            "memory_total_mb": int(parts[3]),
                            "memory_available_mb": int(parts[4]),
                        })
        except Exception:
            pass

        return gpus

    async def detect_frameworks(self) -> List[Dict]:
        """Detect available ML frameworks."""
        frameworks = []

        # Check transformers
        try:
            import transformers
            frameworks.append({
                "name": "transformers",
                "version": transformers.__version__,
                "available": True,
                "capabilities": {
                    "text_generation": True,
                    "tokenization": True,
                }
            })
        except ImportError:
            pass

        # Check PyTorch
        try:
            import torch
            frameworks.append({
                "name": "pytorch",
                "version": torch.__version__,
                "available": True,
                "capabilities": {
                    "cuda": torch.cuda.is_available(),
                    "mps": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
                }
            })
        except ImportError:
            pass

        # Check vLLM
        try:
            import vllm
            frameworks.append({
                "name": "vllm",
                "version": vllm.__version__,
                "available": True,
                "capabilities": {
                    "streaming": True,
                    "continuous_batching": True,
                }
            })
        except ImportError:
            pass

        return frameworks

    async def detect_models(self, search_paths: List[str] = None) -> List[Dict]:
        """Detect available models."""
        if search_paths is None:
            search_paths = ["/models", "./models", "~/.cache/huggingface"]

        models = []
        # TODO: Implement model detection
        return models

    async def detect_hooks(self) -> List[Dict]:
        """Detect available system hooks."""
        hooks = []

        tools = [
            ("nvidia-smi", "gpu_monitor"),
            ("rocm-smi", "gpu_monitor"),
            ("htop", "cpu_monitor"),
            ("iotop", "disk_monitor"),
        ]

        for tool, tool_type in tools:
            try:
                result = subprocess.run(
                    ["which", tool],
                    capture_output=True,
                    timeout=1,
                )
                if result.returncode == 0:
                    hooks.append({
                        "name": tool,
                        "type": tool_type,
                        "executable_path": result.stdout.decode().strip(),
                        "working": True,
                    })
            except Exception:
                pass

        return hooks

    async def detect_metrics(self) -> List[Dict]:
        """Detect available metrics."""
        metrics = []

        # Basic metrics always available
        metrics.extend([
            {
                "name": "latency",
                "type": "performance",
                "available": True,
                "collection_method": "timer",
                "requires": [],
            },
            {
                "name": "throughput",
                "type": "performance",
                "available": True,
                "collection_method": "calculation",
                "requires": [],
            },
        ])

        # GPU metrics if available
        if await self.detect_gpus():
            metrics.extend([
                {
                    "name": "gpu_utilization",
                    "type": "resource",
                    "available": True,
                    "collection_method": "nvidia-smi",
                    "requires": ["nvidia-smi"],
                },
                {
                    "name": "gpu_memory",
                    "type": "resource",
                    "available": True,
                    "collection_method": "nvidia-smi",
                    "requires": ["nvidia-smi"],
                },
            ])

        return metrics