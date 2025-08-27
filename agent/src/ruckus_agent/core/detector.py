"""System capability detection."""

import logging
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
from ..utils.model_discovery import ModelDiscovery
from .config import settings

logger = logging.getLogger(__name__)


class AgentDetector:
    """Detect agent capabilities."""

    async def detect_all(self) -> Dict[str, Any]:
        """Detect all capabilities."""
        logger.info("AgentDetector starting comprehensive capability detection")
        try:
            result = {
                "system": await self.detect_system(),
                "cpu": await self.detect_cpu(),
                "gpus": await self.detect_gpus(),
                "frameworks": await self.detect_frameworks(),
                "models": await self.detect_models(),
                "hooks": await self.detect_hooks(),
                "metrics": await self.detect_metrics(),
            }
            logger.info("AgentDetector capability detection completed successfully")
            return result
        except Exception as e:
            logger.error(f"AgentDetector capability detection failed: {e}")
            raise

    async def detect_system(self) -> Dict:
        """Detect system information."""
        logger.debug("Detecting system information")
        try:
            import psutil

            result = {
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
            logger.debug(f"System detected: {result['os']} on {result['hostname']}")
            return result
        except Exception as e:
            logger.error(f"System detection failed: {e}")
            raise

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
        logger.debug("Detecting ML frameworks")
        frameworks = []

        # Check transformers
        try:
            import transformers
            framework_info = {
                "name": "transformers",
                "version": transformers.__version__,
                "available": True,
                "capabilities": {
                    "text_generation": True,
                    "tokenization": True,
                }
            }
            frameworks.append(framework_info)
            logger.debug(f"Transformers detected: {transformers.__version__}")
        except ImportError:
            logger.debug("Transformers not available")

        # Check PyTorch
        try:
            import torch
            framework_info = {
                "name": "pytorch",
                "version": torch.__version__,
                "available": True,
                "capabilities": {
                    "cuda": torch.cuda.is_available(),
                    "mps": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
                }
            }
            frameworks.append(framework_info)
            logger.debug(f"PyTorch detected: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        except ImportError:
            logger.debug("PyTorch not available")

        # Check vLLM
        try:
            import vllm
            framework_info = {
                "name": "vllm",
                "version": vllm.__version__,
                "available": True,
                "capabilities": {
                    "streaming": True,
                    "continuous_batching": True,
                }
            }
            frameworks.append(framework_info)
            logger.debug(f"vLLM detected: {vllm.__version__}")
        except ImportError:
            logger.debug("vLLM not available")

        logger.info(f"Detected {len(frameworks)} ML frameworks")
        return frameworks

    async def detect_models(self, search_paths: List[str] = None) -> List[Dict]:
        """Detect available models."""
        logger.debug("Starting model detection")
        
        try:
            # Use configured model cache directory
            discovery = ModelDiscovery(settings.model_cache_dir)
            models_info = await discovery.discover_all_models()
            
            # Convert ModelInfo objects to dictionaries for storage/transmission
            models = [model.dict() for model in models_info]
            
            logger.info(f"Model detection completed: {len(models)} models found")
            return models
            
        except Exception as e:
            logger.error(f"Model detection failed: {e}")
            return []

    async def detect_hooks(self) -> List[Dict]:
        """Detect available system hooks."""
        logger.debug("Detecting system monitoring hooks")
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
                    hook_info = {
                        "name": tool,
                        "type": tool_type,
                        "executable_path": result.stdout.decode().strip(),
                        "working": True,
                    }
                    hooks.append(hook_info)
                    logger.debug(f"Hook detected: {tool} at {hook_info['executable_path']}")
            except Exception as e:
                logger.debug(f"Hook detection failed for {tool}: {e}")

        logger.info(f"Detected {len(hooks)} system hooks")
        return hooks

    async def detect_metrics(self) -> List[Dict]:
        """Detect available metrics."""
        logger.debug("Detecting available metrics")
        metrics = []

        # Basic metrics always available
        basic_metrics = [
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
        ]
        metrics.extend(basic_metrics)
        logger.debug(f"Added {len(basic_metrics)} basic metrics")

        # GPU metrics if available
        gpus = await self.detect_gpus()
        if gpus:
            gpu_metrics = [
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
            ]
            metrics.extend(gpu_metrics)
            logger.debug(f"Added {len(gpu_metrics)} GPU metrics for {len(gpus)} GPUs")

        logger.info(f"Detected {len(metrics)} available metrics")
        return metrics
