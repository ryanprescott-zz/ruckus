"""System capability detection."""

import logging
import platform
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from .models import (
    GPUInfo, CPUInfo, SystemInfo,
    FrameworkInfo, ModelInfo, HookInfo
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
        """Detect comprehensive GPU information with tensor core capabilities."""
        gpus = []
        
        # Try pynvml for NVIDIA GPUs (most comprehensive)
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                gpu_info = await self._detect_nvidia_gpu_pynvml(i)
                if gpu_info:
                    gpus.append(gpu_info)
                    
            logger.info(f"Detected {len(gpus)} NVIDIA GPUs using pynvml")
            
        except (ImportError, Exception) as e:
            logger.debug(f"pynvml detection failed: {e}, trying nvidia-smi fallback")
            
            # Fallback to nvidia-smi
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
                            gpu_info = {
                                "index": int(parts[0]),
                                "name": parts[1],
                                "uuid": parts[2],
                                "memory_total_mb": int(parts[3]),
                                "memory_available_mb": int(parts[4]),
                            }
                            # Add tensor core detection based on GPU name
                            gpu_info.update(await self._detect_tensor_capabilities_by_name(parts[1]))
                            gpus.append(gpu_info)
                            
            except Exception as e2:
                logger.debug(f"nvidia-smi fallback also failed: {e2}")
        
        # Try PyTorch for any GPU (NVIDIA, AMD, Intel, Apple Silicon, etc.)
        if not gpus:
            try:
                torch_gpus = await self._detect_gpus_pytorch()
                gpus.extend(torch_gpus)
                logger.info(f"Detected {len(torch_gpus)} GPUs using PyTorch")
            except Exception as e:
                logger.debug(f"PyTorch GPU detection failed: {e}")

        return gpus

    async def _detect_nvidia_gpu_pynvml(self, device_index: int) -> Optional[Dict]:
        """Detect NVIDIA GPU capabilities using pynvml."""
        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            
            # Basic GPU information
            name = pynvml.nvmlDeviceGetName(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            
            # Memory information
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Compute capability (for tensor core mapping)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            compute_capability = f"{major}.{minor}"
            
            # Live metrics
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                memory_util = utilization.memory
            except:
                gpu_util = memory_util = None
                
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = None
                
            try:
                power_info = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_watts = power_info / 1000.0  # Convert mW to W
            except:
                power_watts = None
            
            # Map compute capability to tensor core support
            tensor_capabilities = self._map_compute_capability_to_tensor_cores(major, minor)
            supported_precisions = self._get_supported_precisions_from_compute(major, minor)
            
            gpu_info = {
                "index": device_index,
                "name": name,
                "uuid": uuid,
                "memory_total_mb": memory_info.total // (1024 * 1024),
                "memory_available_mb": memory_info.free // (1024 * 1024),
                "memory_used_mb": memory_info.used // (1024 * 1024),
                "compute_capability": compute_capability,
                "tensor_cores": tensor_capabilities,
                "supported_precisions": supported_precisions,
                "current_utilization_percent": gpu_util,
                "memory_utilization_percent": memory_util,
                "temperature_celsius": temperature,
                "power_usage_watts": power_watts,
                "driver_version": pynvml.nvmlSystemGetDriverVersion(),
                "detection_method": "pynvml"
            }
            
            return gpu_info
            
        except Exception as e:
            logger.debug(f"Failed to detect GPU {device_index} with pynvml: {e}")
            return None

    async def _detect_gpus_pytorch(self) -> List[Dict]:
        """Detect GPUs using PyTorch (cross-platform fallback)."""
        gpus = []
        
        try:
            import torch
            
            # CUDA GPUs
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    
                    # Get current memory info
                    torch.cuda.set_device(i)
                    memory_allocated = torch.cuda.memory_allocated(i) // (1024 * 1024)
                    memory_cached = torch.cuda.memory_reserved(i) // (1024 * 1024)
                    memory_total = props.total_memory // (1024 * 1024)
                    
                    # Test supported precisions by actually trying operations
                    supported_precisions = await self._test_precision_support_pytorch(i)
                    
                    gpu_info = {
                        "index": i,
                        "name": props.name,
                        "uuid": f"pytorch-cuda-{i}",
                        "memory_total_mb": memory_total,
                        "memory_available_mb": memory_total - memory_allocated - memory_cached,
                        "memory_used_mb": memory_allocated,
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multiprocessor_count": props.multi_processor_count,
                        "supported_precisions": supported_precisions,
                        "tensor_cores": self._map_compute_capability_to_tensor_cores(props.major, props.minor),
                        "detection_method": "pytorch_cuda"
                    }
                    gpus.append(gpu_info)
            
            # MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                supported_precisions = await self._test_precision_support_pytorch('mps')
                
                gpu_info = {
                    "index": 0,
                    "name": "Apple Silicon GPU",
                    "uuid": "pytorch-mps-0", 
                    "memory_total_mb": -1,  # MPS doesn't report memory limits
                    "memory_available_mb": -1,
                    "supported_precisions": supported_precisions,
                    "tensor_cores": [],  # Apple Silicon uses different acceleration
                    "detection_method": "pytorch_mps"
                }
                gpus.append(gpu_info)
                
        except Exception as e:
            logger.debug(f"PyTorch GPU detection failed: {e}")
            
        return gpus

    async def _detect_tensor_capabilities_by_name(self, gpu_name: str) -> Dict:
        """Detect tensor capabilities based on GPU name (fallback method)."""
        gpu_name_lower = gpu_name.lower()
        
        # Map known GPU names to tensor core capabilities
        if any(arch in gpu_name_lower for arch in ['a100', 'a6000', 'a5000', 'a4000', 'a2000']):
            # Ampere architecture
            return {
                "tensor_cores": ["3rd_gen"],
                "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "int8", "fp64"]
            }
        elif any(arch in gpu_name_lower for arch in ['rtx 40', '4090', '4080', '4070', '4060']):
            # Ada Lovelace architecture  
            return {
                "tensor_cores": ["4th_gen"],
                "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "int8", "fp8"]
            }
        elif any(arch in gpu_name_lower for arch in ['rtx 30', '3090', '3080', '3070', '3060']):
            # Ampere architecture (consumer)
            return {
                "tensor_cores": ["3rd_gen"],
                "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "int8"]
            }
        elif any(arch in gpu_name_lower for arch in ['rtx 20', '2080', '2070', '2060', 'titan rtx']):
            # Turing architecture
            return {
                "tensor_cores": ["2nd_gen"],
                "supported_precisions": ["fp32", "fp16", "int8"]
            }
        elif any(arch in gpu_name_lower for arch in ['v100', 'titan v']):
            # Volta architecture
            return {
                "tensor_cores": ["1st_gen"],
                "supported_precisions": ["fp32", "fp16"]
            }
        else:
            # Unknown/older GPU
            return {
                "tensor_cores": [],
                "supported_precisions": ["fp32"]
            }

    def _map_compute_capability_to_tensor_cores(self, major: int, minor: int) -> List[str]:
        """Map CUDA compute capability to tensor core generations."""
        compute_cap = f"{major}.{minor}"
        
        if major >= 9:  # Hopper (H100, etc.)
            return ["4th_gen", "fp8_support"]
        elif major >= 8:  # Ampere
            if minor >= 6:  # A100
                return ["3rd_gen", "sparsity_support"]
            else:  # RTX 30 series
                return ["3rd_gen"]
        elif major == 7 and minor >= 5:  # Turing
            return ["2nd_gen"]
        elif major == 7 and minor == 0:  # Volta
            return ["1st_gen"]
        else:
            return []  # No tensor cores

    def _get_supported_precisions_from_compute(self, major: int, minor: int) -> List[str]:
        """Get supported precisions based on compute capability."""
        base_precisions = ["fp32"]
        
        if major >= 9:  # Hopper
            return base_precisions + ["tf32", "fp16", "bf16", "int8", "fp8", "fp64"]
        elif major >= 8:  # Ampere  
            return base_precisions + ["tf32", "fp16", "bf16", "int8", "fp64"]
        elif major == 7 and minor >= 5:  # Turing
            return base_precisions + ["fp16", "int8"]
        elif major == 7 and minor == 0:  # Volta
            return base_precisions + ["fp16", "fp64"]
        else:
            return base_precisions

    async def _test_precision_support_pytorch(self, device) -> List[str]:
        """Test which precisions are actually supported by running operations."""
        supported = ["fp32"]  # Always supported
        
        try:
            import torch
            
            # Test different precisions
            test_precisions = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "int8": torch.int8,
            }
            
            for precision_name, dtype in test_precisions.items():
                try:
                    # Try creating a small tensor and doing a simple operation
                    if device == 'mps':
                        test_tensor = torch.randn(4, 4, dtype=dtype, device='mps')
                    else:
                        test_tensor = torch.randn(4, 4, dtype=dtype).cuda(device)
                    
                    # Simple matrix multiply to test
                    result = torch.mm(test_tensor, test_tensor.T)
                    supported.append(precision_name)
                    
                except Exception:
                    pass  # Precision not supported
                    
        except Exception:
            pass
            
        return supported

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
            discovery = ModelDiscovery(settings.model_path)
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
