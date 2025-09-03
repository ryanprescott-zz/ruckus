"""Mock utilities for hardware/GPU detection components in tests."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List, Optional
import torch
import subprocess
from datetime import datetime


class MockGpuDevice:
    """Mock GPU device for testing."""
    
    def __init__(self, 
                 name: str = "NVIDIA GeForce RTX 4090",
                 memory_total: int = 25769803776,  # 24GB
                 memory_free: int = 24317591552,   # ~22.7GB
                 memory_used: int = 1452212224,    # ~1.3GB
                 compute_capability: tuple = (8, 9),
                 temperature: int = 45,
                 power_usage: int = 25000,  # 25W in milliwatts
                 utilization_gpu: int = 15,
                 utilization_memory: int = 4,
                 driver_version: str = "535.104.05",
                 cuda_version: str = "12.2"):
        
        self.name = name
        self.memory_total = memory_total
        self.memory_free = memory_free
        self.memory_used = memory_used
        self.compute_capability = compute_capability
        self.temperature = temperature
        self.power_usage = power_usage
        self.utilization_gpu = utilization_gpu
        self.utilization_memory = utilization_memory
        self.driver_version = driver_version
        self.cuda_version = cuda_version
        
        # Derived properties
        self.memory_total_mb = memory_total // (1024 * 1024)
        self.memory_free_mb = memory_free // (1024 * 1024)
        self.memory_used_mb = memory_used // (1024 * 1024)
        
        # Tensor core generation mapping
        self.tensor_core_generation = self._get_tensor_core_generation()
    
    def _get_tensor_core_generation(self) -> int:
        """Map compute capability to tensor core generation."""
        major, minor = self.compute_capability
        
        if major >= 9:  # Hopper and beyond
            return 5
        elif major == 8:  # Ampere/Ada Lovelace
            if minor >= 6:
                return 3 if minor == 6 else 4  # RTX 30 series vs RTX 40 series
            else:
                return 3  # A100, etc.
        elif major == 7:  # Volta/Turing
            return 1 if minor == 0 else 2  # V100 vs RTX 20 series
        else:
            return 0  # Pre-tensor core
    
    def to_pynvml_device(self) -> MagicMock:
        """Convert to pynvml-style device mock."""
        device = MagicMock()
        
        # Device properties
        device.name = self.name.encode('utf-8')
        device.total_memory = self.memory_total
        device.free_memory = self.memory_free
        device.used_memory = self.memory_used
        device.temperature = self.temperature
        device.power_usage = self.power_usage
        
        # Compute capability
        device.major = self.compute_capability[0]
        device.minor = self.compute_capability[1]
        
        # Utilization
        device.gpu_utilization = self.utilization_gpu
        device.memory_utilization = self.utilization_memory
        
        return device
    
    def to_pytorch_device(self) -> MagicMock:
        """Convert to PyTorch-style device properties mock."""
        props = MagicMock()
        props.name = self.name
        props.total_memory = self.memory_total
        props.major = self.compute_capability[0]
        props.minor = self.compute_capability[1]
        return props
    
    def to_nvidia_smi_xml(self, gpu_id: int = 0) -> str:
        """Generate nvidia-smi XML output for this device."""
        return f"""
        <gpu id="0000000{gpu_id:01d}:01:00.0">
            <product_name>{self.name}</product_name>
            <product_brand>GeForce</product_brand>
            <product_architecture>Ada Lovelace</product_architecture>
            <driver_version>{self.driver_version}</driver_version>
            <cuda_version>{self.cuda_version}</cuda_version>
            <uuid>GPU-{gpu_id:08d}-1234-1234-1234-123456789012</uuid>
            <fb_memory_usage>
                <total>{self.memory_total_mb} MiB</total>
                <used>{self.memory_used_mb} MiB</used>
                <free>{self.memory_free_mb} MiB</free>
            </fb_memory_usage>
            <utilization>
                <gpu_util>{self.utilization_gpu} %</gpu_util>
                <memory_util>{self.utilization_memory} %</memory_util>
            </utilization>
            <temperature>
                <gpu_temp>{self.temperature} C</gpu_temp>
            </temperature>
            <power_readings>
                <power_draw>{self.power_usage / 1000:.2f} W</power_draw>
            </power_readings>
        </gpu>
        """


class MockSystemInfo:
    """Mock system information for testing."""
    
    def __init__(self,
                 hostname: str = "test-agent-host",
                 os_name: str = "Linux",
                 os_version: str = "Ubuntu 20.04",
                 cpu_model: str = "Intel(R) Xeon(R) CPU E5-2686 v4",
                 cpu_cores: int = 8,
                 memory_total_gb: float = 32.0,
                 python_version: str = "3.12.11"):
        
        self.hostname = hostname
        self.os_name = os_name
        self.os_version = os_version
        self.cpu_model = cpu_model
        self.cpu_cores = cpu_cores
        self.memory_total_gb = memory_total_gb
        self.python_version = python_version
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "hostname": self.hostname,
            "os": self.os_name,
            "os_version": self.os_version,
            "cpu_model": self.cpu_model,
            "cpu_cores": self.cpu_cores,
            "memory_total_gb": self.memory_total_gb,
            "python_version": self.python_version
        }


class MockFramework:
    """Mock framework information for testing."""
    
    def __init__(self,
                 name: str,
                 version: str,
                 available: bool = True,
                 gpu_support: bool = True):
        
        self.name = name
        self.version = version
        self.available = available
        self.gpu_support = gpu_support
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "version": self.version,
            "available": self.available,
            "gpu_support": self.gpu_support
        }


class HardwareMockManager:
    """Centralized manager for hardware detection mocks."""
    
    def __init__(self):
        self.gpus: List[MockGpuDevice] = []
        self.system_info = MockSystemInfo()
        self.frameworks = {
            "pytorch": MockFramework("pytorch", "2.0.1"),
            "transformers": MockFramework("transformers", "4.21.0"), 
            "vllm": MockFramework("vllm", "0.2.7"),
            "numpy": MockFramework("numpy", "1.24.3", gpu_support=False)
        }
        self.monitoring_tools = ["nvidia-smi", "htop"]
        
        # Default GPU setup
        self.add_gpu()  # Add one RTX 4090 by default
    
    def add_gpu(self, **kwargs) -> MockGpuDevice:
        """Add a mock GPU device."""
        gpu = MockGpuDevice(**kwargs)
        self.gpus.append(gpu)
        return gpu
    
    def clear_gpus(self):
        """Remove all GPU devices."""
        self.gpus.clear()
    
    def add_framework(self, name: str, **kwargs) -> MockFramework:
        """Add a mock framework."""
        framework = MockFramework(name, **kwargs)
        self.frameworks[name] = framework
        return framework
    
    def set_system_info(self, **kwargs):
        """Update system information."""
        self.system_info = MockSystemInfo(**kwargs)
    
    def generate_nvidia_smi_xml(self) -> str:
        """Generate complete nvidia-smi XML output."""
        if not self.gpus:
            return """<?xml version="1.0" ?>
            <!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_log.dtd">
            <nvidia_smi_log>
                <attached_gpus>0</attached_gpus>
            </nvidia_smi_log>"""
        
        gpu_xml = ""
        for i, gpu in enumerate(self.gpus):
            gpu_xml += gpu.to_nvidia_smi_xml(i)
        
        return f"""<?xml version="1.0" ?>
        <!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_log.dtd">
        <nvidia_smi_log>
            <attached_gpus>{len(self.gpus)}</attached_gpus>
            {gpu_xml}
        </nvidia_smi_log>"""


# Global mock manager instance
mock_hardware = HardwareMockManager()


class MockPyNVML:
    """Mock pynvml module for testing."""
    
    def __init__(self, hardware_manager: HardwareMockManager):
        self.hardware = hardware_manager
        self._initialized = False
        self._device_handles = {}
    
    def nvmlInit(self):
        """Mock pynvml initialization."""
        if not self.hardware.gpus:
            raise RuntimeError("No GPUs available")
        self._initialized = True
    
    def nvmlDeviceGetCount(self) -> int:
        """Mock device count."""
        if not self._initialized:
            raise RuntimeError("NVML not initialized")
        return len(self.hardware.gpus)
    
    def nvmlDeviceGetHandleByIndex(self, index: int) -> MagicMock:
        """Mock device handle retrieval."""
        if index >= len(self.hardware.gpus):
            raise RuntimeError(f"Invalid device index: {index}")
        
        if index not in self._device_handles:
            self._device_handles[index] = MagicMock()
            self._device_handles[index]._gpu_index = index
        
        return self._device_handles[index]
    
    def nvmlDeviceGetName(self, handle) -> bytes:
        """Mock device name retrieval."""
        gpu = self.hardware.gpus[handle._gpu_index]
        return gpu.name.encode('utf-8')
    
    def nvmlDeviceGetMemoryInfo(self, handle) -> MagicMock:
        """Mock memory info retrieval."""
        gpu = self.hardware.gpus[handle._gpu_index]
        mem_info = MagicMock()
        mem_info.total = gpu.memory_total
        mem_info.free = gpu.memory_free
        mem_info.used = gpu.memory_used
        return mem_info
    
    def nvmlDeviceGetCudaComputeCapability(self, handle) -> tuple:
        """Mock compute capability retrieval."""
        gpu = self.hardware.gpus[handle._gpu_index]
        return gpu.compute_capability
    
    def nvmlDeviceGetTemperature(self, handle, sensor_type=0) -> int:
        """Mock temperature retrieval."""
        gpu = self.hardware.gpus[handle._gpu_index]
        return gpu.temperature
    
    def nvmlDeviceGetPowerUsage(self, handle) -> int:
        """Mock power usage retrieval."""
        gpu = self.hardware.gpus[handle._gpu_index]
        return gpu.power_usage
    
    def nvmlDeviceGetUtilizationRates(self, handle) -> MagicMock:
        """Mock utilization rates retrieval."""
        gpu = self.hardware.gpus[handle._gpu_index]
        util = MagicMock()
        util.gpu = gpu.utilization_gpu
        util.memory = gpu.utilization_memory
        return util
    
    def nvmlDeviceGetClockInfo(self, handle, clock_type: int) -> int:
        """Mock clock info retrieval."""
        # Return mock clock speeds based on type
        clock_speeds = {
            0: 210,   # NVML_CLOCK_GRAPHICS
            1: 210,   # NVML_CLOCK_SM
            2: 405,   # NVML_CLOCK_MEM
        }
        return clock_speeds.get(clock_type, 0)
    
    def nvmlDeviceGetMaxClockInfo(self, handle, clock_type: int) -> int:
        """Mock max clock info retrieval."""
        max_clock_speeds = {
            0: 2520,  # NVML_CLOCK_GRAPHICS
            1: 2520,  # NVML_CLOCK_SM
            2: 1313,  # NVML_CLOCK_MEM
        }
        return max_clock_speeds.get(clock_type, 0)


# Pytest fixtures for easy use in tests

@pytest.fixture
def mock_hardware_manager():
    """Provide a fresh hardware mock manager for each test."""
    manager = HardwareMockManager()
    return manager


@pytest.fixture
def mock_single_gpu():
    """Mock system with single RTX 4090."""
    manager = HardwareMockManager()
    manager.clear_gpus()
    manager.add_gpu()  # Default RTX 4090
    return manager


@pytest.fixture
def mock_multi_gpu():
    """Mock system with multiple GPUs."""
    manager = HardwareMockManager()
    manager.clear_gpus()
    manager.add_gpu(name="NVIDIA GeForce RTX 4090", memory_total=25769803776)
    manager.add_gpu(name="NVIDIA GeForce RTX 3080", memory_total=10737418240, compute_capability=(8, 6))
    return manager


@pytest.fixture
def mock_no_gpu():
    """Mock system with no GPUs."""
    manager = HardwareMockManager()
    manager.clear_gpus()
    return manager


@pytest.fixture
def mock_cpu_only():
    """Mock CPU-only system."""
    manager = HardwareMockManager()
    manager.clear_gpus()
    manager.set_system_info(
        hostname="cpu-only-agent",
        cpu_cores=16,
        memory_total_gb=64.0
    )
    return manager


# Context managers for patching hardware detection

class mock_nvidia_smi:
    """Context manager for mocking nvidia-smi calls."""
    
    def __init__(self, hardware_manager: HardwareMockManager, available: bool = True):
        self.hardware = hardware_manager
        self.available = available
    
    def __enter__(self):
        if self.available:
            # Mock nvidia-smi being available
            patcher1 = patch('shutil.which', return_value='/usr/bin/nvidia-smi')
            patcher2 = patch('subprocess.run')
            
            self.which_mock = patcher1.start()
            self.subprocess_mock = patcher2.start()
            
            # Mock subprocess result
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = self.hardware.generate_nvidia_smi_xml()
            self.subprocess_mock.return_value = mock_result
        else:
            # Mock nvidia-smi not available
            patcher1 = patch('shutil.which', return_value=None)
            self.which_mock = patcher1.start()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        patch.stopall()


class mock_pynvml:
    """Context manager for mocking pynvml."""
    
    def __init__(self, hardware_manager: HardwareMockManager, available: bool = True):
        self.hardware = hardware_manager
        self.available = available
    
    def __enter__(self):
        if self.available and self.hardware.gpus:
            # Mock pynvml being available
            mock_pynvml_module = MockPyNVML(self.hardware)
            
            self.patchers = [
                patch('pynvml.nvmlInit', mock_pynvml_module.nvmlInit),
                patch('pynvml.nvmlDeviceGetCount', mock_pynvml_module.nvmlDeviceGetCount),
                patch('pynvml.nvmlDeviceGetHandleByIndex', mock_pynvml_module.nvmlDeviceGetHandleByIndex),
                patch('pynvml.nvmlDeviceGetName', mock_pynvml_module.nvmlDeviceGetName),
                patch('pynvml.nvmlDeviceGetMemoryInfo', mock_pynvml_module.nvmlDeviceGetMemoryInfo),
                patch('pynvml.nvmlDeviceGetCudaComputeCapability', mock_pynvml_module.nvmlDeviceGetCudaComputeCapability),
                patch('pynvml.nvmlDeviceGetTemperature', mock_pynvml_module.nvmlDeviceGetTemperature),
                patch('pynvml.nvmlDeviceGetPowerUsage', mock_pynvml_module.nvmlDeviceGetPowerUsage),
                patch('pynvml.nvmlDeviceGetUtilizationRates', mock_pynvml_module.nvmlDeviceGetUtilizationRates),
                patch('pynvml.nvmlDeviceGetClockInfo', mock_pynvml_module.nvmlDeviceGetClockInfo),
                patch('pynvml.nvmlDeviceGetMaxClockInfo', mock_pynvml_module.nvmlDeviceGetMaxClockInfo)
            ]
            
            for patcher in self.patchers:
                patcher.start()
        else:
            # Mock pynvml not available or no GPUs
            patcher = patch('pynvml.nvmlInit', side_effect=Exception("pynvml not available"))
            self.patchers = [patcher]
            patcher.start()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for patcher in self.patchers:
            patcher.stop()


class mock_pytorch_cuda:
    """Context manager for mocking PyTorch CUDA."""
    
    def __init__(self, hardware_manager: HardwareMockManager, available: bool = True):
        self.hardware = hardware_manager
        self.available = available and len(hardware_manager.gpus) > 0
    
    def __enter__(self):
        patcher1 = patch('torch.cuda.is_available', return_value=self.available)
        patcher1.start()
        
        if self.available:
            patcher2 = patch('torch.cuda.device_count', return_value=len(self.hardware.gpus))
            patcher3 = patch('torch.cuda.get_device_properties')
            patcher4 = patch('torch.cuda.get_device_capability')
            patcher5 = patch('torch.cuda.memory_stats')
            patcher6 = patch('torch.version.cuda', '12.1')
            
            patcher2.start()
            
            # Mock device properties
            mock_props = patcher3.start()
            mock_props.side_effect = lambda idx: self.hardware.gpus[idx].to_pytorch_device()
            
            # Mock compute capability
            mock_capability = patcher4.start()
            mock_capability.side_effect = lambda idx=0: self.hardware.gpus[idx].compute_capability
            
            # Mock memory stats
            mock_mem_stats = patcher5.start()
            mock_mem_stats.side_effect = lambda idx=0: {
                'allocated_bytes.all.current': self.hardware.gpus[idx].memory_used,
                'reserved_bytes.all.current': self.hardware.gpus[idx].memory_used + 1073741824  # +1GB reserved
            }
            
            patcher6.start()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        patch.stopall()


# Comprehensive hardware detection mock
class mock_hardware_detection:
    """Context manager for mocking all hardware detection components."""
    
    def __init__(self, hardware_manager: HardwareMockManager):
        self.hardware = hardware_manager
    
    def __enter__(self):
        # Start all hardware detection mocks
        self.nvidia_smi_mock = mock_nvidia_smi(self.hardware, available=True)
        self.pynvml_mock = mock_pynvml(self.hardware, available=True) 
        self.pytorch_mock = mock_pytorch_cuda(self.hardware, available=True)
        
        self.nvidia_smi_mock.__enter__()
        self.pynvml_mock.__enter__()
        self.pytorch_mock.__enter__()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pytorch_mock.__exit__(exc_type, exc_val, exc_tb)
        self.pynvml_mock.__exit__(exc_type, exc_val, exc_tb)
        self.nvidia_smi_mock.__exit__(exc_type, exc_val, exc_tb)


# Example usage functions for common test scenarios

def create_test_agent_with_mocked_hardware(hardware_manager: HardwareMockManager) -> 'Agent':
    """Create a test agent with mocked hardware detection."""
    from ruckus_agent.core.agent import Agent
    from ruckus_agent.core.config import Settings
    from ruckus_agent.core.storage import InMemoryStorage
    from ruckus_common.models import AgentType
    
    settings = Settings(
        agent_type=AgentType.WHITE_BOX,
        max_concurrent_jobs=2
    )
    
    agent = Agent(settings, InMemoryStorage())
    
    # Pre-populate with mock hardware info
    system_info = {
        "system": hardware_manager.system_info.to_dict(),
        "cpu": {
            "cores": hardware_manager.system_info.cpu_cores,
            "model": hardware_manager.system_info.cpu_model
        },
        "gpus": [
            {
                "name": gpu.name,
                "memory_total_mb": gpu.memory_total_mb,
                "memory_free_mb": gpu.memory_free_mb,
                "compute_capability": list(gpu.compute_capability),
                "tensor_core_generation": gpu.tensor_core_generation
            } for gpu in hardware_manager.gpus
        ],
        "frameworks": [fw.to_dict() for fw in hardware_manager.frameworks.values()],
        "models": ["test-model-7b", "test-model-13b"],
        "hooks": hardware_manager.monitoring_tools,
        "metrics": ["latency", "throughput", "memory", "gpu_utilization"]
    }
    
    # Store in agent's storage (synchronous - for testing)
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(agent.storage.store_system_info(system_info))
    finally:
        loop.close()
    
    return agent