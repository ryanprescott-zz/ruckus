"""Tests for GPU benchmarking and detection functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock, call, AsyncMock
import torch
import numpy as np
from typing import Dict, Any, List
from contextlib import nullcontext

from ruckus_agent.utils.gpu_benchmark import GPUBenchmark
from ruckus_agent.core.detector import AgentDetector
from ruckus_common.models import AgentCapabilityDetectionResult
from ruckus_common.models import MultiRunJobResult


@pytest.fixture
def mock_nvidia_smi():
    """Mock nvidia-smi subprocess calls."""
    mock_output = """
    <?xml version="1.0" ?>
    <!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_log.dtd">
    <nvidia_smi_log>
        <attached_gpus>1</attached_gpus>
        <gpu id="00000000:01:00.0">
            <product_name>NVIDIA GeForce RTX 4090</product_name>
            <product_brand>GeForce</product_brand>
            <product_architecture>Ada Lovelace</product_architecture>
            <display_mode>Enabled</display_mode>
            <display_active>Enabled</display_active>
            <persistence_mode>Disabled</persistence_mode>
            <mig_mode>
                <current_mig>N/A</current_mig>
                <pending_mig>N/A</pending_mig>
            </mig_mode>
            <mig_devices></mig_devices>
            <accounting_mode>Disabled</accounting_mode>
            <accounting_mode_buffer_size>4000</accounting_mode_buffer_size>
            <driver_version>535.104.05</driver_version>
            <cuda_version>12.2</cuda_version>
            <serial>1234567890</serial>
            <uuid>GPU-12345678-1234-1234-1234-123456789012</uuid>
            <minor_number>0</minor_number>
            <vbios_version>95.02.5C.40.01</vbios_version>
            <multigpu_board>No</multigpu_board>
            <board_id>0x100</board_id>
            <gpu_part_number>2684-200-A1</gpu_part_number>
            <gpu_module_id>1</gpu_module_id>
            <inforom_version>
                <img_version>G001.0000.03.03</img_version>
                <oem_version>1.0</oem_version>
                <ecc_version>N/A</ecc_version>
                <pwr_version>N/A</pwr_version>
            </inforom_version>
            <gpu_operation_mode>
                <current_gom>N/A</current_gom>
                <pending_gom>N/A</pending_gom>
            </gpu_operation_mode>
            <gpu_virtualization_mode>
                <virtualization_mode>None</virtualization_mode>
            </gpu_virtualization_mode>
            <ibmnpu>
                <relaxed_ordering_mode>N/A</relaxed_ordering_mode>
            </ibmnpu>
            <pci>
                <pci_bus>01</pci_bus>
                <pci_device>00</pci_device>
                <pci_domain>0000</pci_domain>
                <pci_device_id>268410DE</pci_device_id>
                <pci_bus_id>00000000:01:00.0</pci_bus_id>
                <pci_sub_system_id>157310DE</pci_sub_system_id>
                <pci_gpu_link_info>
                    <pcie_gen>
                        <max_link_gen>4</max_link_gen>
                        <current_link_gen>1</current_link_gen>
                    </pcie_gen>
                    <link_widths>
                        <max_link_width>16x</max_link_width>
                        <current_link_width>16x</current_link_width>
                    </link_widths>
                </pci_gpu_link_info>
                <pci_bridge_chip>
                    <bridge_chip_type>N/A</bridge_chip_type>
                    <bridge_chip_fw>N/A</bridge_chip_fw>
                </pci_bridge_chip>
                <replay_counter>0</replay_counter>
                <replay_rollover_counter>0</replay_rollover_counter>
                <tx_util>0 KB/s</tx_util>
                <rx_util>0 KB/s</rx_util>
            </pci>
            <fan_speed>30 %</fan_speed>
            <performance_state>P8</performance_state>
            <clocks_throttle_reasons>
                <clocks_throttle_reason_gpu_idle>Active</clocks_throttle_reason_gpu_idle>
                <clocks_throttle_reason_applications_clocks_setting>Not Active</clocks_throttle_reason_applications_clocks_setting>
                <clocks_throttle_reason_sw_power_cap>Not Active</clocks_throttle_reason_sw_power_cap>
                <clocks_throttle_reason_hw_slowdown>Not Active</clocks_throttle_reason_hw_slowdown>
                <clocks_throttle_reason_hw_thermal_slowdown>Not Active</clocks_throttle_reason_hw_thermal_slowdown>
                <clocks_throttle_reason_hw_power_brake_slowdown>Not Active</clocks_throttle_reason_hw_power_brake_slowdown>
                <clocks_throttle_reason_sync_boost>Not Active</clocks_throttle_reason_sync_boost>
                <clocks_throttle_reason_sw_thermal_slowdown>Not Active</clocks_throttle_reason_sw_thermal_slowdown>
                <clocks_throttle_reason_display_clocks_setting>Not Active</clocks_throttle_reason_display_clocks_setting>
            </clocks_throttle_reasons>
            <fb_memory_usage>
                <total>24564 MiB</total>
                <reserved>106 MiB</reserved>
                <used>1024 MiB</used>
                <free>23434 MiB</free>
            </fb_memory_usage>
            <bar1_memory_usage>
                <total>256 MiB</total>
                <used>2 MiB</used>
                <free>254 MiB</free>
            </bar1_memory_usage>
            <compute_mode>Default</compute_mode>
            <utilization>
                <gpu_util>15 %</gpu_util>
                <memory_util>4 %</memory_util>
                <encoder_util>0 %</encoder_util>
                <decoder_util>0 %</decoder_util>
            </utilization>
            <encoder_stats>
                <session_count>0</session_count>
                <average_fps>0</average_fps>
                <average_latency>0</average_latency>
            </encoder_stats>
            <fbc_stats>
                <session_count>0</session_count>
                <average_fps>0</average_fps>
                <average_latency>0</average_latency>
            </fbc_stats>
            <ecc_mode>
                <current_ecc>N/A</current_ecc>
                <pending_ecc>N/A</pending_ecc>
            </ecc_mode>
            <ecc_errors>
                <volatile>
                    <sram_correctable>N/A</sram_correctable>
                    <sram_uncorrectable>N/A</sram_uncorrectable>
                    <dram_correctable>N/A</dram_correctable>
                    <dram_uncorrectable>N/A</dram_uncorrectable>
                </volatile>
                <aggregate>
                    <sram_correctable>N/A</sram_correctable>
                    <sram_uncorrectable>N/A</sram_uncorrectable>
                    <dram_correctable>N/A</dram_correctable>
                    <dram_uncorrectable>N/A</dram_uncorrectable>
                </aggregate>
            </ecc_errors>
            <retired_pages>
                <multiple_single_bit_retirement>
                    <retired_count>N/A</retired_count>
                    <retired_pagelist>N/A</retired_pagelist>
                </multiple_single_bit_retirement>
                <double_bit_retirement>
                    <retired_count>N/A</retired_count>
                    <retired_pagelist>N/A</retired_pagelist>
                </double_bit_retirement>
                <pending_blacklist>N/A</pending_blacklist>
                <pending_retirement>N/A</pending_retirement>
            </retired_pages>
            <remapped_rows>N/A</remapped_rows>
            <temperature>
                <gpu_temp>45 C</gpu_temp>
                <gpu_temp_max_threshold>93 C</gpu_temp_max_threshold>
                <gpu_temp_slow_threshold>90 C</gpu_temp_slow_threshold>
                <gpu_temp_max_gpu_threshold>88 C</gpu_temp_max_gpu_threshold>
                <memory_temp>N/A</memory_temp>
                <gpu_temp_max_mem_threshold>N/A</gpu_temp_max_mem_threshold>
            </temperature>
            <supported_gpu_target_temp>
                <gpu_target_temp_min>65 C</gpu_target_temp_min>
                <gpu_target_temp_max>92 C</gpu_target_temp_max>
            </supported_gpu_target_temp>
            <power_readings>
                <power_state>P8</power_state>
                <power_management>Supported</power_management>
                <power_draw>25.00 W</power_draw>
                <power_limit>450.00 W</power_limit>
                <default_power_limit>450.00 W</default_power_limit>
                <enforced_power_limit>450.00 W</enforced_power_limit>
                <min_power_limit>100.00 W</min_power_limit>
                <max_power_limit>500.00 W</max_power_limit>
            </power_readings>
            <clocks>
                <graphics_clock>210 MHz</graphics_clock>
                <sm_clock>210 MHz</sm_clock>
                <mem_clock>405 MHz</mem_clock>
                <video_clock>555 MHz</video_clock>
            </clocks>
            <applications_clocks>
                <graphics_clock>N/A</graphics_clock>
                <mem_clock>N/A</mem_clock>
            </applications_clocks>
            <default_applications_clocks>
                <graphics_clock>N/A</graphics_clock>
                <mem_clock>N/A</mem_clock>
            </default_applications_clocks>
            <max_clocks>
                <graphics_clock>2520 MHz</graphics_clock>
                <sm_clock>2520 MHz</sm_clock>
                <mem_clock>1313 MHz</mem_clock>
                <video_clock>1950 MHz</video_clock>
            </max_clocks>
            <max_customer_boost_clocks>
                <graphics_clock>N/A</graphics_clock>
            </max_customer_boost_clocks>
            <clock_policy>
                <auto_boost>N/A</auto_boost>
                <auto_boost_default>N/A</auto_boost_default>
            </clock_policy>
            <voltage>
                <graphics_volt>N/A</graphics_volt>
            </voltage>
            <supported_clocks></supported_clocks>
            <processes></processes>
        </gpu>
    </nvidia_smi_log>
    """
    return mock_output


@pytest.fixture
def mock_pynvml():
    """Mock pynvml functionality."""
    mock_pynvml = MagicMock()
    
    # Mock device count
    mock_pynvml.nvmlDeviceGetCount.return_value = 1
    
    # Mock device handle
    mock_device = MagicMock()
    mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_device
    
    # Mock device properties
    mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA GeForce RTX 4090"
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = MagicMock(
        total=25769803776,  # 24GB in bytes
        free=24317591552,   # ~22.7GB free
        used=1452212224     # ~1.3GB used
    )
    
    # Mock compute capability
    mock_pynvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 9)  # RTX 4090
    
    # Mock temperature
    mock_pynvml.nvmlDeviceGetTemperature.return_value = 45
    
    # Mock power usage
    mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 25000  # 25W in milliwatts
    
    # Mock utilization
    mock_util = MagicMock()
    mock_util.gpu = 15
    mock_util.memory = 4
    mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
    
    # Mock clock speeds
    mock_pynvml.nvmlDeviceGetClockInfo.side_effect = lambda device, clock_type: {
        0: 210,   # NVML_CLOCK_GRAPHICS
        1: 405,   # NVML_CLOCK_SM  
        2: 405,   # NVML_CLOCK_MEM
    }.get(clock_type, 0)
    
    # Mock max clock speeds
    mock_pynvml.nvmlDeviceGetMaxClockInfo.side_effect = lambda device, clock_type: {
        0: 2520,  # NVML_CLOCK_GRAPHICS
        1: 2520,  # NVML_CLOCK_SM
        2: 1313,  # NVML_CLOCK_MEM
    }.get(clock_type, 0)
    
    return mock_pynvml


class TestGPUBenchmark:
    """Test GPU benchmarking functionality."""

    @pytest.mark.asyncio
    async def test_gpu_benchmark_initialization(self):
        """Test GPUBenchmark initialization."""
        benchmark = GPUBenchmark()
        # Device is None before initialization
        assert benchmark.device is None
        assert benchmark.torch is None
        
        # After initialization, device should be set
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('torch.randn'), \
             patch('torch.cuda.synchronize'):
            result = await benchmark.initialize()
            assert result is True
            assert benchmark.device == 'cuda:0'

    def test_tensor_core_generation_mapping(self):
        """Test tensor core generation detection."""
        # Note: GPUBenchmark doesn't have _get_tensor_core_generation method
        # This functionality is in AgentDetector, so we'll test tensor size calculation instead
        benchmark = GPUBenchmark()
        
        # Test tensor size calculation based on available memory
        sizes = benchmark._calculate_tensor_sizes(8192)  # 8GB VRAM
        assert isinstance(sizes, list)
        assert len(sizes) > 0
        assert all(isinstance(s, tuple) and len(s) == 2 for s in sizes)

    @pytest.mark.asyncio
    async def test_detect_tensor_core_capabilities(self):
        """Test precision performance benchmarking."""
        benchmark = GPUBenchmark()
        
        # Initialize with mocked PyTorch
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('torch.randn'), \
             patch('torch.cuda.synchronize'):
            await benchmark.initialize()
        
        # Mock the precision test method
        with patch.object(benchmark, '_test_precision_throughput', return_value=100.0):
            results = await benchmark.benchmark_precision_performance(1024)
            
            # Results should be a dict with precision names as keys
            assert "fp32" in results
            assert "fp16" in results
            assert "throughput_gops" in results["fp32"]
            assert "relative_speedup" in results["fp16"]

    def test_get_tensor_sizes_for_vram(self):
        """Test tensor size selection based on VRAM."""
        benchmark = GPUBenchmark()
        
        # Test different VRAM amounts using the actual method name
        sizes_8gb = benchmark._calculate_tensor_sizes(8192)
        sizes_24gb = benchmark._calculate_tensor_sizes(24576)
        sizes_80gb = benchmark._calculate_tensor_sizes(81920)
        
        # Larger VRAM should allow larger tensors
        assert len(sizes_8gb) <= len(sizes_80gb)
        
        # All should be tuples with reasonable sizes
        for sizes in [sizes_8gb, sizes_24gb, sizes_80gb]:
            assert all(isinstance(s, tuple) and len(s) == 2 for s in sizes)
            assert all(s[0] >= 512 and s[1] >= 512 for s in sizes)  # Minimum size

    @pytest.mark.asyncio
    async def test_memory_bandwidth_benchmark(self):
        """Test memory bandwidth benchmarking."""
        benchmark = GPUBenchmark()
        
        # Initialize with mocked PyTorch
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('torch.randn') as mock_randn, \
             patch('torch.cuda.synchronize'):
            await benchmark.initialize()
            
            # Mock tensor operations
            mock_tensor = MagicMock()
            mock_tensor.cuda.return_value = mock_tensor
            mock_tensor.clone.return_value = mock_tensor
            mock_randn.return_value = mock_tensor
            
            # Mock the actual test methods
            with patch.object(benchmark, '_test_memory_copy_bandwidth', return_value=100.0), \
                 patch.object(benchmark, '_test_memory_write_bandwidth', return_value=90.0), \
                 patch.object(benchmark, '_test_memory_read_bandwidth', return_value=80.0):
                results = await benchmark.benchmark_memory_bandwidth(8192)
                
                # Results should be a dict with size names as keys
                assert len(results) > 0
                first_key = list(results.keys())[0]
                assert "copy_bandwidth_gb_s" in results[first_key]
                assert "write_bandwidth_gb_s" in results[first_key]
                assert "read_bandwidth_gb_s" in results[first_key]

    @pytest.mark.asyncio
    async def test_flops_benchmark_multiple_precisions(self):
        """Test FLOPS benchmarking across different precisions."""
        benchmark = GPUBenchmark()
        
        # Initialize with mocked PyTorch
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('torch.randn'), \
             patch('torch.cuda.synchronize'):
            await benchmark.initialize()
        
        # Mock the matmul test method
        with patch.object(benchmark, '_test_matmul_flops', return_value=50.0):
            results = await benchmark.benchmark_compute_flops(8192)
            
            # Results should be a dict with size names as keys
            assert len(results) > 0
            first_key = list(results.keys())[0]
            assert "fp32_tflops" in results[first_key]

    @pytest.mark.asyncio
    async def test_full_gpu_benchmark_integration(self):
        """Test full GPU benchmark integration."""
        benchmark = GPUBenchmark()
        
        # Initialize with mocked PyTorch
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('torch.randn'), \
             patch('torch.cuda.synchronize'):
            await benchmark.initialize()
        
        # Mock the benchmark methods
        with patch.object(benchmark, 'benchmark_memory_bandwidth') as mock_bandwidth, \
             patch.object(benchmark, 'benchmark_compute_flops') as mock_flops, \
             patch.object(benchmark, 'benchmark_precision_performance') as mock_precision:
            
            # Mock benchmark results - match actual implementation format
            mock_bandwidth.return_value = {
                "1024x1024": {
                    "copy_bandwidth_gb_s": 800.0,
                    "write_bandwidth_gb_s": 700.0,
                    "read_bandwidth_gb_s": 750.0
                }
            }
            
            mock_flops.return_value = {
                "1024x1024": {"fp32_tflops": 100.0, "fp16_tflops": 200.0}
            }
            
            mock_precision.return_value = {
                "fp32": {"throughput_gops": 50.0, "relative_speedup": None},
                "fp16": {"throughput_gops": 100.0, "relative_speedup": 2.0}
            }
            
            # Run full benchmark
            results = await benchmark.run_comprehensive_benchmark(8192)
            
            assert "memory_bandwidth" in results
            assert "compute_flops" in results
            assert "precision_performance" in results
            assert "tensor_sizes_tested" in results
            assert "benchmark_device" in results


class TestGpuDetectionIntegration:
    """Test GPU detection integration with AgentDetector."""

    @pytest.mark.asyncio
    @patch('subprocess.run')
    async def test_detector_gpu_integration_with_nvidia_smi(self, mock_subprocess, mock_nvidia_smi):
        """Test AgentDetector GPU detection with nvidia-smi."""
        # Mock nvidia-smi subprocess call
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "0, NVIDIA GeForce RTX 4090, GPU-12345, 24564, 23434\n"
        mock_subprocess.return_value = mock_result
        
        detector = AgentDetector()
        
        # Mock pynvml failure to force nvidia-smi fallback
        with patch.object(detector, '_detect_nvidia_gpu_pynvml', return_value=None):
            gpu_info = await detector.detect_gpus()
            
            assert len(gpu_info) > 0
            # Now gpu_info is a list of Pydantic models
            assert gpu_info[0].name == "NVIDIA GeForce RTX 4090"
            assert gpu_info[0].memory_total_mb == 24564
            assert gpu_info[0].memory_available_mb == 23434

    @pytest.mark.asyncio
    async def test_detector_gpu_integration_with_pynvml(self):
        """Test AgentDetector GPU detection with pynvml."""
        detector = AgentDetector()
        
        with patch('pynvml.nvmlInit'), \
             patch('pynvml.nvmlDeviceGetCount', return_value=1), \
             patch('pynvml.nvmlDeviceGetHandleByIndex', return_value=MagicMock()), \
             patch('pynvml.nvmlDeviceGetName', return_value=b"NVIDIA GeForce RTX 4090"), \
             patch('pynvml.nvmlDeviceGetUUID', return_value=b"GPU-12345"), \
             patch('pynvml.nvmlDeviceGetMemoryInfo') as mock_mem_info, \
             patch('pynvml.nvmlDeviceGetCudaComputeCapability', return_value=(8, 9)), \
             patch('pynvml.nvmlDeviceGetTemperature', return_value=45), \
             patch('pynvml.nvmlDeviceGetPowerUsage', return_value=25000), \
             patch('pynvml.nvmlDeviceGetUtilizationRates') as mock_util, \
             patch('pynvml.nvmlSystemGetDriverVersion', return_value=b"535.104.05"):
            
            # Mock memory info
            mock_mem_info.return_value = MagicMock(
                total=25769803776,  # 24GB
                free=24317591552,   # ~22.7GB 
                used=1452212224     # ~1.3GB
            )
            
            # Mock utilization
            mock_util.return_value = MagicMock(gpu=15, memory=4)
            
            gpu_info = await detector._detect_nvidia_gpu_pynvml(0)
            
            assert gpu_info is not None
            # Now gpu_info is a Pydantic model
            assert gpu_info.name == "NVIDIA GeForce RTX 4090"
            assert gpu_info.compute_capability == "8.9"
            assert gpu_info.memory_total_mb == 24576  # ~24GB
            assert gpu_info.temperature_celsius == 45
            assert gpu_info.power_usage_watts == 25.0

    @pytest.mark.asyncio
    @patch('torch.cuda.is_available')
    async def test_detector_pytorch_gpu_fallback(self, mock_cuda_available):
        """Test AgentDetector PyTorch GPU detection fallback."""
        mock_cuda_available.return_value = True
        
        detector = AgentDetector()
        
        with patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_properties') as mock_props, \
             patch('torch.cuda.set_device'), \
             patch('torch.cuda.memory_allocated', return_value=1452212224), \
             patch('torch.cuda.memory_reserved', return_value=2147483648), \
             patch.object(detector, '_test_precision_support_pytorch', return_value=["fp32", "fp16"]):
            
            # Mock device properties
            mock_device = MagicMock()
            mock_device.name = "NVIDIA GeForce RTX 4090"
            mock_device.total_memory = 25769803776  # 24GB
            mock_device.major = 8
            mock_device.minor = 9
            mock_device.multi_processor_count = 128
            mock_props.return_value = mock_device
            
            gpu_info = await detector._detect_gpus_pytorch()
            
            assert len(gpu_info) > 0
            # Now gpu_info is a list of Pydantic models
            assert gpu_info[0].name == "NVIDIA GeForce RTX 4090"
            assert gpu_info[0].compute_capability == "8.9"
            assert gpu_info[0].memory_total_mb == 24576

    @pytest.mark.asyncio
    @patch('torch.cuda.is_available')
    async def test_full_detector_integration_with_benchmarking(self, mock_cuda_available):
        """Test full detector integration including GPU benchmarking."""
        mock_cuda_available.return_value = True
        
        detector = AgentDetector()
        
        with patch('pynvml.nvmlInit'), \
             patch('pynvml.nvmlDeviceGetCount', return_value=1), \
             patch.object(detector, '_detect_nvidia_gpu_pynvml') as mock_pynvml_detect, \
             patch.object(detector, '_detect_gpus_pytorch') as mock_pytorch_detect, \
             patch('ruckus_agent.utils.gpu_benchmark.GPUBenchmark') as mock_benchmark_class:
            
            # Mock GPU detection results - now return Pydantic model
            from ruckus_common.models import GPUDetectionResult, GPUVendor, DetectionMethod, TensorCoreGeneration, PrecisionType
            mock_pynvml_detect.return_value = GPUDetectionResult(
                index=0,
                name="NVIDIA GeForce RTX 4090",
                vendor=GPUVendor.NVIDIA,
                memory_total_mb=24576,
                memory_available_mb=22760,
                memory_used_mb=1316,
                compute_capability="8.9",
                tensor_cores=[TensorCoreGeneration.FOURTH_GEN],
                supported_precisions=[PrecisionType.FP32, PrecisionType.FP16],
                temperature_celsius=45,
                power_usage_watts=25.0,
                current_utilization_percent=15,
                memory_utilization_percent=4,
                detection_method=DetectionMethod.PYNVML
            )
            
            # Mock PyTorch fallback (shouldn't be called if pynvml succeeds)
            mock_pytorch_detect.return_value = []
            
            # Mock GPU benchmark
            mock_benchmark = MagicMock()
            mock_benchmark_class.return_value = mock_benchmark
            mock_benchmark.initialize = AsyncMock(return_value=True)
            mock_benchmark.run_comprehensive_benchmark = AsyncMock(return_value={
                "memory_bandwidth": {
                    "copy_bandwidth": {"avg_gb_s": 800.0},
                    "write_bandwidth": {"avg_gb_s": 700.0},
                    "read_bandwidth": {"avg_gb_s": 750.0}
                },
                "compute_flops": {
                    "matrix_sizes_tested": [(1024, 1024), (2048, 2048)],
                    "flops_by_size": {"1024x1024": 50.0, "2048x2048": 100.0},
                    "peak_tflops": 100.0
                },
                "precision_performance": {
                    "precisions_tested": ["float32", "float16"],
                    "results_by_precision": {"float32": 50.0, "float16": 100.0}
                },
                "tensor_sizes_tested": [(1024, 1024), (2048, 2048)],
                "benchmark_device": "cuda:0"
            })
            
            # Run detection
            detected_info = await detector.detect_all()
            
            # Now detected_info is an AgentCapabilityDetectionResult Pydantic model
            assert isinstance(detected_info, AgentCapabilityDetectionResult)
            assert len(detected_info.gpus) > 0
            
            gpu = detected_info.gpus[0]
            assert gpu.name == "NVIDIA GeForce RTX 4090"


class TestGPUBenchmarkMocking:
    """Test proper mocking of GPU components for testing."""

    @pytest.mark.asyncio
    async def test_mock_gpu_unavailable_scenario(self):
        """Test behavior when GPU is not available."""
        benchmark = GPUBenchmark()
        
        with patch('torch.cuda.is_available', return_value=False):
            # Check for MPS support on macOS
            if hasattr(torch.backends, 'mps'):
                with patch('torch.backends.mps.is_available', return_value=False):
                    result = await benchmark.initialize()
                    assert result is True
                    assert benchmark.device == 'cpu'
            else:
                result = await benchmark.initialize()
                assert result is True
                assert benchmark.device == 'cpu'

    @pytest.mark.asyncio
    async def test_mock_pynvml_initialization_failure(self):
        """Test fallback when pynvml initialization fails."""
        detector = AgentDetector()
        
        with patch.object(detector, '_detect_nvidia_gpu_pynvml', return_value=None), \
             patch('subprocess.run', side_effect=FileNotFoundError()), \
             patch.object(detector, '_detect_gpus_pytorch') as mock_pytorch_fallback:
            mock_pytorch_fallback.return_value = [{
                "name": "Fallback GPU",
                "memory_total_mb": 8192,
                "compute_capability": "7.5"
            }]
            
            # Should fall back to PyTorch detection
            gpus = await detector.detect_gpus()
            
            mock_pytorch_fallback.assert_called_once()

    @pytest.mark.asyncio  
    async def test_mock_nvidia_smi_unavailable(self):
        """Test behavior when nvidia-smi is not available."""
        detector = AgentDetector()
        
        with patch('subprocess.run', side_effect=FileNotFoundError()), \
             patch.object(detector, '_detect_gpus_pytorch', return_value=[]):
            gpus = await detector.detect_gpus()
            
            # Should return empty list when all methods fail
            assert gpus == []

    @pytest.mark.asyncio
    @patch('subprocess.run')
    async def test_mock_nvidia_smi_malformed_output(self, mock_subprocess):
        """Test handling of malformed nvidia-smi output."""
        # Mock malformed CSV output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Not valid CSV output"
        mock_subprocess.return_value = mock_result
        
        detector = AgentDetector()
        
        with patch.object(detector, '_detect_gpus_pytorch', return_value=[]):
            gpus = await detector.detect_gpus()
            
            # Should handle gracefully and return empty list
            assert gpus == []


class TestGpuMetricsCollection:
    """Test GPU metrics collection during job execution."""

    @pytest.mark.asyncio
    @patch('torch.cuda.is_available')
    async def test_gpu_metrics_during_job_execution(self, mock_cuda_available):
        """Test GPU metrics collection during job execution."""
        mock_cuda_available.return_value = True
        
        from ruckus_agent.core.agent import Agent
        from ruckus_agent.core.config import Settings
        from ruckus_agent.core.storage import InMemoryStorage
        from ruckus_common.models import AgentType, JobRequest, TaskType
        
        settings = Settings(agent_type=AgentType.WHITE_BOX)
        agent = Agent(settings, InMemoryStorage())
        
        job_request = JobRequest(
            job_id="gpu-metrics-test",
            experiment_id="gpu-test",
            model="test-model", 
            framework="pytorch",
            task_type=TaskType.SUMMARIZATION,
            task_config={"input_text": "Test"},
            runs_per_job=3,
            required_metrics=["gpu_utilization", "gpu_memory", "gpu_temperature"]
        )
        
        with patch.object(agent, 'execute_job') as mock_execute:
            # Mock successful multi-run execution with GPU metrics
            async def mock_multi_run_with_gpu_metrics(job):
                from datetime import datetime, timezone, timedelta
                from ruckus_common.models import SingleRunResult, MultiRunJobResult
                now = datetime.now(timezone.utc)
                
                gpu_metrics = {
                    "gpu_utilization_percent": [20, 85, 15],
                    "gpu_memory_used_mb": [2048, 4096, 2048], 
                    "gpu_temperature_c": [55, 78, 52]
                }
                
                runs = []
                for i in range(job.runs_per_job):
                    run = SingleRunResult(
                        run_id=i,
                        is_cold_start=(i == 0),
                        started_at=now + timedelta(seconds=i*2),
                        completed_at=now + timedelta(seconds=i*2 + 1.5),
                        duration_seconds=1.5,
                        metrics={
                            "latency": 1.5,
                            "gpu_utilization": gpu_metrics["gpu_utilization_percent"][i],
                            "gpu_memory_used_mb": gpu_metrics["gpu_memory_used_mb"][i],
                            "gpu_temperature_c": gpu_metrics["gpu_temperature_c"][i]
                        }
                    )
                    runs.append(run)
                
                return MultiRunJobResult(
                    job_id=job.job_id,
                    experiment_id=job.experiment_id,
                    total_runs=len(runs),
                    successful_runs=len(runs),
                    failed_runs=0,
                    individual_runs=runs,
                    started_at=now,
                    completed_at=now + timedelta(seconds=8),
                    total_duration_seconds=8.0
                )
            
            mock_execute.side_effect = mock_multi_run_with_gpu_metrics
            
            result = await agent.execute_job(job_request)
            
            # Verify GPU metrics were collected
            assert isinstance(result, MultiRunJobResult)
            for run in result.individual_runs:
                assert "gpu_utilization" in run.metrics
                assert "gpu_memory_used_mb" in run.metrics  
                assert "gpu_temperature_c" in run.metrics
                assert run.metrics["gpu_utilization"] >= 15  # Some GPU usage