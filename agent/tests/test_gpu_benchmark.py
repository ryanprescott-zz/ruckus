"""Tests for GPU benchmarking and detection functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import torch
import numpy as np
from typing import Dict, Any, List

from ruckus_agent.utils.gpu_benchmark import GpuBenchmark
from ruckus_agent.core.detector import AgentDetector


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


class TestGpuBenchmark:
    """Test GPU benchmarking functionality."""

    def test_gpu_benchmark_initialization(self):
        """Test GpuBenchmark initialization."""
        benchmark = GpuBenchmark()
        assert benchmark.device is not None
        assert isinstance(benchmark.tensor_sizes, list)
        assert len(benchmark.tensor_sizes) > 0

    def test_tensor_core_generation_mapping(self):
        """Test tensor core generation detection."""
        benchmark = GpuBenchmark()
        
        # Test known compute capabilities
        assert benchmark._get_tensor_core_generation((7, 0)) == 1  # V100
        assert benchmark._get_tensor_core_generation((7, 5)) == 2  # RTX 20 series
        assert benchmark._get_tensor_core_generation((8, 6)) == 3  # RTX 30 series  
        assert benchmark._get_tensor_core_generation((8, 9)) == 4  # RTX 40 series
        assert benchmark._get_tensor_core_generation((9, 0)) == 5  # H100
        
        # Test unknown compute capability
        assert benchmark._get_tensor_core_generation((6, 1)) == 0  # Pre-tensor core

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_capability')
    def test_detect_tensor_core_capabilities(self, mock_capability, mock_cuda_available):
        """Test tensor core capability detection."""
        mock_cuda_available.return_value = True
        mock_capability.return_value = (8, 9)  # RTX 4090
        
        benchmark = GpuBenchmark()
        capabilities = benchmark._detect_tensor_core_capabilities()
        
        assert "generation" in capabilities
        assert capabilities["generation"] == 4
        assert "supported_precisions" in capabilities
        assert "FP16" in capabilities["supported_precisions"]
        assert "BF16" in capabilities["supported_precisions"]
        assert "FP8" in capabilities["supported_precisions"]

    def test_get_tensor_sizes_for_vram(self):
        """Test tensor size selection based on VRAM."""
        benchmark = GpuBenchmark()
        
        # Test different VRAM amounts
        sizes_8gb = benchmark._get_tensor_sizes_for_vram(8192)
        sizes_24gb = benchmark._get_tensor_sizes_for_vram(24576)
        sizes_80gb = benchmark._get_tensor_sizes_for_vram(81920)
        
        # Larger VRAM should allow larger tensors
        assert max(sizes_8gb) <= max(sizes_24gb)
        assert max(sizes_24gb) <= max(sizes_80gb)
        
        # All should be reasonable sizes
        for sizes in [sizes_8gb, sizes_24gb, sizes_80gb]:
            assert all(size >= 64 for size in sizes)  # Minimum size
            assert all(size <= 32768 for size in sizes)  # Maximum reasonable size

    @patch('torch.cuda.is_available')
    @patch('torch.randn')
    @patch('torch.matmul')
    def test_memory_bandwidth_benchmark(self, mock_matmul, mock_randn, mock_cuda_available):
        """Test memory bandwidth benchmarking."""
        mock_cuda_available.return_value = True
        
        # Mock tensor creation
        mock_tensor = MagicMock()
        mock_tensor.cuda.return_value = mock_tensor
        mock_tensor.shape = (1024, 1024)
        mock_randn.return_value = mock_tensor
        
        # Mock matrix multiplication
        mock_matmul.return_value = mock_tensor
        
        # Mock CUDA synchronization
        with patch('torch.cuda.synchronize'):
            benchmark = GpuBenchmark()
            
            with patch.object(benchmark, '_get_tensor_sizes_for_vram', return_value=[512, 1024]):
                results = benchmark._benchmark_memory_bandwidth(vram_mb=8192)
                
                assert "bandwidth_results" in results
                assert "peak_bandwidth_gb_s" in results
                assert isinstance(results["bandwidth_results"], list)

    @patch('torch.cuda.is_available')
    def test_flops_benchmark_multiple_precisions(self, mock_cuda_available):
        """Test FLOPS benchmarking across different precisions."""
        mock_cuda_available.return_value = True
        
        benchmark = GpuBenchmark()
        
        # Mock tensor operations for different dtypes
        with patch('torch.randn') as mock_randn, \
             patch('torch.matmul') as mock_matmul, \
             patch('torch.cuda.synchronize'):
            
            mock_tensor = MagicMock()
            mock_tensor.cuda.return_value = mock_tensor
            mock_tensor.to.return_value = mock_tensor
            mock_randn.return_value = mock_tensor
            mock_matmul.return_value = mock_tensor
            
            # Test specific precision
            results = benchmark._benchmark_flops(
                tensor_size=1024,
                dtype=torch.float16,
                num_iterations=10
            )
            
            assert "avg_time_ms" in results
            assert "tflops" in results
            assert "operations_per_second" in results
            assert results["dtype"] == "float16"

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    def test_full_gpu_benchmark_integration(self, mock_get_props, mock_cuda_available):
        """Test full GPU benchmark integration."""
        mock_cuda_available.return_value = True
        
        # Mock device properties
        mock_props = MagicMock()
        mock_props.name = "NVIDIA GeForce RTX 4090"
        mock_props.total_memory = 25769803776  # 24GB
        mock_get_props.return_value = mock_props
        
        benchmark = GpuBenchmark()
        
        with patch.object(benchmark, '_benchmark_memory_bandwidth') as mock_bandwidth, \
             patch.object(benchmark, '_benchmark_flops') as mock_flops, \
             patch.object(benchmark, '_detect_tensor_core_capabilities') as mock_tensor_caps:
            
            # Mock benchmark results
            mock_bandwidth.return_value = {
                "bandwidth_results": [{"size": 1024, "bandwidth_gb_s": 800.0}],
                "peak_bandwidth_gb_s": 800.0
            }
            
            mock_flops.return_value = {
                "avg_time_ms": 5.0,
                "tflops": 100.0,
                "operations_per_second": 100000000000000,
                "dtype": "float16"
            }
            
            mock_tensor_caps.return_value = {
                "generation": 4,
                "supported_precisions": ["FP32", "FP16", "BF16", "FP8"]
            }
            
            # Run full benchmark
            results = benchmark.run_full_benchmark()
            
            assert "device_info" in results
            assert "tensor_core_capabilities" in results
            assert "memory_bandwidth" in results
            assert "compute_performance" in results
            assert results["device_info"]["name"] == "NVIDIA GeForce RTX 4090"


class TestGpuDetectionIntegration:
    """Test GPU detection integration with AgentDetector."""

    @patch('subprocess.run')
    @patch('pynvml.nvmlInit')
    def test_detector_gpu_integration_with_nvidia_smi(self, mock_nvml_init, mock_subprocess, mock_nvidia_smi):
        """Test AgentDetector GPU detection with nvidia-smi."""
        # Mock nvidia-smi subprocess call
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_nvidia_smi
        mock_subprocess.return_value = mock_result
        
        # Mock pynvml initialization failure (fallback to nvidia-smi)
        mock_nvml_init.side_effect = Exception("pynvml not available")
        
        detector = AgentDetector()
        
        with patch('shutil.which', return_value='/usr/bin/nvidia-smi'):
            gpu_info = detector._detect_gpus_nvidia_smi()
            
            assert len(gpu_info) > 0
            assert gpu_info[0]["name"] == "NVIDIA GeForce RTX 4090"
            assert gpu_info[0]["memory_total_mb"] == 24564
            assert gpu_info[0]["memory_free_mb"] == 23434
            assert gpu_info[0]["driver_version"] == "535.104.05"
            assert gpu_info[0]["cuda_version"] == "12.2"

    def test_detector_gpu_integration_with_pynvml(self, mock_pynvml):
        """Test AgentDetector GPU detection with pynvml."""
        detector = AgentDetector()
        
        with patch('pynvml.nvmlInit'), \
             patch('pynvml.nvmlDeviceGetCount', return_value=1), \
             patch('pynvml.nvmlDeviceGetHandleByIndex', return_value=MagicMock()), \
             patch('pynvml.nvmlDeviceGetName', return_value=b"NVIDIA GeForce RTX 4090"), \
             patch('pynvml.nvmlDeviceGetMemoryInfo') as mock_mem_info, \
             patch('pynvml.nvmlDeviceGetCudaComputeCapability', return_value=(8, 9)), \
             patch('pynvml.nvmlDeviceGetTemperature', return_value=45), \
             patch('pynvml.nvmlDeviceGetPowerUsage', return_value=25000), \
             patch('pynvml.nvmlDeviceGetUtilizationRates') as mock_util:
            
            # Mock memory info
            mock_mem_info.return_value = MagicMock(
                total=25769803776,  # 24GB
                free=24317591552,   # ~22.7GB 
                used=1452212224     # ~1.3GB
            )
            
            # Mock utilization
            mock_util.return_value = MagicMock(gpu=15, memory=4)
            
            gpu_info = detector._detect_gpus_pynvml()
            
            assert len(gpu_info) > 0
            assert gpu_info[0]["name"] == "NVIDIA GeForce RTX 4090"
            assert gpu_info[0]["compute_capability"] == [8, 9]
            assert gpu_info[0]["tensor_core_generation"] == 4
            assert gpu_info[0]["memory_total_mb"] == 24576  # ~24GB
            assert gpu_info[0]["temperature_c"] == 45
            assert gpu_info[0]["power_usage_w"] == 25

    @patch('torch.cuda.is_available')
    @patch('torch.version.cuda') 
    def test_detector_pytorch_gpu_fallback(self, mock_cuda_version, mock_cuda_available):
        """Test AgentDetector PyTorch GPU detection fallback."""
        mock_cuda_available.return_value = True
        mock_cuda_version = "12.1"
        
        detector = AgentDetector()
        
        with patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_properties') as mock_props, \
             patch('torch.cuda.get_device_capability', return_value=(8, 9)), \
             patch('torch.cuda.memory_stats') as mock_mem_stats:
            
            # Mock device properties
            mock_props.return_value = MagicMock(
                name="NVIDIA GeForce RTX 4090",
                total_memory=25769803776,  # 24GB
                major=8,
                minor=9
            )
            
            # Mock memory stats
            mock_mem_stats.return_value = {
                'allocated_bytes.all.current': 1452212224,  # ~1.3GB
                'reserved_bytes.all.current': 2147483648    # ~2GB
            }
            
            gpu_info = detector._detect_gpus_pytorch()
            
            assert len(gpu_info) > 0
            assert gpu_info[0]["name"] == "NVIDIA GeForce RTX 4090"
            assert gpu_info[0]["compute_capability"] == [8, 9]
            assert gpu_info[0]["tensor_core_generation"] == 4
            assert gpu_info[0]["memory_total_mb"] == 24576

    @pytest.mark.asyncio
    @patch('torch.cuda.is_available')
    async def test_full_detector_integration_with_benchmarking(self, mock_cuda_available):
        """Test full detector integration including GPU benchmarking."""
        mock_cuda_available.return_value = True
        
        detector = AgentDetector()
        
        with patch.object(detector, '_detect_gpus_pynvml') as mock_pynvml_detect, \
             patch.object(detector, '_detect_gpus_pytorch') as mock_pytorch_detect, \
             patch('ruckus_agent.utils.gpu_benchmark.GpuBenchmark') as mock_benchmark_class:
            
            # Mock GPU detection results
            mock_pynvml_detect.return_value = [{
                "name": "NVIDIA GeForce RTX 4090",
                "compute_capability": [8, 9],
                "tensor_core_generation": 4,
                "memory_total_mb": 24576,
                "memory_free_mb": 22760,
                "memory_used_mb": 1316,
                "temperature_c": 45,
                "power_usage_w": 25,
                "utilization_gpu": 15,
                "utilization_memory": 4
            }]
            
            # Mock PyTorch fallback (shouldn't be called if pynvml succeeds)
            mock_pytorch_detect.return_value = []
            
            # Mock GPU benchmark
            mock_benchmark = MagicMock()
            mock_benchmark_class.return_value = mock_benchmark
            mock_benchmark.run_full_benchmark.return_value = {
                "device_info": {
                    "name": "NVIDIA GeForce RTX 4090",
                    "memory_gb": 24.0,
                    "compute_capability": (8, 9)
                },
                "tensor_core_capabilities": {
                    "generation": 4,
                    "supported_precisions": ["FP32", "FP16", "BF16", "FP8"]
                },
                "memory_bandwidth": {
                    "peak_bandwidth_gb_s": 800.0,
                    "bandwidth_results": [
                        {"size": 1024, "bandwidth_gb_s": 600.0},
                        {"size": 2048, "bandwidth_gb_s": 750.0},
                        {"size": 4096, "bandwidth_gb_s": 800.0}
                    ]
                },
                "compute_performance": {
                    "FP32": {"tflops": 50.0, "avg_time_ms": 10.0},
                    "FP16": {"tflops": 120.0, "avg_time_ms": 4.0},
                    "BF16": {"tflops": 110.0, "avg_time_ms": 4.5}
                }
            }
            
            # Run detection with benchmarking enabled
            detected_info = await detector.detect_all(run_gpu_benchmark=True)
            
            assert "gpus" in detected_info
            assert len(detected_info["gpus"]) > 0
            
            gpu = detected_info["gpus"][0]
            assert gpu["name"] == "NVIDIA GeForce RTX 4090"
            assert "benchmark_results" in gpu
            assert gpu["benchmark_results"]["tensor_core_capabilities"]["generation"] == 4
            assert gpu["benchmark_results"]["memory_bandwidth"]["peak_bandwidth_gb_s"] == 800.0
            
            # Verify benchmark was called
            mock_benchmark.run_full_benchmark.assert_called_once()


class TestGpuBenchmarkMocking:
    """Test proper mocking of GPU components for testing."""

    def test_mock_gpu_unavailable_scenario(self):
        """Test behavior when GPU is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            benchmark = GpuBenchmark()
            
            # Should handle gracefully
            assert benchmark.device == torch.device('cpu')
            
            # Benchmarks should return empty/default results
            results = benchmark.run_full_benchmark()
            assert results["device_info"]["name"] == "CPU (No GPU Available)"

    @patch('pynvml.nvmlInit')
    def test_mock_pynvml_initialization_failure(self, mock_nvml_init):
        """Test fallback when pynvml initialization fails."""
        mock_nvml_init.side_effect = Exception("NVML initialization failed")
        
        detector = AgentDetector()
        
        with patch.object(detector, '_detect_gpus_pytorch') as mock_pytorch_fallback:
            mock_pytorch_fallback.return_value = [{
                "name": "Fallback GPU",
                "memory_total_mb": 8192,
                "compute_capability": [7, 5]
            }]
            
            # Should fall back to PyTorch detection
            gpus = detector._detect_gpus()
            
            assert len(gpus) > 0
            assert gpus[0]["name"] == "Fallback GPU"
            mock_pytorch_fallback.assert_called_once()

    def test_mock_nvidia_smi_unavailable(self):
        """Test behavior when nvidia-smi is not available."""
        detector = AgentDetector()
        
        with patch('shutil.which', return_value=None):  # nvidia-smi not found
            gpus = detector._detect_gpus_nvidia_smi()
            
            # Should return empty list when nvidia-smi unavailable
            assert gpus == []

    @patch('subprocess.run')
    def test_mock_nvidia_smi_malformed_output(self, mock_subprocess):
        """Test handling of malformed nvidia-smi output."""
        # Mock malformed XML output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Not valid XML output"
        mock_subprocess.return_value = mock_result
        
        detector = AgentDetector()
        
        with patch('shutil.which', return_value='/usr/bin/nvidia-smi'):
            gpus = detector._detect_gpus_nvidia_smi()
            
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
        
        with patch.object(agent, '_collect_gpu_metrics_during_execution') as mock_gpu_metrics:
            mock_gpu_metrics.return_value = {
                "gpu_utilization_percent": [20, 85, 15],
                "gpu_memory_used_mb": [2048, 4096, 2048], 
                "gpu_temperature_c": [55, 78, 52],
                "gpu_power_usage_w": [150, 300, 120]
            }
            
            with patch.object(agent, 'execute_job') as mock_execute:
                # Mock successful multi-run execution with GPU metrics
                async def mock_multi_run_with_gpu_metrics(job):
                    from datetime import datetime, timedelta
                    now = datetime.utcnow()
                    
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
                                "gpu_utilization": mock_gpu_metrics.return_value["gpu_utilization_percent"][i],
                                "gpu_memory_used_mb": mock_gpu_metrics.return_value["gpu_memory_used_mb"][i],
                                "gpu_temperature_c": mock_gpu_metrics.return_value["gpu_temperature_c"][i]
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