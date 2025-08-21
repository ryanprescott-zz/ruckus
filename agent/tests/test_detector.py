"""Tests for system detection functionality."""

import pytest
from unittest.mock import patch, Mock, MagicMock, AsyncMock
from ruckus_agent.core.detector import AgentDetector


class TestAgentDetector:
    """Test system detection capabilities."""

    @pytest.mark.asyncio
    async def test_detect_all_structure(self):
        """Test that detect_all returns expected structure."""
        detector = AgentDetector()
        
        with patch.multiple(
            detector,
            detect_system=AsyncMock(return_value={"hostname": "test"}),
            detect_cpu=AsyncMock(return_value={"cores": 4}),
            detect_gpus=AsyncMock(return_value=[]),
            detect_frameworks=AsyncMock(return_value=[]),
            detect_models=AsyncMock(return_value=[]),
            detect_hooks=AsyncMock(return_value=[]),
            detect_metrics=AsyncMock(return_value=[])
        ):
            result = await detector.detect_all()
            
            expected_keys = ["system", "cpu", "gpus", "frameworks", "models", "hooks", "metrics"]
            for key in expected_keys:
                assert key in result

    @pytest.mark.asyncio
    @patch('platform.node')
    @patch('platform.system') 
    @patch('platform.version')
    @patch('platform.release')
    @patch('platform.python_version')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    async def test_detect_system(self, mock_disk_usage, mock_virtual_memory, 
                                mock_python_version, mock_release, mock_version, 
                                mock_system, mock_node):
        """Test system information detection."""
        # Mock platform functions
        mock_node.return_value = "test-hostname"
        mock_system.return_value = "Linux"
        mock_version.return_value = "5.4.0"
        mock_release.return_value = "5.4.0-generic"
        mock_python_version.return_value = "3.12.0"
        
        # Mock psutil
        mock_memory = Mock()
        mock_memory.total = 16 * (1024**3)  # 16GB
        mock_memory.available = 8 * (1024**3)  # 8GB available
        mock_virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.total = 1000 * (1024**3)  # 1TB
        mock_disk.free = 500 * (1024**3)  # 500GB free
        mock_disk_usage.return_value = mock_disk
        
        detector = AgentDetector()
        result = await detector.detect_system()
        
        assert result["hostname"] == "test-hostname"
        assert result["os"] == "Linux"
        assert result["python_version"] == "3.12.0"
        assert result["total_memory_gb"] == pytest.approx(16.0, rel=0.1)
        assert result["available_memory_gb"] == pytest.approx(8.0, rel=0.1)

    @pytest.mark.asyncio
    @patch('platform.processor')
    @patch('platform.machine')
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_freq')
    async def test_detect_cpu(self, mock_cpu_freq, mock_cpu_count, mock_machine, mock_processor):
        """Test CPU information detection."""
        mock_processor.return_value = "Intel Core i7"
        mock_machine.return_value = "x86_64"
        
        mock_cpu_count.side_effect = lambda logical=True: 8 if logical else 4
        
        mock_freq = Mock()
        mock_freq.current = 2800.0
        mock_cpu_freq.return_value = mock_freq
        
        detector = AgentDetector()
        result = await detector.detect_cpu()
        
        assert result["model"] == "Intel Core i7"
        assert result["cores_physical"] == 4
        assert result["cores_logical"] == 8
        assert result["frequency_mhz"] == 2800.0
        assert result["architecture"] == "x86_64"

    @pytest.mark.asyncio
    @patch('ruckus_agent.core.detector.subprocess')
    async def test_detect_gpus_nvidia(self, mock_subprocess):
        """Test GPU detection with nvidia-smi."""
        # Mock successful nvidia-smi output
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "0, Tesla V100, GPU-12345, 32510, 30000"
        mock_subprocess.run.return_value = mock_result
        
        detector = AgentDetector()
        result = await detector.detect_gpus()
        
        assert len(result) == 1
        gpu = result[0]
        assert gpu["index"] == 0
        assert gpu["name"] == "Tesla V100"
        assert gpu["uuid"] == "GPU-12345"
        assert gpu["memory_total_mb"] == 32510
        assert gpu["memory_available_mb"] == 30000

    @pytest.mark.asyncio
    @patch('ruckus_agent.core.detector.subprocess')
    async def test_detect_gpus_no_nvidia(self, mock_subprocess):
        """Test GPU detection when nvidia-smi fails."""
        # Mock failed nvidia-smi
        mock_subprocess.run.side_effect = FileNotFoundError()
        
        detector = AgentDetector()
        result = await detector.detect_gpus()
        
        assert result == []

    @pytest.mark.asyncio
    async def test_detect_frameworks_mock_imports(self):
        """Test framework detection with mocked imports."""
        detector = AgentDetector()
        
        # Mock transformers import
        with patch.dict('sys.modules', {
            'transformers': Mock(__version__="4.20.0")
        }):
            result = await detector.detect_frameworks()
            
            # Should detect transformers
            transformers_found = any(fw["name"] == "transformers" for fw in result)
            assert transformers_found
            
            if transformers_found:
                transformers_fw = next(fw for fw in result if fw["name"] == "transformers")
                assert transformers_fw["version"] == "4.20.0"
                assert transformers_fw["available"] is True

    @pytest.mark.asyncio
    @patch('ruckus_agent.core.detector.subprocess')
    async def test_detect_hooks(self, mock_subprocess):
        """Test system hooks detection."""
        # Mock which command success
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = b"/usr/bin/nvidia-smi\n"
        mock_subprocess.run.return_value = mock_result
        
        detector = AgentDetector()
        result = await detector.detect_hooks()
        
        assert len(result) >= 1
        nvidia_hook = next((h for h in result if h["name"] == "nvidia-smi"), None)
        assert nvidia_hook is not None
        assert nvidia_hook["type"] == "gpu_monitor"
        assert nvidia_hook["executable_path"] == "/usr/bin/nvidia-smi"
        assert nvidia_hook["working"] is True

    @pytest.mark.asyncio
    async def test_detect_metrics_basic(self):
        """Test basic metrics detection."""
        detector = AgentDetector()
        
        with patch.object(detector, 'detect_gpus', return_value=[]):
            result = await detector.detect_metrics()
            
            # Should always have basic performance metrics
            metric_names = [m["name"] for m in result]
            assert "latency" in metric_names
            assert "throughput" in metric_names
            
            latency_metric = next(m for m in result if m["name"] == "latency")
            assert latency_metric["type"] == "performance"
            assert latency_metric["available"] is True

    @pytest.mark.asyncio
    async def test_detect_metrics_with_gpu(self):
        """Test metrics detection when GPU is available."""
        detector = AgentDetector()
        
        mock_gpu = {"name": "Tesla", "memory_mb": 8000}
        with patch.object(detector, 'detect_gpus', return_value=[mock_gpu]):
            result = await detector.detect_metrics()
            
            metric_names = [m["name"] for m in result]
            assert "gpu_utilization" in metric_names
            assert "gpu_memory" in metric_names
            
            gpu_metric = next(m for m in result if m["name"] == "gpu_utilization")
            assert gpu_metric["type"] == "resource"
            assert "nvidia-smi" in gpu_metric["requires"]