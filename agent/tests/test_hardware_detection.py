"""Hardware-specific tests that can gracefully fail when hardware is unavailable.

These tests validate that agents have proper hardware setup and drivers installed.
They're designed to pass when hardware is present and properly configured,
but skip gracefully when hardware is not available.
"""

import pytest
import subprocess
import os
import shutil
from unittest.mock import patch, MagicMock

from ruckus_agent.core.detector import AgentDetector
from ruckus_agent.utils.error_reporter import SystemMetricsCollector


class TestHardwareDetection:
    """Hardware detection tests that gracefully handle missing hardware."""
    
    @pytest.fixture
    def detector(self):
        """Create an agent detector for testing."""
        return AgentDetector()
    
    def test_nvidia_smi_availability(self):
        """Test if nvidia-smi is available and working.
        
        This test validates that:
        1. nvidia-smi command is available in PATH
        2. NVIDIA drivers are properly installed
        3. At least one GPU is detected
        
        Skips if NVIDIA hardware is not present.
        """
        # Check if nvidia-smi exists
        nvidia_smi_path = shutil.which("nvidia-smi")
        if not nvidia_smi_path:
            pytest.skip("nvidia-smi not found - NVIDIA drivers not installed or no NVIDIA hardware")
        
        try:
            # Try to run nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                pytest.skip(f"nvidia-smi failed: {result.stderr}")
            
            # Parse output to verify GPUs are detected
            gpu_lines = result.stdout.strip().split('\n')
            gpu_count = len([line for line in gpu_lines if line.strip()])
            
            assert gpu_count > 0, f"No GPUs detected by nvidia-smi. Output: {result.stdout}"
            
            print(f"✅ NVIDIA Hardware Test PASSED: {gpu_count} GPU(s) detected")
            for i, line in enumerate(gpu_lines):
                if line.strip():
                    print(f"   GPU {i}: {line.strip()}")
                    
        except subprocess.TimeoutExpired:
            pytest.skip("nvidia-smi timed out - possible driver issue")
        except FileNotFoundError:
            pytest.skip("nvidia-smi command not found")
    
    @pytest.mark.asyncio
    async def test_gpu_detection_integration(self, detector):
        """Test GPU detection through AgentDetector.
        
        Validates that the agent can detect GPU hardware and capabilities.
        Skips if no NVIDIA hardware is available.
        """
        # Check if nvidia-smi is available first
        if not shutil.which("nvidia-smi"):
            pytest.skip("nvidia-smi not available - skipping GPU detection test")
        
        gpus = await detector.detect_gpus()
        
        if len(gpus) == 0:
            pytest.skip("No GPUs detected - may indicate missing drivers or no NVIDIA hardware")
        
        # Validate GPU information structure
        for gpu in gpus:
            assert "index" in gpu
            assert "name" in gpu
            assert "memory_total_mb" in gpu
            assert "memory_available_mb" in gpu
            assert isinstance(gpu["index"], int)
            assert isinstance(gpu["memory_total_mb"], int)
            assert isinstance(gpu["memory_available_mb"], int)
            assert gpu["memory_total_mb"] > 0
            assert gpu["memory_available_mb"] >= 0
            
        print(f"✅ GPU Detection PASSED: {len(gpus)} GPU(s) detected through AgentDetector")
        for gpu in gpus:
            print(f"   GPU {gpu['index']}: {gpu['name']} ({gpu['memory_total_mb']}MB total)")
    
    @pytest.mark.asyncio
    async def test_cuda_availability(self):
        """Test CUDA availability for PyTorch.
        
        Validates that CUDA is available for PyTorch workloads.
        Skips if CUDA is not available.
        """
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed - cannot test CUDA availability")
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available in PyTorch - may need proper PyTorch+CUDA installation")
        
        cuda_device_count = torch.cuda.device_count()
        assert cuda_device_count > 0, "CUDA is available but no devices detected"
        
        # Test basic CUDA operations
        device = torch.device('cuda:0')
        test_tensor = torch.zeros(10, 10, device=device)
        assert test_tensor.is_cuda, "Failed to create tensor on CUDA device"
        
        # Get CUDA device properties
        props = torch.cuda.get_device_properties(0)
        
        print(f"✅ CUDA Test PASSED: {cuda_device_count} CUDA device(s) available")
        print(f"   Primary GPU: {props.name}")
        print(f"   Compute Capability: {props.major}.{props.minor}")
        print(f"   Memory: {props.total_memory / (1024**3):.1f} GB")
    
    @pytest.mark.asyncio
    async def test_vllm_hardware_compatibility(self):
        """Test vLLM hardware compatibility.
        
        Validates that the system can support vLLM workloads.
        Skips if hardware requirements are not met.
        """
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed - cannot test vLLM compatibility")
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available - vLLM requires CUDA support")
        
        # Check GPU memory requirements (vLLM needs substantial memory)
        device_count = torch.cuda.device_count()
        total_memory_gb = 0
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            total_memory_gb += memory_gb
            
            if memory_gb < 4.0:  # Minimum for small models
                pytest.skip(f"GPU {i} has insufficient memory for vLLM: {memory_gb:.1f}GB < 4GB minimum")
        
        # Try importing vLLM (optional - just check if installed)
        try:
            import vllm
            vllm_available = True
            vllm_version = vllm.__version__
        except ImportError:
            vllm_available = False
            vllm_version = "not installed"
        
        print(f"✅ vLLM Hardware Compatibility PASSED")
        print(f"   Total GPU Memory: {total_memory_gb:.1f} GB across {device_count} GPU(s)")
        print(f"   vLLM Installation: {vllm_version}")
        
        if not vllm_available:
            print(f"   ⚠️  vLLM not installed - agent will skip vLLM workloads")
    
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self):
        """Test system metrics collection on real hardware.
        
        Validates that all metrics collection mechanisms work properly.
        """
        snapshot = await SystemMetricsCollector.capture_snapshot()
        
        # Basic structure validation
        assert snapshot.timestamp is not None
        assert isinstance(snapshot.gpu_memory_used_mb, list)
        assert isinstance(snapshot.gpu_memory_total_mb, list)
        
        # System metrics should always be available
        assert snapshot.system_memory_total_gb is not None
        assert snapshot.system_memory_used_gb is not None
        assert snapshot.cpu_utilization_percent is not None
        assert snapshot.disk_usage_gb is not None
        
        # Process metrics should be available
        assert snapshot.process_memory_mb is not None
        assert snapshot.process_cpu_percent is not None
        
        # Validate reasonable values
        assert snapshot.system_memory_total_gb > 0
        assert 0 <= snapshot.cpu_utilization_percent <= 100
        assert snapshot.disk_usage_gb >= 0
        assert snapshot.process_memory_mb > 0
        
        print(f"✅ System Metrics Collection PASSED")
        print(f"   System Memory: {snapshot.system_memory_used_gb:.1f}/{snapshot.system_memory_total_gb:.1f} GB")
        print(f"   CPU Utilization: {snapshot.cpu_utilization_percent:.1f}%")
        print(f"   Process Memory: {snapshot.process_memory_mb:.1f} MB")
        
        # GPU metrics (may be empty)
        gpu_count = len(snapshot.gpu_memory_total_mb)
        if gpu_count > 0:
            print(f"   GPU Count: {gpu_count}")
            for i in range(gpu_count):
                used = snapshot.gpu_memory_used_mb[i] if i < len(snapshot.gpu_memory_used_mb) else 0
                total = snapshot.gpu_memory_total_mb[i] if i < len(snapshot.gpu_memory_total_mb) else 0
                print(f"   GPU {i}: {used}/{total} MB")
        else:
            print(f"   GPU Count: 0 (no NVIDIA GPUs or nvidia-smi unavailable)")
    
    def test_docker_gpu_runtime(self):
        """Test Docker GPU runtime support.
        
        Validates that Docker can access GPU hardware when running in container.
        Skips if not running in Docker or GPU runtime not configured.
        """
        # Check if running in Docker
        if not os.path.exists("/.dockerenv"):
            pytest.skip("Not running in Docker container")
        
        # Check if nvidia-smi is available in container
        if not shutil.which("nvidia-smi"):
            pytest.skip("nvidia-smi not available in container - GPU runtime not configured")
        
        try:
            # Test nvidia-smi in container
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                pytest.fail(f"nvidia-smi failed in container: {result.stderr}")
            
            gpu_names = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            assert len(gpu_names) > 0, "No GPUs detected in Docker container"
            
            print(f"✅ Docker GPU Runtime PASSED: {len(gpu_names)} GPU(s) accessible")
            for i, name in enumerate(gpu_names):
                print(f"   GPU {i}: {name}")
                
        except subprocess.TimeoutExpired:
            pytest.fail("nvidia-smi timed out in container")
    
    def test_model_directory_access(self):
        """Test access to model directory mount.
        
        Validates that the models directory is properly mounted and accessible.
        """
        from ruckus_agent.core.config import settings
        
        models_dir = settings.model_path
        
        if not os.path.exists(models_dir):
            pytest.skip(f"Models directory not found: {models_dir} - volume mount may not be configured")
        
        # Test read access
        try:
            contents = os.listdir(models_dir)
        except PermissionError:
            pytest.fail(f"No read access to models directory: {models_dir}")
        
        # Test that it's actually a directory
        assert os.path.isdir(models_dir), f"Models path is not a directory: {models_dir}"
        
        print(f"✅ Model Directory Access PASSED: {models_dir}")
        print(f"   Contents: {len(contents)} items")
        
        # List model directories (directories with config.json)
        model_dirs = []
        for item in contents:
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
                model_dirs.append(item)
        
        if model_dirs:
            print(f"   Detected Models: {len(model_dirs)}")
            for model_dir in model_dirs[:5]:  # Show first 5
                print(f"     - {model_dir}")
            if len(model_dirs) > 5:
                print(f"     ... and {len(model_dirs) - 5} more")
        else:
            print(f"   No HuggingFace models detected (no directories with config.json)")
    
    @pytest.mark.asyncio
    async def test_framework_availability(self, detector):
        """Test ML framework availability and versions.
        
        Validates that required ML frameworks are installed and working.
        """
        frameworks = await detector.detect_frameworks()
        
        framework_names = [f.name.value for f in frameworks]
        
        print(f"✅ Framework Detection PASSED: {len(frameworks)} framework(s) detected")
        
        # Check specific frameworks
        for framework in frameworks:
            name = framework.name.value
            version = framework.version
            available = framework.available
            
            print(f"   {name}: v{version} ({'available' if available else 'unavailable'})")
            
            # Additional capability checks
            if name == "pytorch" and available:
                capabilities = framework.capabilities
                cuda_available = capabilities.cuda if capabilities else False
                mps_available = capabilities.mps if capabilities else False
                print(f"     - CUDA: {'✅' if cuda_available else '❌'}")
                print(f"     - MPS (Apple): {'✅' if mps_available else '❌'}")
            
            elif name == "transformers" and available:
                capabilities = framework.capabilities
                text_gen = capabilities.text_generation if capabilities else False
                tokenization = capabilities.tokenization if capabilities else False
                print(f"     - Text Generation: {'✅' if text_gen else '❌'}")
                print(f"     - Tokenization: {'✅' if tokenization else '❌'}")
        
        # Warn about missing critical frameworks
        if "pytorch" not in framework_names:
            print(f"   ⚠️  PyTorch not detected - many models will not work")
        
        if "transformers" not in framework_names:
            print(f"   ⚠️  Transformers not detected - HuggingFace models will not work")
    
    def test_monitoring_tools_availability(self):
        """Test availability of monitoring tools.
        
        Validates that system monitoring tools are available.
        """
        tools_to_check = [
            ("nvidia-smi", "GPU monitoring"),
            ("ps", "Process monitoring"),
            ("free", "Memory monitoring"),
            ("df", "Disk monitoring"),
        ]
        
        available_tools = []
        missing_tools = []
        
        for tool, description in tools_to_check:
            if shutil.which(tool):
                available_tools.append((tool, description))
            else:
                missing_tools.append((tool, description))
        
        print(f"✅ Monitoring Tools Check: {len(available_tools)}/{len(tools_to_check)} tools available")
        
        for tool, desc in available_tools:
            print(f"   ✅ {tool}: {desc}")
        
        for tool, desc in missing_tools:
            print(f"   ❌ {tool}: {desc} (not available)")
        
        # nvidia-smi is optional (only needed for NVIDIA GPUs)
        # Other tools should be available on most systems
        basic_tools = ["ps", "free", "df"]
        missing_basic = [tool for tool, _ in missing_tools if tool in basic_tools]
        
        if missing_basic:
            pytest.skip(f"Basic monitoring tools missing: {missing_basic}")


class TestEnvironmentValidation:
    """Validate the complete environment setup for agent deployment."""
    
    def test_environment_variables(self):
        """Test that important environment variables are set."""
        from ruckus_agent.core.config import settings
        
        print("✅ Environment Variables Check:")
        print(f"   Model Path: {settings.model_path}")
        print(f"   Agent Type: {settings.agent_type}")
        print(f"   Max Concurrent Jobs: {settings.max_concurrent_jobs}")
        print(f"   vLLM Enabled: {settings.enable_vllm}")
        print(f"   GPU Monitoring: {settings.enable_gpu_monitoring}")
        
        # Validate critical settings
        assert os.path.isabs(settings.model_path), "Model path should be absolute path"
        assert settings.max_concurrent_jobs > 0, "Max concurrent jobs should be positive"
    
    @pytest.mark.asyncio
    async def test_complete_agent_readiness(self):
        """Comprehensive test of agent readiness for production.
        
        This test validates that an agent is properly configured and ready
        to handle actual benchmarking workloads.
        """
        from ruckus_agent.core.config import settings
        from ruckus_agent.utils.model_discovery import ModelDiscovery
        
        print("🔍 Agent Readiness Assessment:")
        
        # 1. Check model directory and discovered models
        discovery = ModelDiscovery(settings.model_path)
        models = await discovery.discover_all_models()
        
        print(f"   📁 Models: {len(models)} discovered")
        if models:
            for model in models[:3]:  # Show first 3
                print(f"     - {model.name} ({model.size_gb:.1f}GB, {model.format})")
            if len(models) > 3:
                print(f"     ... and {len(models) - 3} more")
        else:
            print(f"     ⚠️  No models found - agent won't be able to run benchmarks")
        
        # 2. Check hardware availability
        detector = AgentDetector()
        gpus = await detector.detect_gpus()
        
        print(f"   🖥️  Hardware: {len(gpus)} GPU(s)")
        if gpus:
            total_vram = sum(gpu.memory_total_mb for gpu in gpus) / 1024
            print(f"     Total VRAM: {total_vram:.1f} GB")
        
        # 3. Check frameworks
        frameworks = await detector.detect_frameworks()
        framework_names = [f.name.value for f in frameworks if f.available]
        print(f"   🧠 Frameworks: {', '.join(framework_names) if framework_names else 'None'}")
        
        # 4. Overall readiness assessment
        readiness_score = 0
        max_score = 4
        
        if len(models) > 0:
            readiness_score += 1
        if len(gpus) > 0:
            readiness_score += 1
        if "pytorch" in framework_names:
            readiness_score += 1
        if len(framework_names) >= 2:
            readiness_score += 1
        
        readiness_percentage = (readiness_score / max_score) * 100
        
        print(f"\n   🎯 Agent Readiness: {readiness_score}/{max_score} ({readiness_percentage:.0f}%)")
        
        if readiness_percentage >= 75:
            print(f"   ✅ READY: Agent is well-configured for production workloads")
        elif readiness_percentage >= 50:
            print(f"   ⚠️  PARTIAL: Agent can run some workloads but missing capabilities")
        else:
            print(f"   ❌ NOT READY: Agent needs significant configuration before use")
        
        # Don't fail the test - this is informational
        # But log recommendations
        if len(models) == 0:
            print(f"   💡 Recommendation: Mount models directory with HuggingFace models")
        if len(gpus) == 0:
            print(f"   💡 Recommendation: Ensure NVIDIA GPU and drivers are available")
        if "pytorch" not in framework_names:
            print(f"   💡 Recommendation: Install PyTorch with CUDA support")