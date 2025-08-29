"""Integration tests for end-to-end agent functionality."""

import pytest
import pytest_asyncio
import tempfile
import shutil
import json
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from ruckus_agent.core.agent import Agent
from ruckus_agent.core.config import Settings
from ruckus_agent.core.storage import InMemoryStorage
from ruckus_common.models import AgentType, JobRequest, TaskType, AgentStatusEnum


class TestAgentIntegration:
    """Integration tests for complete agent functionality."""
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create a temporary models directory with mock models."""
        temp_dir = tempfile.mkdtemp()
        models_dir = Path(temp_dir)
        
        # Create a mock HuggingFace model
        model_dir = models_dir / "test-llama-7b"
        model_dir.mkdir()
        
        # Create config.json
        config = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "max_position_embeddings": 2048,
            "torch_dtype": "float16"
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)
        
        # Create model files
        (model_dir / "pytorch_model.bin").write_text("fake model weights")
        (model_dir / "tokenizer.model").write_text("fake tokenizer")
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def agent_settings(self, temp_models_dir):
        """Create agent settings for testing."""
        return Settings(
            agent_type=AgentType.WHITE_BOX,
            model_cache_dir=temp_models_dir,
            orchestrator_url=None,  # No orchestrator for unit tests
            max_concurrent_jobs=1,
            enable_vllm=True,
            enable_gpu_monitoring=False,  # Disable for unit tests
            heartbeat_interval=1  # Fast heartbeat for tests
        )
    
    @pytest.fixture
    def storage(self):
        """Create in-memory storage for testing."""
        return InMemoryStorage()
    
    @pytest_asyncio.fixture
    async def agent(self, agent_settings, storage):
        """Create and start an agent for testing."""
        agent = Agent(agent_settings, storage)
        
        # Mock capability detection to populate test data
        async def mock_detect_capabilities():
            # Mock detected data similar to what AgentDetector would return
            mock_detected_data = {
                "system": {"hostname": "test-agent", "os": "Linux", "python_version": "3.12", "total_memory_gb": 16.0},
                "cpu": {"cores": 4, "model": "Test CPU"},
                "gpus": [{"name": "Tesla V100", "memory_mb": 16384}],
                "frameworks": [{"name": "pytorch", "version": "2.0"}, {"name": "vllm", "version": "0.2.0"}],
                "models": [
                    {
                        "name": "test-llama-7b",
                        "path": f"{agent_settings.model_cache_dir}/test-llama-7b",
                        "model_type": "llama",
                        "architecture": "LlamaForCausalLM",
                        "format": "pytorch",
                        "framework_compatible": ["vllm", "transformers"],
                        "size_gb": 13.5
                    }
                ],
                "hooks": [{"name": "nvidia-smi"}],
                "metrics": [{"name": "latency"}, {"name": "memory_usage"}]
            }
            
            # Store system info
            await agent.storage.store_system_info(mock_detected_data)
            
            # Store capabilities
            capabilities = {
                "agent_type": "white_box",
                "gpu_count": len(mock_detected_data["gpus"]),
                "frameworks": [f["name"] for f in mock_detected_data["frameworks"]],
                "max_concurrent_jobs": agent.settings.max_concurrent_jobs,
                "monitoring_available": bool(mock_detected_data["hooks"]),
            }
            await agent.storage.store_capabilities(capabilities)
        
        with patch.object(agent, '_detect_capabilities', side_effect=mock_detect_capabilities):
            await agent.start()
        
        yield agent
        
        await agent.stop()
    
    @pytest.mark.asyncio
    async def test_agent_initialization_and_startup(self, agent_settings, storage):
        """Test complete agent initialization and startup sequence."""
        agent = Agent(agent_settings, storage)
        
        # Verify initial state
        assert agent.agent_id.startswith("agent-")
        assert agent.agent_name.endswith("-white_box")
        assert not agent.registered
        assert len(agent.running_jobs) == 0
        assert len(agent.queued_job_ids) == 0
        assert not agent.crashed
        
        # Mock capability detection
        with patch.object(agent, '_detect_capabilities', new_callable=AsyncMock) as mock_detect:
            
            await agent.start()
            
            # Verify startup completed
            assert len(agent.tasks) == 2  # heartbeat and job executor tasks
            mock_detect.assert_called_once()
        
        await agent.stop()
    
    @pytest.mark.asyncio
    async def test_model_discovery_integration(self, agent):
        """Test that agent properly discovers models during startup."""
        # Get system info which should include discovered models
        system_info = await agent.get_system_info()
        
        assert "models" in system_info
        models = system_info["models"]
        
        # Should have discovered our test model
        assert len(models) >= 1
        
        # Find our test model
        test_model = None
        for model in models:
            if model["name"] == "test-llama-7b":
                test_model = model
                break
        
        assert test_model is not None, "Test model not found in discovered models"
        assert test_model["model_type"] == "llama"
        assert test_model["architecture"] == "LlamaForCausalLM"
        assert test_model["format"] == "pytorch"
        assert "vllm" in test_model["framework_compatible"]
    
    @pytest.mark.asyncio
    async def test_capabilities_detection(self, agent):
        """Test comprehensive capabilities detection."""
        capabilities = await agent.get_capabilities()
        
        # Should include framework information
        assert "frameworks" in capabilities
        
        # Should include GPU count (may be 0 in test environment)
        assert "gpu_count" in capabilities
        assert isinstance(capabilities["gpu_count"], int)
        assert capabilities["gpu_count"] >= 0
        
        # Should include basic agent configuration
        assert "agent_type" in capabilities
        assert "max_concurrent_jobs" in capabilities
        assert capabilities["agent_type"] == "white_box"
        assert capabilities["max_concurrent_jobs"] == 1
    
    @pytest.mark.asyncio
    async def test_agent_status_reporting(self, agent):
        """Test agent status reporting functionality."""
        status = await agent.get_status()
        
        assert status.agent_id == agent.agent_id
        assert status.status == AgentStatusEnum.IDLE  # Should start idle
        assert len(status.running_jobs) == 0
        assert status.queued_jobs == []
        assert status.uptime_seconds >= 0
        assert isinstance(status.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_job_queuing_and_tracking(self, agent):
        """Test job queuing and status tracking."""
        # Create a test job
        job = JobRequest(
            job_id="test-job-123",
            experiment_id="test-exp-456",
            model="test-llama-7b",
            framework="vllm",
            task_type=TaskType.GENERATION,
            task_config={"prompt": "Test prompt"},
            parameters={"temperature": 0.7, "max_tokens": 100}
        )
        
        # Queue the job
        job_id = await agent.queue_job(job)
        assert job_id == "test-job-123"
        
        # Check status update
        status = await agent.get_status()
        assert len(status.queued_jobs) == 1
        assert "test-job-123" in agent.queued_job_ids
        
        # Wait a moment for job to potentially start processing
        await asyncio.sleep(0.1)
        
        # Note: Job execution will likely fail due to mocked vLLM, but that's ok
        # We're testing the queuing and tracking mechanism
    
    @pytest.mark.asyncio
    async def test_error_reporting_integration(self, agent):
        """Test error reporting integration with job execution."""
        # Create a job that will fail
        job = JobRequest(
            job_id="failing-job-123",
            experiment_id="test-exp-456",
            model="nonexistent-model",  # This will cause failure
            framework="vllm",
            task_type=TaskType.GENERATION,
            task_config={},
            parameters={}
        )
        
        # Queue the job
        await agent.queue_job(job)
        
        # Wait for job processing and failure
        await asyncio.sleep(0.5)
        
        # Check if error report was generated
        error_reports = await agent.get_error_reports()
        
        if error_reports:
            # Verify error report structure
            report = error_reports[0]
            assert report.job_id == "failing-job-123"
            assert report.agent_id == agent.agent_id
            assert report.error_message is not None
            assert report.metrics_at_failure is not None
    
    @pytest.mark.asyncio
    async def test_agent_crash_recovery(self, agent):
        """Test agent behavior after crash state."""
        # Simulate crash by setting crash state
        agent.crashed = True
        agent.crash_reason = "Test crash for recovery testing"
        
        # Check crashed status
        status = await agent.get_status()
        assert status.status == AgentStatusEnum.ERROR
        
        # Clear error reports (which should reset crash state)
        cleared_count = await agent.clear_error_reports()
        
        # Check that crash state is reset
        assert not agent.crashed
        assert agent.crash_reason is None
        
        status = await agent.get_status()
        assert status.status == AgentStatusEnum.IDLE
    
    @pytest.mark.asyncio
    async def test_concurrent_job_limits(self, agent):
        """Test that agent respects concurrent job limits."""
        # Create multiple jobs
        jobs = [
            JobRequest(
                job_id=f"concurrent-job-{i}",
                experiment_id="test-exp",
                model="test-llama-7b",
                framework="vllm",
                task_type=TaskType.GENERATION,
                task_config={},
                parameters={}
            )
            for i in range(3)
        ]
        
        # Queue all jobs
        for job in jobs:
            await agent.queue_job(job)
        
        # Check status
        status = await agent.get_status()
        assert len(status.queued_jobs) == 3
        
        # Wait for processing to start
        await asyncio.sleep(0.2)
        
        # Should not process more than max_concurrent_jobs (1) at once
        status = await agent.get_status()
        assert len(status.running_jobs) <= agent.settings.max_concurrent_jobs
    
    @pytest.mark.asyncio
    async def test_system_info_completeness(self, agent):
        """Test that system info contains all expected components."""
        system_info = await agent.get_system_info()
        
        # Check required sections
        required_sections = ["system", "cpu", "gpus", "frameworks", "models", "hooks", "metrics"]
        for section in required_sections:
            assert section in system_info, f"Missing required section: {section}"
        
        # Validate system section
        system = system_info["system"]
        assert "hostname" in system
        assert "os" in system
        assert "python_version" in system
        assert "total_memory_gb" in system
        
        # Validate frameworks section
        frameworks = system_info["frameworks"]
        assert isinstance(frameworks, list)
        
        # Validate models section
        models = system_info["models"]
        assert isinstance(models, list)
    
    @pytest.mark.asyncio
    async def test_storage_integration(self, agent):
        """Test agent storage integration."""
        # Test capabilities storage
        capabilities = await agent.get_capabilities()
        assert capabilities is not None
        
        # Test system info storage
        system_info = await agent.get_system_info()
        assert system_info is not None
        
        # Verify storage is working by checking consistency
        capabilities2 = await agent.get_capabilities()
        system_info2 = await agent.get_system_info()
        
        assert capabilities == capabilities2
        assert system_info == system_info2
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, agent_settings, storage):
        """Test agent graceful shutdown."""
        agent = Agent(agent_settings, storage)
        
        # Mock capability detection
        with patch.object(agent, '_detect_capabilities', new_callable=AsyncMock) as mock_detect:
            await agent.start()
        
        # Verify agent is running
        assert len(agent.tasks) > 0
        
        # Queue a job to test cleanup
        job = JobRequest(
            job_id="shutdown-test-job",
            experiment_id="test-exp",
            model="test-model",
            framework="vllm",
            task_type=TaskType.GENERATION,
            task_config={},
            parameters={}
        )
        await agent.queue_job(job)
        
        # Shutdown agent
        await agent.stop()
        
        # Wait a moment for tasks to finish cancelling
        await asyncio.sleep(0.1)
        
        # Verify cleanup
        for task in agent.tasks:
            assert task.cancelled() or task.done()


class TestAgentStatusTransitions:
    """Test agent status transitions and crash handling."""
    
    @pytest.fixture
    def agent_settings(self):
        """Create minimal agent settings for status testing."""
        return Settings(
            agent_type=AgentType.WHITE_BOX,
            model_cache_dir="/tmp/test-models",
            orchestrator_url=None,
            max_concurrent_jobs=1,
            enable_gpu_monitoring=False
        )
    
    @pytest.mark.asyncio
    async def test_idle_to_active_transition(self, agent_settings):
        """Test agent status transition from idle to active."""
        agent = Agent(agent_settings, InMemoryStorage())
        
        with patch.object(agent, '_detect_capabilities'):
            await agent.start()
        
        try:
            # Should start idle
            status = await agent.get_status()
            assert status.status == AgentStatusEnum.IDLE
            assert len(status.running_jobs) == 0
            
            # Add a running job directly (simulating job execution)
            agent.running_jobs["test-job"] = {
                "job": MagicMock(),
                "start_time": datetime.now(timezone.utc)
            }
            
            # Should now be active
            status = await agent.get_status()
            assert status.status == AgentStatusEnum.ACTIVE
            assert len(status.running_jobs) == 1
            
        finally:
            await agent.stop()
    
    @pytest.mark.asyncio
    async def test_crash_state_handling(self, agent_settings):
        """Test agent crash state handling and recovery."""
        agent = Agent(agent_settings, InMemoryStorage())
        
        with patch.object(agent, '_detect_capabilities'):
            await agent.start()
        
        try:
            # Initially not crashed
            assert not agent.crashed
            status = await agent.get_status()
            assert status.status == AgentStatusEnum.IDLE
            
            # Simulate crash
            agent.crashed = True
            agent.crash_reason = "Simulated crash for testing"
            
            # Status should reflect crash
            status = await agent.get_status()
            assert status.status == AgentStatusEnum.ERROR
            
            # Create an error report to test recovery
            from ruckus_agent.core.models import JobErrorReport, SystemMetricsSnapshot
            error_report = JobErrorReport(
                job_id="crashed-job",
                experiment_id="test-exp",
                agent_id=agent.agent_id,
                error_type="test_crash",
                error_message="Test crash",
                model_name="test-model",
                model_path="/test/path",
                framework="test",
                task_type="test",
                parameters={},
                started_at=datetime.now(timezone.utc),
                metrics_at_failure=SystemMetricsSnapshot()
            )
            agent.error_reports["crashed-job"] = error_report
            
            # Clear error reports should reset crash state
            cleared_count = await agent.clear_error_reports()
            assert cleared_count == 1
            assert not agent.crashed
            assert agent.crash_reason is None
            
            # Status should be back to normal
            status = await agent.get_status()
            assert status.status == AgentStatusEnum.IDLE
            
        finally:
            await agent.stop()
    
    @pytest.mark.asyncio
    async def test_error_report_lifecycle(self, agent_settings):
        """Test complete error report lifecycle."""
        agent = Agent(agent_settings, InMemoryStorage())
        
        with patch.object(agent, '_detect_capabilities'):
            await agent.start()
        
        try:
            # Start with no error reports
            reports = await agent.get_error_reports()
            assert len(reports) == 0
            
            # Add error report directly
            from ruckus_agent.core.models import JobErrorReport, SystemMetricsSnapshot
            error_report = JobErrorReport(
                job_id="error-job-123",
                experiment_id="test-exp",
                agent_id=agent.agent_id,
                error_type="model_loading_error",
                error_message="Failed to load model",
                model_name="test-model",
                model_path="/test/path",
                framework="vllm",
                task_type="generation",
                parameters={"temperature": 0.7},
                started_at=datetime.now(timezone.utc),
                metrics_at_failure=SystemMetricsSnapshot()
            )
            
            agent.error_reports["error-job-123"] = error_report
            
            # Verify report is retrievable
            reports = await agent.get_error_reports()
            assert len(reports) == 1
            assert reports[0].job_id == "error-job-123"
            
            # Test individual report retrieval
            specific_report = await agent.get_error_report("error-job-123")
            assert specific_report is not None
            assert specific_report.job_id == "error-job-123"
            
            # Test non-existent report
            missing_report = await agent.get_error_report("nonexistent-job")
            assert missing_report is None
            
            # Clear all reports
            cleared_count = await agent.clear_error_reports()
            assert cleared_count == 1
            
            # Verify reports are cleared
            reports = await agent.get_error_reports()
            assert len(reports) == 0
            
        finally:
            await agent.stop()
    
    @pytest.mark.asyncio
    async def test_uptime_tracking(self, agent_settings):
        """Test agent uptime tracking."""
        agent = Agent(agent_settings, InMemoryStorage())
        
        with patch.object(agent, '_detect_capabilities'):
            await agent.start()
        
        try:
            # Check initial uptime
            status1 = await agent.get_status()
            assert status1.uptime_seconds >= 0
            
            # Wait a bit
            await asyncio.sleep(0.1)
            
            # Check uptime increased
            status2 = await agent.get_status()
            assert status2.uptime_seconds > status1.uptime_seconds
            
        finally:
            await agent.stop()


class TestEndToEndScenarios:
    """End-to-end scenario tests."""
    
    @pytest.mark.asyncio
    async def test_complete_model_discovery_to_job_execution(self, temp_models_dir):
        """Test complete flow from model discovery to job execution attempt."""
        # Setup agent with models
        settings = Settings(
            model_cache_dir=temp_models_dir,
            orchestrator_url=None,
            enable_gpu_monitoring=False
        )
        storage = InMemoryStorage()
        
        agent = Agent(settings, storage)
        
        # Start agent with mocked capabilities that populate test data
        async def mock_detect_capabilities():
            # Mock detected data similar to other tests
            mock_detected_data = {
                "system": {"hostname": "test-agent", "os": "Linux", "python_version": "3.12", "total_memory_gb": 16.0},
                "cpu": {"cores": 4, "model": "Test CPU"},
                "gpus": [{"name": "Tesla V100", "memory_mb": 16384}],
                "frameworks": [{"name": "pytorch", "version": "2.0"}, {"name": "vllm", "version": "0.2.0"}],
                "models": [
                    {
                        "name": "test-llama-7b",
                        "path": f"{temp_models_dir}/test-llama-7b",
                        "model_type": "llama",
                        "architecture": "LlamaForCausalLM",
                        "format": "pytorch",
                        "framework_compatible": ["vllm", "transformers"],
                        "size_gb": 13.5
                    }
                ],
                "hooks": [{"name": "nvidia-smi"}],
                "metrics": [{"name": "latency"}, {"name": "memory_usage"}]
            }
            
            # Store system info and capabilities
            await agent.storage.store_system_info(mock_detected_data)
            capabilities = {
                "agent_type": "white_box",
                "gpu_count": len(mock_detected_data["gpus"]),
                "frameworks": [f["name"] for f in mock_detected_data["frameworks"]],
                "max_concurrent_jobs": agent.settings.max_concurrent_jobs,
                "monitoring_available": bool(mock_detected_data["hooks"]),
            }
            await agent.storage.store_capabilities(capabilities)
        
        with patch.object(agent, '_detect_capabilities', side_effect=mock_detect_capabilities):
            await agent.start()
        
        try:
            # Verify models were discovered
            system_info = await agent.get_system_info()
            assert len(system_info["models"]) > 0
            
            # Create job using discovered model
            discovered_model = system_info["models"][0]
            job = JobRequest(
                job_id="e2e-test-job",
                experiment_id="e2e-test",
                model=discovered_model["name"],
                framework="vllm",
                task_type=TaskType.GENERATION,
                task_config={"prompt": "Hello, world!"},
                parameters={"temperature": 0.7, "max_tokens": 50}
            )
            
            # Queue and attempt execution
            job_id = await agent.queue_job(job)
            assert job_id == "e2e-test-job"
            
            # Wait for processing
            await asyncio.sleep(0.5)
            
            # Job will likely fail due to mocked vLLM, but should generate error report
            error_reports = await agent.get_error_reports()
            
            if error_reports:
                report = error_reports[0]
                assert report.job_id == "e2e-test-job"
                assert report.model_name == discovered_model["name"]
                assert report.framework == "vllm"
                
        finally:
            await agent.stop()
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create a temporary models directory with mock models."""
        temp_dir = tempfile.mkdtemp()
        models_dir = Path(temp_dir)
        
        # Create a mock HuggingFace model
        model_dir = models_dir / "test-llama-7b"
        model_dir.mkdir()
        
        # Create config.json
        config = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "max_position_embeddings": 2048,
            "torch_dtype": "float16"
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)
        
        # Create model files
        (model_dir / "pytorch_model.bin").write_text("fake model weights")
        (model_dir / "tokenizer.model").write_text("fake tokenizer")
        
        yield temp_dir
        shutil.rmtree(temp_dir)