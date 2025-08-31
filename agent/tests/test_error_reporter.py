"""Tests for error reporting and metrics capture functionality."""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock
from typing import List

from ruckus_agent.utils.error_reporter import ErrorReporter, SystemMetricsCollector
from ruckus_agent.core.models import SystemMetricsSnapshot, JobErrorReport, JobFailureContext


class TestSystemMetricsCollector:
    """Test system metrics collection functionality."""
    
    @pytest.mark.asyncio
    async def test_capture_snapshot_basic(self):
        """Test basic snapshot capture without hardware dependencies."""
        snapshot = await SystemMetricsCollector.capture_snapshot()
        
        # Basic structure should exist
        assert isinstance(snapshot, SystemMetricsSnapshot)
        assert isinstance(snapshot.timestamp, datetime)
        
        # Lists should be initialized (may be empty on systems without GPU)
        assert isinstance(snapshot.gpu_memory_used_mb, list)
        assert isinstance(snapshot.gpu_memory_total_mb, list)
        assert isinstance(snapshot.gpu_utilization_percent, list)
        assert isinstance(snapshot.gpu_temperature_c, list)
        assert isinstance(snapshot.gpu_power_draw_w, list)
    
    @pytest.mark.asyncio
    @patch('asyncio.create_subprocess_exec')
    async def test_gpu_metrics_with_nvidia_smi_success(self, mock_subprocess_exec):
        """Test GPU metrics capture with successful nvidia-smi."""
        # Create mock process
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (
            b"1024, 8192, 85.5, 65.0, 250.5\n2048, 8192, 90.0, 70.0, 275.0",
            b""
        )
        mock_subprocess_exec.return_value = mock_process
        
        snapshot = SystemMetricsSnapshot()
        await SystemMetricsCollector._capture_gpu_metrics(snapshot)
        
        assert len(snapshot.gpu_memory_used_mb) == 2
        assert snapshot.gpu_memory_used_mb == [1024, 2048]
        assert snapshot.gpu_memory_total_mb == [8192, 8192]
        assert snapshot.gpu_utilization_percent == [85.5, 90.0]
        assert snapshot.gpu_temperature_c == [65.0, 70.0]
        assert snapshot.gpu_power_draw_w == [250.5, 275.0]
    
    @pytest.mark.asyncio
    @patch('subprocess.run')
    async def test_gpu_metrics_with_nvidia_smi_failure(self, mock_subprocess):
        """Test GPU metrics capture when nvidia-smi fails."""
        # Mock nvidia-smi failure
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "nvidia-smi: command not found"
        
        snapshot = SystemMetricsSnapshot()
        await SystemMetricsCollector._capture_gpu_metrics(snapshot)
        
        # Should have empty lists when nvidia-smi fails
        assert snapshot.gpu_memory_used_mb == []
        assert snapshot.gpu_memory_total_mb == []
        assert snapshot.gpu_utilization_percent == []
        assert snapshot.gpu_temperature_c == []
        assert snapshot.gpu_power_draw_w == []
    
    @pytest.mark.asyncio
    @patch('subprocess.run')
    async def test_gpu_metrics_with_malformed_output(self, mock_subprocess):
        """Test GPU metrics capture with malformed nvidia-smi output."""
        # Mock malformed output
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "invalid, output, format\n"
        
        snapshot = SystemMetricsSnapshot()
        await SystemMetricsCollector._capture_gpu_metrics(snapshot)
        
        # Should handle malformed output gracefully
        assert snapshot.gpu_memory_used_mb == []
    
    @pytest.mark.asyncio
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @patch('psutil.disk_usage')
    async def test_system_metrics_capture(self, mock_disk, mock_cpu, mock_memory):
        """Test system-level metrics capture."""
        # Mock psutil responses
        mock_memory.return_value = MagicMock(
            total=8 * 1024**3,  # 8GB
            available=4 * 1024**3  # 4GB available
        )
        mock_cpu.return_value = 75.5
        mock_disk.return_value = MagicMock(used=100 * 1024**3)  # 100GB used
        
        snapshot = SystemMetricsSnapshot()
        await SystemMetricsCollector._capture_system_metrics(snapshot)
        
        assert snapshot.system_memory_total_gb == 8.0
        assert snapshot.system_memory_used_gb == 4.0  # total - available
        assert snapshot.cpu_utilization_percent == 75.5
        assert snapshot.disk_usage_gb == 100.0
    
    @pytest.mark.asyncio
    @patch('psutil.Process')
    async def test_process_metrics_capture(self, mock_process_class):
        """Test process-specific metrics capture."""
        # Mock current process
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=512 * 1024 * 1024)  # 512MB
        mock_process.cpu_percent.return_value = 25.0
        mock_process_class.return_value = mock_process
        
        snapshot = SystemMetricsSnapshot()
        await SystemMetricsCollector._capture_process_metrics(snapshot)
        
        assert snapshot.process_memory_mb == 512.0
        assert snapshot.process_cpu_percent == 25.0
    
    @pytest.mark.asyncio
    async def test_capture_snapshot_exception_handling(self):
        """Test that snapshot capture handles exceptions gracefully."""
        # Even if individual metrics fail, should return a snapshot
        with patch('psutil.virtual_memory', side_effect=Exception("psutil error")):
            snapshot = await SystemMetricsCollector.capture_snapshot()
            
            # Should still return a valid snapshot object
            assert isinstance(snapshot, SystemMetricsSnapshot)
            assert isinstance(snapshot.timestamp, datetime)


class TestErrorReporter:
    """Test error reporting functionality."""
    
    @pytest.fixture
    def error_reporter(self):
        """Create an error reporter for testing."""
        return ErrorReporter("test-agent-123")
    
    @pytest.mark.asyncio
    async def test_start_job_tracking(self, error_reporter):
        """Test starting job tracking."""
        job_id = "job-test-123"
        
        context = await error_reporter.start_job_tracking(job_id, "initializing")
        
        assert context.job_id == job_id
        assert context.stage == "initializing"
        assert isinstance(context.start_time, datetime)
        assert len(context.metrics_snapshots) == 1  # Initial snapshot
        assert "initializing" in context.stage_history
        assert job_id in error_reporter.failure_contexts
    
    @pytest.mark.asyncio
    async def test_update_job_stage(self, error_reporter):
        """Test updating job stage during tracking."""
        job_id = "job-test-123"
        
        # Start tracking
        await error_reporter.start_job_tracking(job_id, "initializing")
        
        # Update stage
        await error_reporter.update_job_stage(job_id, "model_loading")
        
        context = error_reporter.failure_contexts[job_id]
        assert context.stage == "model_loading"
        assert len(context.metrics_snapshots) == 2  # Initial + update
        assert "model_loading" in context.stage_history[1]
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_job_stage(self, error_reporter):
        """Test updating stage for non-tracked job."""
        # Should handle gracefully
        await error_reporter.update_job_stage("nonexistent-job", "test")
        # Should not raise an exception
    
    @pytest.mark.asyncio
    async def test_generate_error_report(self, error_reporter):
        """Test generating comprehensive error report."""
        job_id = "job-test-123"
        experiment_id = "exp-test-456"
        
        # Start tracking
        await error_reporter.start_job_tracking(job_id, "initializing")
        await error_reporter.update_job_stage(job_id, "model_loading")
        
        # Create test error
        test_error = RuntimeError("CUDA out of memory")
        start_time = datetime.now(timezone.utc)
        
        # Generate report
        with patch.object(error_reporter, '_capture_nvidia_smi_full', return_value="nvidia-smi output"):
            report = await error_reporter.generate_error_report(
                job_id=job_id,
                experiment_id=experiment_id,
                error=test_error,
                model_name="test-model",
                model_path="/models/test-model",
                framework="vllm",
                task_type="summarization",
                parameters={"temperature": 0.7},
                started_at=start_time,
                model_size_gb=7.5
            )
        
        # Verify report structure
        assert report.job_id == job_id
        assert report.experiment_id == experiment_id
        assert report.agent_id == "test-agent-123"
        assert report.error_message == "CUDA out of memory"
        assert report.model_name == "test-model"
        assert report.model_path == "/models/test-model"
        assert report.framework == "vllm"
        assert report.model_size_gb == 7.5
        assert report.cuda_out_of_memory is True
        assert "RuntimeError" in report.error_traceback
        assert report.nvidia_smi_output == "nvidia-smi output"
        
        # Check metrics snapshots
        assert isinstance(report.metrics_at_failure, SystemMetricsSnapshot)
        assert isinstance(report.metrics_before_failure, SystemMetricsSnapshot)
        
        # Check timing
        assert report.started_at == start_time
        assert report.failed_at >= start_time
        assert report.duration_before_failure_seconds >= 0
    
    @pytest.mark.asyncio
    async def test_cleanup_job_tracking(self, error_reporter):
        """Test cleaning up job tracking."""
        job_id = "job-test-123"
        
        # Start tracking
        await error_reporter.start_job_tracking(job_id)
        assert job_id in error_reporter.failure_contexts
        
        # Cleanup
        await error_reporter.cleanup_job_tracking(job_id)
        assert job_id not in error_reporter.failure_contexts
    
    def test_classify_error(self, error_reporter):
        """Test error classification logic."""
        snapshot = SystemMetricsSnapshot()
        
        # CUDA out of memory
        cuda_oom_error = RuntimeError("CUDA out of memory: tried to allocate 4.00 GiB")
        assert error_reporter._classify_error(cuda_oom_error, snapshot) == "cuda_out_of_memory"
        
        # Generic CUDA error
        cuda_error = RuntimeError("CUDA error: invalid device ordinal")
        assert error_reporter._classify_error(cuda_error, snapshot) == "cuda_error"
        
        # Memory error
        memory_error = MemoryError("Cannot allocate memory")
        assert error_reporter._classify_error(memory_error, snapshot) == "out_of_memory"
        
        # Timeout error
        timeout_error = asyncio.TimeoutError("Operation timed out")
        assert error_reporter._classify_error(timeout_error, snapshot) == "timeout"
        
        # Import error
        import_error = ImportError("No module named 'vllm'")
        assert error_reporter._classify_error(import_error, snapshot) == "dependency_error"
        
        # File error
        file_error = FileNotFoundError("Model file not found")
        assert error_reporter._classify_error(file_error, snapshot) == "file_error"
        
        # Unknown error
        unknown_error = ValueError("Something went wrong")
        assert error_reporter._classify_error(unknown_error, snapshot) == "unknown_error"
    
    @pytest.mark.asyncio
    async def test_capture_nvidia_smi_full_success(self, error_reporter):
        """Test capturing full nvidia-smi output."""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock successful nvidia-smi call
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"Full nvidia-smi output", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            output = await error_reporter._capture_nvidia_smi_full()
            
            assert output == "Full nvidia-smi output"
    
    @pytest.mark.asyncio
    async def test_capture_nvidia_smi_full_failure(self, error_reporter):
        """Test capturing nvidia-smi output when command fails."""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock failed nvidia-smi call
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"command not found")
            mock_process.returncode = 1
            mock_subprocess.return_value = mock_process
            
            output = await error_reporter._capture_nvidia_smi_full()
            
            assert "nvidia-smi failed" in output
            assert "command not found" in output
    
    @pytest.mark.asyncio
    async def test_capture_nvidia_smi_exception(self, error_reporter):
        """Test nvidia-smi capture with exception."""
        with patch('asyncio.create_subprocess_exec', side_effect=Exception("Process error")):
            output = await error_reporter._capture_nvidia_smi_full()
            
            assert "Failed to run nvidia-smi" in output
            assert "Process error" in output
    
    def test_estimate_available_vram(self, error_reporter):
        """Test VRAM estimation from metrics snapshot."""
        snapshot = SystemMetricsSnapshot(
            gpu_memory_total_mb=[8192, 16384],
            gpu_memory_used_mb=[4096, 12288]
        )
        
        available = error_reporter._estimate_available_vram(snapshot)
        
        # Should return the GPU with most available memory (16384 - 12288 = 4096)
        assert available == 4096
    
    def test_estimate_available_vram_empty(self, error_reporter):
        """Test VRAM estimation with no GPU data."""
        snapshot = SystemMetricsSnapshot()
        
        available = error_reporter._estimate_available_vram(snapshot)
        
        assert available is None
    
    @pytest.mark.asyncio
    async def test_generate_error_report_without_tracking(self, error_reporter):
        """Test generating error report for job without prior tracking."""
        job_id = "untracked-job"
        experiment_id = "exp-test"
        error = RuntimeError("Test error")
        start_time = datetime.now(timezone.utc)
        
        report = await error_reporter.generate_error_report(
            job_id=job_id,
            experiment_id=experiment_id,
            error=error,
            model_name="test-model",
            model_path="/models/test-model",
            framework="vllm",
            task_type="generation",
            parameters={},
            started_at=start_time
        )
        
        # Should still generate valid report
        assert report.job_id == job_id
        assert report.experiment_id == experiment_id
        assert report.metrics_at_failure is not None
        assert report.metrics_before_failure is None  # No prior tracking
    
    @pytest.mark.asyncio
    async def test_multiple_job_tracking(self, error_reporter):
        """Test tracking multiple jobs simultaneously."""
        job1 = "job-1"
        job2 = "job-2"
        
        # Start tracking both jobs
        await error_reporter.start_job_tracking(job1, "initializing")
        await error_reporter.start_job_tracking(job2, "model_loading")
        
        assert job1 in error_reporter.failure_contexts
        assert job2 in error_reporter.failure_contexts
        
        # Update stages independently
        await error_reporter.update_job_stage(job1, "inference")
        await error_reporter.update_job_stage(job2, "completing")
        
        assert error_reporter.failure_contexts[job1].stage == "inference"
        assert error_reporter.failure_contexts[job2].stage == "completing"
        
        # Cleanup one job
        await error_reporter.cleanup_job_tracking(job1)
        
        assert job1 not in error_reporter.failure_contexts
        assert job2 in error_reporter.failure_contexts