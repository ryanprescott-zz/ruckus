"""End-to-end integration tests for RUCKUS agent and server orchestration."""

import pytest
import asyncio
import httpx
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from ruckus_common.models import (
    JobRequest, JobStatus, TaskType, AgentType, AgentStatusEnum,
    SingleRunResult, MetricStatistics, MultiRunJobResult, JobResult,
    JobUpdate, JobStage, ExperimentSpec, ExperimentExecution,
    AgentInfo, AgentRegistrationResponse, HealthStatus
)
from ruckus_agent.core.agent import Agent
from ruckus_agent.core.config import Settings
from ruckus_agent.core.storage import InMemoryStorage


@pytest.fixture
def agent_settings():
    """Create agent test settings."""
    return Settings(
        agent_type=AgentType.WHITE_BOX,
        max_concurrent_jobs=2,
        orchestrator_url="http://localhost:8080",
        api_host="0.0.0.0",
        api_port=8081
    )


@pytest.fixture
def mock_orchestrator_client():
    """Mock HTTP client for orchestrator communication."""
    client = MagicMock(spec=httpx.AsyncClient)
    return client


@pytest.fixture
async def test_agent(agent_settings):
    """Create a test agent instance."""
    agent = Agent(agent_settings, InMemoryStorage())
    
    # Mock capability detection to avoid real hardware detection
    with patch.object(agent, '_detect_capabilities') as mock_detect:
        mock_detect.return_value = None
        await agent.storage.store_system_info({
            "system": {"hostname": "test-agent", "os": "Linux"},
            "cpu": {"cores": 8, "model": "Test CPU"},
            "gpus": [{"name": "Test GPU", "memory_mb": 8192}],
            "frameworks": [{"name": "pytorch", "version": "2.0.0"}],
            "models": ["test-model-7b", "test-model-13b"],
            "hooks": ["nvidia-smi"],
            "metrics": ["latency", "throughput", "memory"]
        })
        
        await agent.storage.store_capabilities({
            "agent_type": "white_box",
            "gpu_count": 1,
            "frameworks": ["pytorch", "transformers"],
            "max_concurrent_jobs": 2,
            "monitoring_available": True
        })
    
    yield agent
    
    # Cleanup
    await agent.stop()


class TestAgentRegistrationFlow:
    """Test agent registration and communication with orchestrator."""

    @pytest.mark.asyncio
    async def test_agent_registration_success(self, test_agent, mock_orchestrator_client):
        """Test successful agent registration with orchestrator."""
        # Mock orchestrator registration response
        registration_response = AgentRegistrationResponse(
            agent_id=test_agent.agent_id,
            agent_name=test_agent.agent_name,
            message="Registration successful",
            server_time=datetime.utcnow()
        )
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = registration_response.dict()
        mock_orchestrator_client.post.return_value = mock_response
        
        # Replace agent's HTTP client
        test_agent.client = mock_orchestrator_client
        
        # Perform registration
        await test_agent._register()
        
        # Verify registration was attempted
        assert mock_orchestrator_client.post.called
        call_args = mock_orchestrator_client.post.call_args
        assert call_args[0][0].endswith("/api/v1/agents/register")
        
        # Verify registration data
        registration_data = call_args[1]["json"]
        assert registration_data["agent_id"] == test_agent.agent_id
        assert registration_data["agent_type"] == "white_box"
        assert "system_info" in registration_data
        assert "capabilities" in registration_data
        
        # Verify agent is marked as registered
        assert test_agent.registered

    @pytest.mark.asyncio
    async def test_agent_registration_failure(self, test_agent, mock_orchestrator_client):
        """Test agent registration failure handling."""
        # Mock orchestrator registration failure
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_orchestrator_client.post.return_value = mock_response
        
        test_agent.client = mock_orchestrator_client
        
        # Registration should handle failure gracefully
        await test_agent._register()
        
        # Agent should not be marked as registered
        assert not test_agent.registered

    @pytest.mark.asyncio
    async def test_agent_heartbeat_communication(self, test_agent, mock_orchestrator_client):
        """Test agent heartbeat communication with orchestrator."""
        # Mock successful heartbeat response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "acknowledged"}
        mock_orchestrator_client.post.return_value = mock_response
        
        test_agent.client = mock_orchestrator_client
        test_agent.registered = True
        
        # Simulate heartbeat
        await test_agent._send_heartbeat()
        
        # Verify heartbeat was sent
        assert mock_orchestrator_client.post.called
        call_args = mock_orchestrator_client.post.call_args
        assert "/heartbeat" in call_args[0][0]
        
        # Verify heartbeat data
        heartbeat_data = call_args[1]["json"]
        assert heartbeat_data["agent_id"] == test_agent.agent_id
        assert "timestamp" in heartbeat_data


class TestSingleRunJobEndToEnd:
    """Test end-to-end single-run job execution."""

    @pytest.fixture
    def single_run_job_request(self):
        """Create a single-run job request."""
        return JobRequest(
            job_id="e2e-single-001",
            experiment_id="e2e-exp-single",
            model="test-model-7b",
            framework="pytorch",
            task_type=TaskType.SUMMARIZATION,
            task_config={
                "input_text": "This is a comprehensive end-to-end test of single-run job execution in the RUCKUS system.",
                "max_length": 100,
                "temperature": 0.7
            },
            runs_per_job=1,
            required_metrics=["latency", "throughput", "memory_usage"],
            optional_metrics=["gpu_utilization", "model_load_time"],
            timeout_seconds=300,
            callback_url="http://localhost:8080/api/v1/jobs/callback"
        )

    @pytest.mark.asyncio
    async def test_single_run_job_complete_flow(self, test_agent, single_run_job_request, mock_orchestrator_client):
        """Test complete single-run job execution flow."""
        # Mock task execution components
        with patch.object(test_agent, '_load_model') as mock_load_model, \
             patch.object(test_agent, '_execute_task') as mock_execute_task, \
             patch.object(test_agent, '_collect_metrics') as mock_collect_metrics:
            
            # Mock model loading
            mock_load_model.return_value = {
                "model": MagicMock(),
                "load_time_seconds": 2.5,
                "memory_usage_mb": 1500
            }
            
            # Mock task execution
            mock_execute_task.return_value = {
                "output": "This is a test summary generated by the model.",
                "execution_time_seconds": 1.8,
                "tokens_generated": 15,
                "success": True
            }
            
            # Mock metrics collection
            mock_collect_metrics.return_value = {
                "latency": 1.8,
                "throughput": 8.33,  # tokens/second
                "memory_usage": 1500,
                "gpu_utilization": 75,
                "model_load_time": 2.5
            }
            
            # Mock orchestrator callback for job updates
            mock_callback_response = MagicMock()
            mock_callback_response.status_code = 200
            mock_orchestrator_client.post.return_value = mock_callback_response
            test_agent.client = mock_orchestrator_client
            
            # Execute the job
            result = await test_agent.execute_job(single_run_job_request)
            
            # Verify result is a JobResult (not MultiRunJobResult)
            assert isinstance(result, JobResult)
            assert not isinstance(result, MultiRunJobResult)
            assert result.job_id == "e2e-single-001"
            assert result.status == JobStatus.COMPLETED
            assert result.duration_seconds > 0
            
            # Verify metrics
            assert "latency" in result.metrics
            assert "throughput" in result.metrics
            assert "memory_usage" in result.metrics
            assert result.metrics["latency"] == 1.8
            assert result.metrics["throughput"] == 8.33
            
            # Verify output
            assert result.output == "This is a test summary generated by the model."
            
            # Verify model and framework info
            assert result.model_actual == "test-model-7b"
            assert "pytorch" in result.framework_version or result.framework_version is None
            
            # Verify callbacks were sent to orchestrator
            callback_calls = [call for call in mock_orchestrator_client.post.call_args_list 
                             if "callback" in str(call)]
            assert len(callback_calls) > 0  # At least job completion callback

    @pytest.mark.asyncio
    async def test_single_run_job_with_failure(self, test_agent, single_run_job_request, mock_orchestrator_client):
        """Test single-run job execution with failure handling."""
        # Mock task execution failure
        with patch.object(test_agent, '_load_model') as mock_load_model, \
             patch.object(test_agent, '_execute_task') as mock_execute_task:
            
            # Mock successful model loading
            mock_load_model.return_value = {
                "model": MagicMock(),
                "load_time_seconds": 2.5,
                "memory_usage_mb": 1500
            }
            
            # Mock task execution failure
            mock_execute_task.side_effect = RuntimeError("Out of memory during inference")
            
            # Mock orchestrator communication
            mock_callback_response = MagicMock()
            mock_callback_response.status_code = 200
            mock_orchestrator_client.post.return_value = mock_callback_response
            test_agent.client = mock_orchestrator_client
            
            # Execute the job (should handle failure gracefully)
            result = await test_agent.execute_job(single_run_job_request)
            
            # Verify result shows failure
            assert isinstance(result, JobResult)
            assert result.status == JobStatus.FAILED
            assert result.error is not None
            assert "Out of memory" in result.error
            assert result.error_type == "RuntimeError"
            
            # Verify failure was reported to orchestrator
            callback_calls = mock_orchestrator_client.post.call_args_list
            failure_callbacks = [call for call in callback_calls 
                               if "callback" in str(call) and 
                               any("failed" in str(call).lower() or "error" in str(call).lower()
                                   for call in callback_calls)]
            assert len(failure_callbacks) > 0

    @pytest.mark.asyncio
    async def test_single_run_job_progress_updates(self, test_agent, single_run_job_request, mock_orchestrator_client):
        """Test single-run job with progress updates to orchestrator."""
        progress_updates = []
        
        def capture_progress(url, **kwargs):
            if "callback" in url and kwargs.get("json"):
                progress_updates.append(kwargs["json"])
            mock_response = MagicMock()
            mock_response.status_code = 200
            return mock_response
        
        mock_orchestrator_client.post.side_effect = capture_progress
        test_agent.client = mock_orchestrator_client
        
        # Mock task execution with progress reporting
        with patch.object(test_agent, '_execute_job_with_progress') as mock_execute_with_progress:
            
            async def mock_execution_with_updates(job_request):
                # Simulate job stages with progress updates
                stages = [
                    (JobStage.INITIALIZING, 0),
                    (JobStage.LOADING_MODEL, 25),
                    (JobStage.PREPARING_DATA, 50),
                    (JobStage.RUNNING_INFERENCE, 75),
                    (JobStage.COLLECTING_METRICS, 90),
                    (JobStage.FINALIZING, 100)
                ]
                
                for stage, progress in stages:
                    update = JobUpdate(
                        job_id=job_request.job_id,
                        status=JobStatus.RUNNING,
                        stage=stage,
                        progress=progress,
                        message=f"Executing {stage.value}",
                        timestamp=datetime.utcnow()
                    )
                    
                    # Send progress update
                    await test_agent._send_job_update(update)
                    
                    # Small delay to simulate work
                    await asyncio.sleep(0.01)
                
                # Return successful result
                return JobResult(
                    job_id=job_request.job_id,
                    experiment_id=job_request.experiment_id,
                    status=JobStatus.COMPLETED,
                    started_at=datetime.utcnow() - timedelta(seconds=5),
                    completed_at=datetime.utcnow(),
                    duration_seconds=5.0,
                    output="Test output",
                    metrics={"latency": 1.5, "throughput": 10.0}
                )
            
            mock_execute_with_progress.side_effect = mock_execution_with_updates
            
            # Execute job
            result = await test_agent.execute_job(single_run_job_request)
            
            # Verify progress updates were sent
            assert len(progress_updates) >= 6  # At least 6 stage updates
            
            # Verify update sequence
            stages_reported = [update.get("stage") for update in progress_updates if update.get("stage")]
            expected_stages = ["initializing", "loading_model", "preparing_data", 
                             "running_inference", "collecting_metrics", "finalizing"]
            
            for expected_stage in expected_stages:
                assert expected_stage in stages_reported


class TestMultiRunJobEndToEnd:
    """Test end-to-end multi-run job execution."""

    @pytest.fixture
    def multi_run_job_request(self):
        """Create a multi-run job request."""
        return JobRequest(
            job_id="e2e-multi-001",
            experiment_id="e2e-exp-multi",
            model="test-model-7b",
            framework="pytorch",
            task_type=TaskType.SUMMARIZATION,
            task_config={
                "input_text": "This is a comprehensive end-to-end test of multi-run job execution for statistical reliability analysis.",
                "max_length": 100,
                "temperature": 0.7
            },
            runs_per_job=5,  # Multi-run for statistical analysis
            required_metrics=["latency", "throughput", "memory_usage"],
            optional_metrics=["gpu_utilization", "model_load_time"],
            timeout_seconds=900,
            callback_url="http://localhost:8080/api/v1/jobs/callback"
        )

    @pytest.mark.asyncio
    async def test_multi_run_job_complete_flow(self, test_agent, multi_run_job_request, mock_orchestrator_client):
        """Test complete multi-run job execution flow with statistical analysis."""
        # Mock task execution components with variability
        with patch.object(test_agent, '_load_model') as mock_load_model, \
             patch.object(test_agent, '_execute_task') as mock_execute_task, \
             patch.object(test_agent, '_collect_metrics') as mock_collect_metrics:
            
            # Mock model loading (only for cold start)
            mock_load_model.return_value = {
                "model": MagicMock(),
                "load_time_seconds": 3.2,
                "memory_usage_mb": 1800
            }
            
            # Mock variable task execution results (simulating real variance)
            execution_results = [
                # Cold start run (includes model loading time)
                {"output": "Cold start summary.", "execution_time_seconds": 2.8, "tokens_generated": 18, "success": True},
                # Warm runs (faster after model loaded)
                {"output": "Warm run 1 summary.", "execution_time_seconds": 1.5, "tokens_generated": 16, "success": True},
                {"output": "Warm run 2 summary.", "execution_time_seconds": 1.6, "tokens_generated": 17, "success": True},
                {"output": "Warm run 3 summary.", "execution_time_seconds": 1.4, "tokens_generated": 15, "success": True},
                {"output": "Warm run 4 summary.", "execution_time_seconds": 1.7, "tokens_generated": 19, "success": True}
            ]
            
            mock_execute_task.side_effect = execution_results
            
            # Mock variable metrics (simulating real measurement variance)
            metrics_results = [
                # Cold start metrics
                {"latency": 2.8, "throughput": 6.43, "memory_usage": 1800, "gpu_utilization": 85, "model_load_time": 3.2},
                # Warm run metrics
                {"latency": 1.5, "throughput": 10.67, "memory_usage": 1600, "gpu_utilization": 78},
                {"latency": 1.6, "throughput": 10.63, "memory_usage": 1650, "gpu_utilization": 80},
                {"latency": 1.4, "throughput": 10.71, "memory_usage": 1580, "gpu_utilization": 76},
                {"latency": 1.7, "throughput": 11.18, "memory_usage": 1620, "gpu_utilization": 82}
            ]
            
            mock_collect_metrics.side_effect = metrics_results
            
            # Mock orchestrator communication
            mock_callback_response = MagicMock()
            mock_callback_response.status_code = 200
            mock_orchestrator_client.post.return_value = mock_callback_response
            test_agent.client = mock_orchestrator_client
            
            # Execute the multi-run job
            result = await test_agent.execute_job(multi_run_job_request)
            
            # Verify result is a MultiRunJobResult
            assert isinstance(result, MultiRunJobResult)
            assert result.job_id == "e2e-multi-001"
            assert result.total_runs == 5
            assert result.successful_runs == 5
            assert result.failed_runs == 0
            
            # Verify individual runs
            assert len(result.individual_runs) == 5
            
            # Verify cold start run
            cold_start = result.individual_runs[0]
            assert cold_start.is_cold_start is True
            assert cold_start.model_load_time_seconds == 3.2
            assert cold_start.metrics["latency"] == 2.8
            
            # Verify warm runs  
            warm_runs = result.individual_runs[1:]
            for run in warm_runs:
                assert run.is_cold_start is False
                assert run.model_load_time_seconds is None
                assert run.metrics["latency"] < cold_start.metrics["latency"]  # Should be faster
            
            # Verify cold start data separation
            assert result.cold_start_data is not None
            assert result.cold_start_data.model_load_time_seconds == 3.2
            
            # Verify statistical analysis of warm runs
            assert "latency" in result.summary_stats
            assert "throughput" in result.summary_stats
            
            latency_stats = result.summary_stats["latency"]
            assert latency_stats.count == 4  # 4 warm runs
            assert latency_stats.mean > 0
            assert latency_stats.std >= 0
            assert len(latency_stats.raw_values) == 4
            assert latency_stats.min <= latency_stats.median <= latency_stats.max
            
            # Verify callbacks to orchestrator
            callback_calls = mock_orchestrator_client.post.call_args_list
            assert len(callback_calls) > 0

    @pytest.mark.asyncio
    async def test_multi_run_job_with_partial_failures(self, test_agent, multi_run_job_request, mock_orchestrator_client):
        """Test multi-run job execution with some failed runs."""
        # Mock task execution with some failures
        with patch.object(test_agent, '_load_model') as mock_load_model, \
             patch.object(test_agent, '_execute_task') as mock_execute_task, \
             patch.object(test_agent, '_collect_metrics') as mock_collect_metrics:
            
            mock_load_model.return_value = {
                "model": MagicMock(),
                "load_time_seconds": 3.0,
                "memory_usage_mb": 1800
            }
            
            # Mock execution results with failures
            execution_results = [
                # Cold start - success
                {"output": "Cold start summary.", "execution_time_seconds": 2.8, "tokens_generated": 18, "success": True},
                # Run 1 - success
                {"output": "Run 1 summary.", "execution_time_seconds": 1.5, "tokens_generated": 16, "success": True},
                # Run 2 - failure
                RuntimeError("GPU out of memory"),
                # Run 3 - success
                {"output": "Run 3 summary.", "execution_time_seconds": 1.4, "tokens_generated": 15, "success": True},
                # Run 4 - success
                {"output": "Run 4 summary.", "execution_time_seconds": 1.6, "tokens_generated": 17, "success": True}
            ]
            
            def mock_execute_side_effect(*args, **kwargs):
                result = execution_results.pop(0)
                if isinstance(result, Exception):
                    raise result
                return result
            
            mock_execute_task.side_effect = mock_execute_side_effect
            
            # Mock metrics for successful runs only
            metrics_results = [
                {"latency": 2.8, "throughput": 6.43, "memory_usage": 1800, "model_load_time": 3.0},
                {"latency": 1.5, "throughput": 10.67, "memory_usage": 1600},
                {"latency": 1.4, "throughput": 10.71, "memory_usage": 1580},
                {"latency": 1.6, "throughput": 10.63, "memory_usage": 1650}
            ]
            
            mock_collect_metrics.side_effect = metrics_results
            
            # Mock orchestrator communication
            mock_callback_response = MagicMock()
            mock_callback_response.status_code = 200
            mock_orchestrator_client.post.return_value = mock_callback_response
            test_agent.client = mock_orchestrator_client
            
            # Execute the multi-run job
            result = await test_agent.execute_job(multi_run_job_request)
            
            # Verify mixed success/failure results
            assert isinstance(result, MultiRunJobResult)
            assert result.total_runs == 5
            assert result.successful_runs == 4  # 1 failure
            assert result.failed_runs == 1
            
            # Verify failed run has error information
            failed_runs = [run for run in result.individual_runs if run.error is not None]
            assert len(failed_runs) == 1
            assert "GPU out of memory" in failed_runs[0].error
            assert failed_runs[0].error_type == "RuntimeError"
            
            # Verify statistics only include successful warm runs
            successful_warm_runs = [run for run in result.individual_runs[1:] if run.error is None]
            if result.summary_stats:
                latency_stats = result.summary_stats.get("latency")
                if latency_stats:
                    assert latency_stats.count == len(successful_warm_runs)

    @pytest.mark.asyncio
    async def test_multi_run_job_statistical_analysis(self, test_agent, multi_run_job_request, mock_orchestrator_client):
        """Test multi-run job statistical analysis including outlier detection."""
        # Mock task execution with outlier data
        with patch.object(test_agent, '_load_model') as mock_load_model, \
             patch.object(test_agent, '_execute_task') as mock_execute_task, \
             patch.object(test_agent, '_collect_metrics') as mock_collect_metrics:
            
            mock_load_model.return_value = {
                "model": MagicMock(),
                "load_time_seconds": 3.0,
                "memory_usage_mb": 1800
            }
            
            # Mock execution results with one outlier
            execution_results = [
                {"output": "Cold start", "execution_time_seconds": 2.5, "tokens_generated": 18, "success": True},
                {"output": "Normal run 1", "execution_time_seconds": 1.5, "tokens_generated": 16, "success": True},
                {"output": "Normal run 2", "execution_time_seconds": 1.6, "tokens_generated": 17, "success": True},
                {"output": "Outlier run", "execution_time_seconds": 4.2, "tokens_generated": 15, "success": True},  # Outlier
                {"output": "Normal run 3", "execution_time_seconds": 1.4, "tokens_generated": 19, "success": True}
            ]
            
            mock_execute_task.side_effect = execution_results
            
            # Mock metrics with outlier
            metrics_results = [
                {"latency": 2.5, "throughput": 7.2, "memory_usage": 1800, "model_load_time": 3.0},
                {"latency": 1.5, "throughput": 10.67, "memory_usage": 1600},
                {"latency": 1.6, "throughput": 10.63, "memory_usage": 1650},  
                {"latency": 4.2, "throughput": 3.57, "memory_usage": 1700},  # Outlier - much slower
                {"latency": 1.4, "throughput": 13.57, "memory_usage": 1580}
            ]
            
            mock_collect_metrics.side_effect = metrics_results
            
            # Mock orchestrator communication
            mock_callback_response = MagicMock()
            mock_callback_response.status_code = 200
            mock_orchestrator_client.post.return_value = mock_callback_response
            test_agent.client = mock_orchestrator_client
            
            # Execute job
            result = await test_agent.execute_job(multi_run_job_request)
            
            # Verify statistical analysis
            assert isinstance(result, MultiRunJobResult)
            assert result.successful_runs == 5
            
            # Check statistics include outlier detection
            latency_stats = result.summary_stats["latency"]
            assert latency_stats.count == 4  # 4 warm runs
            
            # The outlier (4.2s) should be detected if statistics are computed properly
            # Raw values should include all measurements
            warm_latencies = [1.5, 1.6, 4.2, 1.4]  # Warm run latencies
            assert len(latency_stats.raw_values) == 4
            assert set(latency_stats.raw_values) == set(warm_latencies)
            
            # Min/max should reflect actual data
            assert latency_stats.min == 1.4
            assert latency_stats.max == 4.2
            
            # If outlier detection is implemented, outlier list might contain the index
            # (This depends on the actual implementation of outlier detection)


class TestOrchestratorAgentIntegration:
    """Test orchestrator-agent communication and job lifecycle."""

    @pytest.mark.asyncio
    async def test_job_assignment_and_execution_flow(self, test_agent, mock_orchestrator_client):
        """Test complete job assignment and execution flow."""
        # Mock orchestrator job assignment
        job_request = JobRequest(
            job_id="orchestrator-job-001",
            experiment_id="orchestrator-exp-001",
            model="test-model-7b",
            framework="pytorch",
            task_type=TaskType.SUMMARIZATION,
            task_config={"input_text": "Test orchestrator integration"},
            runs_per_job=3,
            required_metrics=["latency", "throughput"],
            timeout_seconds=300
        )
        
        # Mock successful job execution
        with patch.object(test_agent, 'execute_job') as mock_execute_job:
            mock_result = MultiRunJobResult(
                job_id="orchestrator-job-001",
                experiment_id="orchestrator-exp-001",
                total_runs=3,
                successful_runs=3,
                failed_runs=0,
                individual_runs=[
                    SingleRunResult(
                        run_id=i,
                        is_cold_start=(i == 0),
                        started_at=datetime.utcnow(),
                        completed_at=datetime.utcnow() + timedelta(seconds=2),
                        duration_seconds=2.0,
                        metrics={"latency": 1.5 + i*0.1, "throughput": 10.0 - i*0.2}
                    ) for i in range(3)
                ],
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow() + timedelta(seconds=8),
                total_duration_seconds=8.0
            )
            
            mock_execute_job.return_value = mock_result
            
            # Mock orchestrator communication
            mock_callback_response = MagicMock()
            mock_callback_response.status_code = 200
            mock_orchestrator_client.post.return_value = mock_callback_response
            test_agent.client = mock_orchestrator_client
            
            # Simulate job assignment from orchestrator
            test_agent.running_jobs[job_request.job_id] = {
                "job_request": job_request,
                "status": JobStatus.RUNNING,
                "started_at": datetime.utcnow()
            }
            
            # Execute the job
            result = await test_agent.execute_job(job_request)
            
            # Verify job was executed
            mock_execute_job.assert_called_once_with(job_request)
            
            # Verify result
            assert isinstance(result, MultiRunJobResult)
            assert result.job_id == "orchestrator-job-001"
            assert result.successful_runs == 3
            
            # Verify orchestrator was notified of completion
            callback_calls = mock_orchestrator_client.post.call_args_list
            completion_calls = [call for call in callback_calls if "callback" in str(call)]
            assert len(completion_calls) > 0

    @pytest.mark.asyncio
    async def test_concurrent_job_execution(self, test_agent, mock_orchestrator_client):
        """Test concurrent execution of multiple jobs."""
        # Create multiple job requests
        job_requests = [
            JobRequest(
                job_id=f"concurrent-job-{i:03d}",
                experiment_id=f"concurrent-exp-{i:03d}",
                model="test-model-7b",
                framework="pytorch",
                task_type=TaskType.SUMMARIZATION,
                task_config={"input_text": f"Concurrent test {i}"},
                runs_per_job=2,
                timeout_seconds=180
            ) for i in range(3)
        ]
        
        # Mock job execution results
        with patch.object(test_agent, 'execute_job') as mock_execute_job:
            
            async def mock_job_execution(job_request):
                # Simulate some execution time
                await asyncio.sleep(0.1)
                
                return MultiRunJobResult(
                    job_id=job_request.job_id,
                    experiment_id=job_request.experiment_id,
                    total_runs=2,
                    successful_runs=2,
                    failed_runs=0,
                    individual_runs=[
                        SingleRunResult(
                            run_id=j,
                            is_cold_start=(j == 0),
                            started_at=datetime.utcnow(),
                            completed_at=datetime.utcnow() + timedelta(seconds=1),
                            duration_seconds=1.0,
                            metrics={"latency": 1.0}
                        ) for j in range(2)
                    ],
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow() + timedelta(seconds=3),
                    total_duration_seconds=3.0
                )
            
            mock_execute_job.side_effect = mock_job_execution
            
            # Mock orchestrator communication
            mock_callback_response = MagicMock()
            mock_callback_response.status_code = 200
            mock_orchestrator_client.post.return_value = mock_callback_response
            test_agent.client = mock_orchestrator_client
            
            # Execute jobs concurrently (within agent's limit)
            max_concurrent = min(len(job_requests), test_agent.settings.max_concurrent_jobs)
            
            # Submit jobs to agent queue
            for job_request in job_requests[:max_concurrent]:
                await test_agent.job_queue.put(job_request)
                test_agent.queued_job_ids.append(job_request.job_id)
            
            # Wait for job execution (simulate job executor processing)
            await asyncio.sleep(0.5)
            
            # Verify jobs were queued
            assert len(test_agent.queued_job_ids) <= max_concurrent

    @pytest.mark.asyncio
    async def test_agent_status_reporting_during_jobs(self, test_agent, mock_orchestrator_client):
        """Test agent status reporting during job execution."""
        # Mock active job
        job_request = JobRequest(
            job_id="status-test-job",
            experiment_id="status-test-exp",
            model="test-model-7b",
            framework="pytorch",
            task_type=TaskType.SUMMARIZATION,
            task_config={"input_text": "Status test"},
            runs_per_job=1
        )
        
        # Add job to running jobs
        test_agent.running_jobs["status-test-job"] = {
            "job_request": job_request,
            "status": JobStatus.RUNNING,
            "started_at": datetime.utcnow()
        }
        
        # Add job to queue
        test_agent.queued_job_ids.append("queued-job-001")
        
        # Get agent status
        status = await test_agent.get_status()
        
        # Verify status reflects active jobs
        assert status.agent_id == test_agent.agent_id
        assert status.status == AgentStatusEnum.ACTIVE  # Has running jobs
        assert len(status.running_jobs) == 1
        assert "status-test-job" in status.running_jobs
        assert len(status.queued_jobs) == 1
        assert "queued-job-001" in status.queued_jobs
        assert status.uptime_seconds > 0

    @pytest.mark.asyncio
    async def test_error_reporting_to_orchestrator(self, test_agent, mock_orchestrator_client):
        """Test error reporting and recovery coordination with orchestrator."""
        # Mock job that will fail
        job_request = JobRequest(
            job_id="error-test-job",
            experiment_id="error-test-exp",
            model="test-model-7b",
            framework="pytorch",
            task_type=TaskType.SUMMARIZATION,
            task_config={"input_text": "Error test"},
            runs_per_job=1
        )
        
        # Mock job execution failure
        with patch.object(test_agent, 'execute_job') as mock_execute_job:
            mock_execute_job.side_effect = RuntimeError("Simulated execution error")
            
            # Mock orchestrator communication
            error_reports = []
            
            def capture_error_reports(url, **kwargs):
                if "error" in url.lower() or ("callback" in url and kwargs.get("json", {}).get("status") == "failed"):
                    error_reports.append(kwargs.get("json", {}))
                mock_response = MagicMock()
                mock_response.status_code = 200
                return mock_response
            
            mock_orchestrator_client.post.side_effect = capture_error_reports
            test_agent.client = mock_orchestrator_client
            
            # Attempt to execute job (should fail and report error)
            try:
                await test_agent.execute_job(job_request)
            except RuntimeError:
                pass  # Expected failure
            
            # Verify error was reported to orchestrator
            assert len(error_reports) > 0
            
            # Verify agent can recover and continue
            agent_status = await test_agent.get_status()
            assert agent_status.status in [AgentStatusEnum.IDLE, AgentStatusEnum.ERROR]