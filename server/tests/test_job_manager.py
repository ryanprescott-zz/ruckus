"""Tests for JobManager job functionality."""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

from ruckus_server.core.job_manager import JobManager
from ruckus_server.core.config import JobManagerSettings, AgentSettings, HttpClientSettings
from ruckus_server.core.models import JobInfo
from ruckus_server.core.storage.base import StorageBackend
from ruckus_server.api.v1.models import ExperimentResult
from ruckus_common.models import (
    JobStatus,
    JobStatusEnum,
    ExperimentSpec,
    RegisteredAgentInfo,
    AgentStatus,
    AgentStatusEnum,
    JobResult,
    AgentType
)


@pytest.fixture
def job_manager_settings():
    """Create JobManager settings for testing."""
    return JobManagerSettings(
        job_status_polling_interval=1.0,
        agent=AgentSettings(),
        http_client=HttpClientSettings()
    )


@pytest.fixture
def mock_storage():
    """Create a mock storage backend."""
    storage = Mock(spec=StorageBackend)
    
    # Configure common async methods
    storage.get_experiment = AsyncMock()
    storage.get_agent = AsyncMock()
    storage.list_registered_agent_info = AsyncMock()
    storage.set_running_job = AsyncMock()
    storage.add_queued_job = AsyncMock()
    storage.add_completed_job = AsyncMock()
    storage.add_failed_job = AsyncMock()
    storage.get_running_job = AsyncMock()
    storage.get_queued_jobs = AsyncMock()
    storage.get_completed_jobs = AsyncMock()
    storage.get_failed_jobs = AsyncMock()
    storage.clear_running_job = AsyncMock()
    storage.remove_queued_job = AsyncMock()
    storage.update_running_job = AsyncMock()
    storage.store_experiment_result = AsyncMock()
    storage.save_experiment_results = AsyncMock()
    
    return storage


@pytest.fixture
def sample_experiment():
    """Create a sample experiment spec."""
    from ruckus_common.models import (
        TaskSpec, TaskType, FrameworkSpec, FrameworkName, MetricsSpec,
        LLMGenerationParams, PromptTemplate, PromptMessage, PromptRole
    )
    
    prompt_template = PromptTemplate(
        messages=[
            PromptMessage(role=PromptRole.USER, content="Test prompt")
        ]
    )
    
    return ExperimentSpec(
        experiment_id="exp_123",
        name="Test Experiment",
        description="A test experiment",
        model="gpt-3.5-turbo",
        task=TaskSpec(
            name="test_task",
            type=TaskType.LLM_GENERATION,
            params=LLMGenerationParams(prompt_template=prompt_template)
        ),
        framework=FrameworkSpec(name=FrameworkName.TRANSFORMERS, params={}),
        metrics=MetricsSpec(metrics={"accuracy": {"type": "classification"}})
    )


@pytest.fixture
def sample_agent_info():
    """Create a sample registered agent info."""
    return RegisteredAgentInfo(
        agent_id="agent_123",
        agent_name="Test Agent",
        agent_type=AgentType.GRAY_BOX,
        agent_url="http://localhost:8001",
        registered_at=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_agent_status():
    """Create a sample agent status."""
    return AgentStatus(
        agent_id="agent_123",
        status=AgentStatusEnum.IDLE,
        uptime_seconds=3600,
        running_jobs=[]
    )


@pytest_asyncio.fixture
async def job_manager(job_manager_settings, mock_storage):
    """Create a JobManager instance for testing."""
    manager = JobManager(job_manager_settings, mock_storage)
    await manager.start()
    yield manager
    await manager.stop()


class TestJobManagerCreation:
    """Tests for job creation functionality."""
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_create_job_success(self, job_manager, mock_storage, sample_experiment, sample_agent_info, sample_agent_status):
        """Test successful job creation."""
        # Setup mocks
        mock_storage.get_experiment.return_value = sample_experiment
        mock_storage.get_agent.return_value = sample_agent_info
        
        # Mock agent utility methods
        with patch.object(job_manager.agent_utility, 'get_agent_status', return_value=sample_agent_status), \
             patch.object(job_manager.agent_utility, 'execute_experiment', return_value={"status": "success"}):
            
            # Create job
            job_info = await job_manager.create_job("exp_123", "agent_123")
            
            # Assertions
            assert job_info.experiment_id == "exp_123"
            assert job_info.agent_id == "agent_123"
            assert job_info.job_id is not None
            assert job_info.status.status == JobStatusEnum.ASSIGNED
            assert "scheduled" in job_info.status.message
            
            # Verify storage calls
            mock_storage.get_experiment.assert_called_once_with("exp_123")
            mock_storage.get_agent.assert_called_once_with("agent_123")
            mock_storage.set_running_job.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_create_job_experiment_not_found(self, job_manager, mock_storage):
        """Test job creation when experiment doesn't exist."""
        # Setup mocks
        mock_storage.get_experiment.return_value = None
        
        # Create job should return JobInfo with FAILED status
        job_info = await job_manager.create_job("nonexistent_exp", "agent_123")
        
        assert job_info.status.status == JobStatusEnum.FAILED
        assert "does not exist" in job_info.status.message

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_create_job_agent_not_found(self, job_manager, mock_storage, sample_experiment):
        """Test job creation when agent doesn't exist."""
        # Setup mocks
        mock_storage.get_experiment.return_value = sample_experiment
        mock_storage.get_agent.return_value = None
        
        # Create job should return JobInfo with FAILED status
        job_info = await job_manager.create_job("exp_123", "nonexistent_agent")
        
        assert job_info.status.status == JobStatusEnum.FAILED
        assert "does not exist" in job_info.status.message

    @pytest.mark.asyncio
    async def test_create_job_agent_unavailable_via_status_fetch(self, job_manager, mock_storage, sample_experiment, sample_agent_info):
        """Test job creation when agent status fetch shows agent is unavailable."""
        # Setup mocks
        mock_storage.get_experiment.return_value = sample_experiment
        mock_storage.get_agent.return_value = sample_agent_info
        
        # Mock agent utility to return unavailable status (ERROR status means unavailable)
        from ruckus_common.models import AgentStatus, AgentStatusEnum
        error_agent_status = AgentStatus(
            agent_id="agent_123",
            status=AgentStatusEnum.ERROR,
            uptime_seconds=0,
            running_jobs=[]
        )
        
        with patch.object(job_manager.agent_utility, 'get_agent_status', return_value=error_agent_status):
            # Create job
            job_info = await job_manager.create_job("exp_123", "agent_123")
            
            # Should create job with FAILED status
            assert job_info.status.status == JobStatusEnum.FAILED
            assert "error state" in job_info.status.message
            mock_storage.add_failed_job.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_create_job_agent_busy(self, job_manager, mock_storage, sample_experiment, sample_agent_info):
        """Test job creation when agent is busy."""
        # Setup mocks
        busy_agent_status = AgentStatus(
            agent_id="agent_123",
            status=AgentStatusEnum.ACTIVE,
            uptime_seconds=3600,
            running_jobs=["other_job"]
        )
        
        mock_storage.get_experiment.return_value = sample_experiment
        mock_storage.get_agent.return_value = sample_agent_info
        
        # Mock agent utility methods
        with patch.object(job_manager.agent_utility, 'get_agent_status', return_value=busy_agent_status):
            
            # Create job
            job_info = await job_manager.create_job("exp_123", "agent_123")
            
            # Should create job with QUEUED status
            assert job_info.status.status == JobStatusEnum.QUEUED
            assert "busy" in job_info.status.message
            mock_storage.add_queued_job.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_create_job_agent_communication_error(self, job_manager, mock_storage, sample_experiment, sample_agent_info):
        """Test job creation when agent communication fails."""
        # Setup mocks
        mock_storage.get_experiment.return_value = sample_experiment
        mock_storage.get_agent.return_value = sample_agent_info
        
        # Mock agent utility to raise exception
        with patch.object(job_manager.agent_utility, 'get_agent_status', side_effect=Exception("Connection failed")):
            
            # Create job
            job_info = await job_manager.create_job("exp_123", "agent_123")
            
            # Should create job with FAILED status
            assert job_info.status.status == JobStatusEnum.FAILED
            assert "Failed to get agent status" in job_info.status.message
            mock_storage.add_failed_job.assert_called_once()


class TestJobManagerStatusTracking:
    """Tests for job status tracking and polling."""
    
    @pytest.mark.asyncio
    async def test_get_job_status_running_job(self, job_manager, mock_storage):
        """Test getting status for a running job."""
        # Setup mock running job
        running_job = JobInfo(
            job_id="job_123",
            experiment_id="exp_123",
            agent_id="agent_123",
            created_time=datetime.now(timezone.utc),
            status=JobStatus(status=JobStatusEnum.RUNNING, message="Running")
        )
        
        mock_storage.list_registered_agent_info.return_value = [
            RegisteredAgentInfo(
                agent_id="agent_123",
                agent_name="Test Agent",
                agent_type=AgentType.GRAY_BOX,
                agent_url="http://localhost:8001",
                registered_at=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )
        ]
        mock_storage.get_running_job.return_value = running_job
        mock_storage.get_queued_jobs.return_value = []
        mock_storage.get_completed_jobs.return_value = []
        mock_storage.get_failed_jobs.return_value = []
        
        # Get job status
        result = await job_manager.get_job_status("job_123")
        
        assert result is not None
        assert result.job_id == "job_123"
        assert result.status.status == JobStatusEnum.RUNNING

    @pytest.mark.asyncio
    async def test_get_job_status_queued_job(self, job_manager, mock_storage):
        """Test getting status for a queued job."""
        # Setup mock queued job
        queued_job = JobInfo(
            job_id="job_123",
            experiment_id="exp_123",
            agent_id="agent_123",
            created_time=datetime.now(timezone.utc),
            status=JobStatus(status=JobStatusEnum.QUEUED, message="Queued")
        )
        
        mock_storage.list_registered_agent_info.return_value = [
            RegisteredAgentInfo(
                agent_id="agent_123",
                agent_name="Test Agent",
                agent_type=AgentType.GRAY_BOX,
                agent_url="http://localhost:8001",
                registered_at=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )
        ]
        mock_storage.get_running_job.return_value = None
        mock_storage.get_queued_jobs.return_value = [queued_job]
        mock_storage.get_completed_jobs.return_value = []
        mock_storage.get_failed_jobs.return_value = []
        
        # Get job status
        result = await job_manager.get_job_status("job_123")
        
        assert result is not None
        assert result.job_id == "job_123"
        assert result.status.status == JobStatusEnum.QUEUED

    @pytest.mark.asyncio
    async def test_get_job_status_not_found(self, job_manager, mock_storage):
        """Test getting status for non-existent job."""
        mock_storage.list_registered_agent_info.return_value = []
        
        # Get job status
        result = await job_manager.get_job_status("nonexistent_job")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_list_job_info(self, job_manager, mock_storage):
        """Test listing all job information."""
        # Setup mock jobs
        agent_info = RegisteredAgentInfo(
            agent_id="agent_123",
            agent_name="Test Agent",
            agent_type=AgentType.GRAY_BOX,
            agent_url="http://localhost:8001",
            registered_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        running_job = JobInfo(
            job_id="job_running",
            experiment_id="exp_1",
            agent_id="agent_123",
            created_time=datetime.now(timezone.utc),
            status=JobStatus(status=JobStatusEnum.RUNNING, message="Running")
        )
        
        completed_job = JobInfo(
            job_id="job_completed",
            experiment_id="exp_2",
            agent_id="agent_123",
            created_time=datetime.now(timezone.utc),
            status=JobStatus(status=JobStatusEnum.COMPLETED, message="Completed")
        )
        
        mock_storage.list_registered_agent_info.return_value = [agent_info]
        mock_storage.get_running_job.return_value = running_job
        mock_storage.get_queued_jobs.return_value = []
        mock_storage.get_completed_jobs.return_value = [completed_job]
        mock_storage.get_failed_jobs.return_value = []
        
        # List jobs
        result = await job_manager.list_job_info()
        
        assert "agent_123" in result
        assert len(result["agent_123"]) == 2
        job_ids = [job.job_id for job in result["agent_123"]]
        assert "job_running" in job_ids
        assert "job_completed" in job_ids


class TestJobManagerCancellation:
    """Tests for job cancellation functionality."""
    
    @pytest.mark.asyncio
    async def test_cancel_job_success(self, job_manager, mock_storage, sample_agent_info):
        """Test successful job cancellation."""
        # Setup mock job and agent
        running_job = JobInfo(
            job_id="job_123",
            experiment_id="exp_123",
            agent_id="agent_123",
            created_time=datetime.now(timezone.utc),
            status=JobStatus(status=JobStatusEnum.RUNNING, message="Running")
        )
        
        mock_storage.list_registered_agent_info.return_value = [sample_agent_info]
        mock_storage.get_running_job.return_value = running_job
        mock_storage.get_queued_jobs.return_value = []
        mock_storage.get_completed_jobs.return_value = []
        mock_storage.get_failed_jobs.return_value = []
        mock_storage.get_agent.return_value = sample_agent_info
        
        # Mock agent utility cancellation
        with patch.object(job_manager.agent_utility, 'cancel_experiment', return_value=True):
            
            # Cancel job
            await job_manager.cancel_job("job_123")
            
            # Verify storage calls
            mock_storage.clear_running_job.assert_called_once_with("agent_123")
            mock_storage.add_failed_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_job_not_found(self, job_manager, mock_storage):
        """Test canceling non-existent job."""
        mock_storage.list_registered_agent_info.return_value = []
        
        # Cancel job should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            await job_manager.cancel_job("nonexistent_job")
        
        assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cancel_queued_job(self, job_manager, mock_storage, sample_agent_info):
        """Test canceling queued job."""
        # Setup mock queued job
        queued_job = JobInfo(
            job_id="job_123",
            experiment_id="exp_123",
            agent_id="agent_123",
            created_time=datetime.now(timezone.utc),
            status=JobStatus(status=JobStatusEnum.QUEUED, message="Queued")
        )
        
        mock_storage.list_registered_agent_info.return_value = [sample_agent_info]
        mock_storage.get_running_job.return_value = None
        # First call returns the job, second call returns empty (after cancellation)
        mock_storage.get_queued_jobs.side_effect = [[queued_job], []]
        mock_storage.get_completed_jobs.return_value = []
        mock_storage.get_failed_jobs.return_value = []
        mock_storage.get_agent.return_value = sample_agent_info
        
        # Mock agent utility cancellation
        with patch.object(job_manager.agent_utility, 'cancel_experiment', return_value=True):
            
            # Cancel job
            await job_manager.cancel_job("job_123")
            
            # Verify storage calls
            mock_storage.remove_queued_job.assert_called_once_with("agent_123", "job_123")
            mock_storage.add_failed_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_job_agent_not_found(self, job_manager, mock_storage, sample_agent_info):
        """Test canceling job when agent doesn't exist."""
        # Setup mock job
        running_job = JobInfo(
            job_id="job_123",
            experiment_id="exp_123",
            agent_id="agent_123",
            created_time=datetime.now(timezone.utc),
            status=JobStatus(status=JobStatusEnum.RUNNING, message="Running")
        )
        
        mock_storage.list_registered_agent_info.return_value = [sample_agent_info]
        mock_storage.get_running_job.return_value = running_job
        mock_storage.get_queued_jobs.return_value = []
        mock_storage.get_completed_jobs.return_value = []
        mock_storage.get_failed_jobs.return_value = []
        mock_storage.get_agent.return_value = None  # Agent not found
        
        # Cancel job should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            await job_manager.cancel_job("job_123")
        
        assert "does not exist" in str(exc_info.value)


class TestJobManagerResultProcessing:
    """Tests for job result processing."""
    
    @pytest.mark.asyncio
    async def test_process_job_results_success(self, job_manager, mock_storage, sample_agent_info):
        """Test successful job result processing."""
        # Setup mock completed job
        completed_job = JobInfo(
            job_id="job_123",
            experiment_id="exp_123",
            agent_id="agent_123",
            created_time=datetime.now(timezone.utc),
            status=JobStatus(status=JobStatusEnum.COMPLETED, message="Completed")
        )
        
        # Setup mock job result
        job_result = JobResult(
            job_id="job_123",
            experiment_id="exp_123",
            status=JobStatusEnum.COMPLETED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            duration_seconds=60.0,
            output={"output": "test result"},
            metrics={"accuracy": 0.95}
        )
        
        mock_storage.get_agent.return_value = sample_agent_info
        mock_storage.get_completed_jobs.return_value = [completed_job]
        
        # Mock agent utility get results
        with patch.object(job_manager.agent_utility, 'get_experiment_results', return_value=job_result):
            
            # Process job results
            await job_manager.process_job_results("job_123", "agent_123")
            
            # Verify storage calls
            mock_storage.store_experiment_result.assert_called_once()
            mock_storage.save_experiment_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_job_results_job_not_found(self, job_manager, mock_storage, sample_agent_info):
        """Test processing results for non-existent job."""
        mock_storage.get_agent.return_value = sample_agent_info
        mock_storage.get_completed_jobs.return_value = []  # No completed jobs
        
        # Should not raise exception, but log error
        await job_manager.process_job_results("nonexistent_job", "agent_123")
        
        # Verify no storage calls were made
        mock_storage.store_experiment_result.assert_not_called()
        mock_storage.save_experiment_results.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_job_results_agent_communication_error(self, job_manager, mock_storage, sample_agent_info):
        """Test processing results when agent communication fails."""
        # Setup mock completed job
        completed_job = JobInfo(
            job_id="job_123",
            experiment_id="exp_123",
            agent_id="agent_123",
            created_time=datetime.now(timezone.utc),
            status=JobStatus(status=JobStatusEnum.COMPLETED, message="Completed")
        )
        
        mock_storage.get_agent.return_value = sample_agent_info
        mock_storage.get_completed_jobs.return_value = [completed_job]
        
        # Mock agent utility to raise exception
        with patch.object(job_manager.agent_utility, 'get_experiment_results', side_effect=Exception("Connection failed")):
            
            # Should not raise exception, but log error
            await job_manager.process_job_results("job_123", "agent_123")
            
            # Verify no storage calls were made
            mock_storage.store_experiment_result.assert_not_called()


class TestJobManagerPolling:
    """Tests for job status polling functionality."""
    
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, job_manager_settings, mock_storage):
        """Test job manager start/stop lifecycle."""
        manager = JobManager(job_manager_settings, mock_storage)
        
        # Should start successfully
        await manager.start()
        assert manager._running is True
        
        # Should stop successfully
        await manager.stop()
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_polling_task_creation(self, job_manager, mock_storage, sample_experiment, sample_agent_info, sample_agent_status):
        """Test that polling tasks are created for assigned jobs."""
        # Setup mocks
        mock_storage.get_experiment.return_value = sample_experiment
        mock_storage.get_agent.return_value = sample_agent_info
        
        # Mock agent utility methods
        with patch.object(job_manager.agent_utility, 'get_agent_status', return_value=sample_agent_status), \
             patch.object(job_manager.agent_utility, 'execute_experiment', return_value={"status": "success"}):
            
            # Create job
            job_info = await job_manager.create_job("exp_123", "agent_123")
            
            # Should have created a polling task
            assert job_info.job_id in job_manager.polling_tasks
            
            # Clean up task
            if not job_manager.polling_tasks[job_info.job_id].done():
                job_manager.polling_tasks[job_info.job_id].cancel()

    @pytest.mark.asyncio
    async def test_polling_task_cleanup_on_stop(self, job_manager_settings, mock_storage):
        """Test that polling tasks are cleaned up when manager stops."""
        manager = JobManager(job_manager_settings, mock_storage)
        await manager.start()
        
        # Add a mock polling task
        task = asyncio.create_task(asyncio.sleep(10))
        manager.polling_tasks["test_job"] = task
        
        # Stop manager
        await manager.stop()
        
        # Task should be cancelled and removed
        assert task.cancelled()
        assert len(manager.polling_tasks) == 0


class TestJobManagerAgentInteraction:
    """Tests for JobManager interaction with agents."""
    
    @pytest.mark.asyncio
    async def test_get_agent_jobs(self, job_manager, mock_storage):
        """Test getting jobs for a specific agent."""
        # Setup mock jobs
        running_job = JobInfo(
            job_id="running_job",
            experiment_id="exp_1",
            agent_id="agent_123",
            created_time=datetime.now(timezone.utc),
            status=JobStatus(status=JobStatusEnum.RUNNING, message="Running")
        )
        
        queued_job = JobInfo(
            job_id="queued_job",
            experiment_id="exp_2",
            agent_id="agent_123",
            created_time=datetime.now(timezone.utc),
            status=JobStatus(status=JobStatusEnum.QUEUED, message="Queued")
        )
        
        mock_storage.get_running_job.return_value = running_job
        mock_storage.get_queued_jobs.return_value = [queued_job]
        mock_storage.get_completed_jobs.return_value = []
        mock_storage.get_failed_jobs.return_value = []
        
        # Get agent jobs
        result = await job_manager.get_agent_jobs("agent_123")
        
        assert result["running"] == running_job
        assert len(result["queued"]) == 1
        assert result["queued"][0] == queued_job
        assert result["completed"] == []
        assert result["failed"] == []

    @pytest.mark.asyncio
    async def test_process_next_job(self, job_manager, mock_storage, sample_experiment, sample_agent_info, sample_agent_status):
        """Test processing the next queued job for an agent."""
        # Setup mock queued job
        queued_job = JobInfo(
            job_id="queued_job",
            experiment_id="exp_123",
            agent_id="agent_123",
            created_time=datetime.now(timezone.utc),
            status=JobStatus(status=JobStatusEnum.QUEUED, message="Queued")
        )
        
        mock_storage.get_queued_jobs.return_value = [queued_job]
        mock_storage.get_experiment.return_value = sample_experiment
        mock_storage.get_agent.return_value = sample_agent_info
        
        # Mock agent utility methods
        with patch.object(job_manager.agent_utility, 'get_agent_status', return_value=sample_agent_status), \
             patch.object(job_manager.agent_utility, 'execute_experiment', return_value={"status": "success"}):
            
            # Process next job
            await job_manager.process_next_job("agent_123")
            
            # Verify the job was removed from queue and processed
            mock_storage.remove_queued_job.assert_called_once_with("agent_123", "queued_job")

    @pytest.mark.asyncio
    async def test_process_next_job_no_queued_jobs(self, job_manager, mock_storage):
        """Test processing next job when no jobs are queued."""
        mock_storage.get_queued_jobs.return_value = []
        
        # Should not raise exception
        await job_manager.process_next_job("agent_123")
        
        # No jobs should be processed
        mock_storage.remove_queued_job.assert_not_called()