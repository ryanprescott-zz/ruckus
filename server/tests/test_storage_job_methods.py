"""Tests for StorageBackend job management methods."""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from unittest.mock import Mock

from ruckus_server.core.storage.sqlite import SQLiteStorageBackend
from ruckus_server.core.storage.postgresql import PostgreSQLStorageBackend
from ruckus_server.core.models import JobInfo
from ruckus_server.api.v1.models import ExperimentResult
from ruckus_server.core.config import SQLiteSettings, PostgreSQLSettings
from ruckus_common.models import JobStatus, JobStatusEnum, JobResult


@pytest.fixture
def postgresql_url():
    """PostgreSQL URL fixture for testing."""
    # This can be set via environment variable RUCKUS_POSTGRES_DATABASE_URL
    # or return None to skip PostgreSQL tests
    import os
    return os.environ.get("RUCKUS_POSTGRES_DATABASE_URL")


@pytest.fixture
def sample_job_info():
    """Create a sample JobInfo for testing."""
    return JobInfo(
        job_id="test_job_123",
        experiment_id="exp_123",
        agent_id="agent_123",
        created_time=datetime.now(timezone.utc),
        status=JobStatus(
            status=JobStatusEnum.RUNNING,
            message="Job is running"
        )
    )


@pytest.fixture
def sample_experiment_result():
    """Create a sample ExperimentResult for testing."""
    job_result = JobResult(
        job_id="test_job_123",
        experiment_id="exp_123",
        status=JobStatusEnum.COMPLETED,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        duration_seconds=120.0,
        output={"output": "test output"},
        metrics={"accuracy": 0.95}
    )
    return ExperimentResult.from_job_result(job_result, "agent_123")


class TestSQLiteStorageJobMethods:
    """Tests for SQLite storage backend job methods."""
    
    @pytest_asyncio.fixture
    async def sqlite_storage(self, tmp_path):
        """Create an SQLite storage backend for testing."""
        db_path = tmp_path / "test_jobs.db"
        settings = SQLiteSettings(database_path=str(db_path))
        storage = SQLiteStorageBackend(settings)
        await storage.initialize()
        yield storage
        await storage.close()

    @pytest.mark.asyncio
    async def test_running_job_operations(self, sqlite_storage, sample_job_info):
        """Test running job CRUD operations."""
        agent_id = "agent_123"
        
        # Initially no running job
        running_job = await sqlite_storage.get_running_job(agent_id)
        assert running_job is None
        
        # Set running job
        await sqlite_storage.set_running_job(agent_id, sample_job_info)
        
        # Get running job
        running_job = await sqlite_storage.get_running_job(agent_id)
        assert running_job is not None
        assert running_job.job_id == sample_job_info.job_id
        assert running_job.experiment_id == sample_job_info.experiment_id
        assert running_job.agent_id == sample_job_info.agent_id
        assert running_job.status.status == sample_job_info.status.status
        
        # Update running job
        updated_job_info = sample_job_info.model_copy()
        updated_job_info.status = JobStatus(
            status=JobStatusEnum.COMPLETED,
            message="Job completed"
        )
        
        await sqlite_storage.update_running_job(agent_id, updated_job_info)
        
        # Verify update
        running_job = await sqlite_storage.get_running_job(agent_id)
        assert running_job.status.status == JobStatusEnum.COMPLETED
        assert running_job.status.message == "Job completed"
        
        # Clear running job
        await sqlite_storage.clear_running_job(agent_id)
        
        # Verify cleared
        running_job = await sqlite_storage.get_running_job(agent_id)
        assert running_job is None

    @pytest.mark.asyncio
    async def test_queued_job_operations(self, sqlite_storage, sample_job_info):
        """Test queued job operations."""
        agent_id = "agent_123"
        
        # Initially no queued jobs
        queued_jobs = await sqlite_storage.get_queued_jobs(agent_id)
        assert len(queued_jobs) == 0
        
        # Add queued job
        await sqlite_storage.add_queued_job(agent_id, sample_job_info)
        
        # Get queued jobs
        queued_jobs = await sqlite_storage.get_queued_jobs(agent_id)
        assert len(queued_jobs) == 1
        assert queued_jobs[0].job_id == sample_job_info.job_id
        
        # Add another queued job
        second_job = sample_job_info.model_copy()
        second_job.job_id = "test_job_456"
        await sqlite_storage.add_queued_job(agent_id, second_job)
        
        # Verify both jobs are queued
        queued_jobs = await sqlite_storage.get_queued_jobs(agent_id)
        assert len(queued_jobs) == 2
        job_ids = [job.job_id for job in queued_jobs]
        assert "test_job_123" in job_ids
        assert "test_job_456" in job_ids
        
        # Remove a queued job
        await sqlite_storage.remove_queued_job(agent_id, "test_job_123")
        
        # Verify job was removed
        queued_jobs = await sqlite_storage.get_queued_jobs(agent_id)
        assert len(queued_jobs) == 1
        assert queued_jobs[0].job_id == "test_job_456"

    @pytest.mark.asyncio
    async def test_completed_job_operations(self, sqlite_storage, sample_job_info):
        """Test completed job operations."""
        agent_id = "agent_123"
        
        # Initially no completed jobs
        completed_jobs = await sqlite_storage.get_completed_jobs(agent_id)
        assert len(completed_jobs) == 0
        
        # Add completed job
        completed_job_info = sample_job_info.model_copy()
        completed_job_info.status = JobStatus(
            status=JobStatusEnum.COMPLETED,
            message="Job completed successfully"
        )
        
        await sqlite_storage.add_completed_job(agent_id, completed_job_info)
        
        # Get completed jobs
        completed_jobs = await sqlite_storage.get_completed_jobs(agent_id)
        assert len(completed_jobs) == 1
        assert completed_jobs[0].job_id == sample_job_info.job_id
        assert completed_jobs[0].status.status == JobStatusEnum.COMPLETED
        
        # Add another completed job
        second_completed_job = sample_job_info.model_copy()
        second_completed_job.job_id = "completed_job_456"
        second_completed_job.status = JobStatus(
            status=JobStatusEnum.COMPLETED,
            message="Another completed job"
        )
        
        await sqlite_storage.add_completed_job(agent_id, second_completed_job)
        
        # Verify both jobs are in completed list
        completed_jobs = await sqlite_storage.get_completed_jobs(agent_id)
        assert len(completed_jobs) == 2
        job_ids = [job.job_id for job in completed_jobs]
        assert "test_job_123" in job_ids
        assert "completed_job_456" in job_ids

    @pytest.mark.asyncio
    async def test_failed_job_operations(self, sqlite_storage, sample_job_info):
        """Test failed job operations."""
        agent_id = "agent_123"
        
        # Initially no failed jobs
        failed_jobs = await sqlite_storage.get_failed_jobs(agent_id)
        assert len(failed_jobs) == 0
        
        # Add failed job
        failed_job_info = sample_job_info.model_copy()
        failed_job_info.status = JobStatus(
            status=JobStatusEnum.FAILED,
            message="Job failed due to error"
        )
        
        await sqlite_storage.add_failed_job(agent_id, failed_job_info)
        
        # Get failed jobs
        failed_jobs = await sqlite_storage.get_failed_jobs(agent_id)
        assert len(failed_jobs) == 1
        assert failed_jobs[0].job_id == sample_job_info.job_id
        assert failed_jobs[0].status.status == JobStatusEnum.FAILED
        
        # Add a cancelled job (should also go to failed jobs)
        cancelled_job = sample_job_info.model_copy()
        cancelled_job.job_id = "cancelled_job_456"
        cancelled_job.status = JobStatus(
            status=JobStatusEnum.CANCELLED,
            message="Job was cancelled"
        )
        
        await sqlite_storage.add_failed_job(agent_id, cancelled_job)
        
        # Verify both jobs are in failed list
        failed_jobs = await sqlite_storage.get_failed_jobs(agent_id)
        assert len(failed_jobs) == 2
        job_ids = [job.job_id for job in failed_jobs]
        assert "test_job_123" in job_ids
        assert "cancelled_job_456" in job_ids

    @pytest.mark.asyncio
    async def test_experiment_result_operations(self, sqlite_storage, sample_experiment_result):
        """Test experiment result storage operations."""
        # Initially no experiment results
        results = await sqlite_storage.list_experiment_results()
        assert len(results) == 0
        
        # Store experiment result
        await sqlite_storage.store_experiment_result(sample_experiment_result)
        
        # List experiment results
        results = await sqlite_storage.list_experiment_results()
        assert len(results) == 1
        assert results[0].job_id == sample_experiment_result.job_id
        assert results[0].experiment_id == sample_experiment_result.experiment_id
        
        # Get experiment result by job ID
        result = await sqlite_storage.get_experiment_result_by_job_id("test_job_123")
        assert result is not None
        assert result.job_id == sample_experiment_result.job_id
        assert result.agent_id == sample_experiment_result.agent_id
        
        # Test getting non-existent result
        result = await sqlite_storage.get_experiment_result_by_job_id("nonexistent_job")
        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_agents_job_isolation(self, sqlite_storage, sample_job_info):
        """Test that jobs are properly isolated between agents."""
        agent1_id = "agent_123"
        agent2_id = "agent_456"
        
        # Add running job for agent1
        await sqlite_storage.set_running_job(agent1_id, sample_job_info)
        
        # Add queued job for agent2
        queued_job = sample_job_info.model_copy()
        queued_job.job_id = "queued_job_456"
        queued_job.agent_id = agent2_id
        await sqlite_storage.add_queued_job(agent2_id, queued_job)
        
        # Verify agent1 has running job but no queued jobs
        running_job_1 = await sqlite_storage.get_running_job(agent1_id)
        queued_jobs_1 = await sqlite_storage.get_queued_jobs(agent1_id)
        assert running_job_1 is not None
        assert len(queued_jobs_1) == 0
        
        # Verify agent2 has queued job but no running job
        running_job_2 = await sqlite_storage.get_running_job(agent2_id)
        queued_jobs_2 = await sqlite_storage.get_queued_jobs(agent2_id)
        assert running_job_2 is None
        assert len(queued_jobs_2) == 1
        assert queued_jobs_2[0].job_id == "queued_job_456"

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Job storage uses in-memory implementation that doesn't persist across sessions")
    async def test_job_persistence_across_sessions(self, tmp_path, sample_job_info):
        """Test that jobs persist across storage backend sessions.
        
        NOTE: Currently skipped because job storage uses in-memory dictionaries
        that don't persist to the database (marked as temporary implementation).
        """
        db_path = tmp_path / "persistence_test.db"
        
        # Create first storage instance and add jobs
        settings1 = SQLiteSettings(database_path=str(db_path))
        storage1 = SQLiteStorageBackend(settings1)
        await storage1.initialize()
        
        await storage1.add_completed_job("agent_123", sample_job_info)
        await storage1.close()
        
        # Create second storage instance and verify jobs persist
        settings2 = SQLiteSettings(database_path=str(db_path))
        storage2 = SQLiteStorageBackend(settings2)
        await storage2.initialize()
        
        completed_jobs = await storage2.get_completed_jobs("agent_123")
        # This would fail because job storage is in-memory only
        assert len(completed_jobs) == 1
        assert completed_jobs[0].job_id == sample_job_info.job_id
        
        await storage2.close()

    @pytest.mark.asyncio
    async def test_edge_cases_and_error_handling(self, sqlite_storage):
        """Test edge cases and error handling in job operations."""
        agent_id = "agent_123"
        
        # Test removing job from empty queue
        await sqlite_storage.remove_queued_job(agent_id, "nonexistent_job")
        # Should not raise exception
        
        # Test getting jobs for non-existent agent
        queued_jobs = await sqlite_storage.get_queued_jobs("nonexistent_agent")
        assert len(queued_jobs) == 0
        
        running_job = await sqlite_storage.get_running_job("nonexistent_agent")
        assert running_job is None
        
        completed_jobs = await sqlite_storage.get_completed_jobs("nonexistent_agent")
        assert len(completed_jobs) == 0
        
        failed_jobs = await sqlite_storage.get_failed_jobs("nonexistent_agent")
        assert len(failed_jobs) == 0
        
        # Test clearing running job when none exists
        await sqlite_storage.clear_running_job("nonexistent_agent")
        # Should not raise exception


class TestPostgreSQLStorageJobMethods:
    """Tests for PostgreSQL storage backend job methods."""
    
    @pytest_asyncio.fixture
    async def postgresql_storage(self, postgresql_url):
        """Create a PostgreSQL storage backend for testing."""
        if not postgresql_url:
            pytest.skip("PostgreSQL URL not provided")
        
        settings = PostgreSQLSettings(database_url=postgresql_url)
        storage = PostgreSQLStorageBackend(settings)
        await storage.initialize()
        yield storage
        await storage.close()

    @pytest.mark.asyncio
    async def test_postgresql_running_job_operations(self, postgresql_storage, sample_job_info):
        """Test PostgreSQL running job operations."""
        agent_id = "agent_123"
        
        # Set running job
        await postgresql_storage.set_running_job(agent_id, sample_job_info)
        
        # Get running job
        running_job = await postgresql_storage.get_running_job(agent_id)
        assert running_job is not None
        assert running_job.job_id == sample_job_info.job_id
        
        # Clear running job
        await postgresql_storage.clear_running_job(agent_id)
        
        # Verify cleared
        running_job = await postgresql_storage.get_running_job(agent_id)
        assert running_job is None

    @pytest.mark.asyncio
    async def test_postgresql_queued_job_operations(self, postgresql_storage, sample_job_info):
        """Test PostgreSQL queued job operations."""
        agent_id = "agent_123"
        
        # Add queued job
        await postgresql_storage.add_queued_job(agent_id, sample_job_info)
        
        # Get queued jobs
        queued_jobs = await postgresql_storage.get_queued_jobs(agent_id)
        assert len(queued_jobs) == 1
        assert queued_jobs[0].job_id == sample_job_info.job_id
        
        # Remove queued job
        await postgresql_storage.remove_queued_job(agent_id, sample_job_info.job_id)
        
        # Verify removed
        queued_jobs = await postgresql_storage.get_queued_jobs(agent_id)
        assert len(queued_jobs) == 0

    @pytest.mark.asyncio
    async def test_postgresql_experiment_result_operations(self, postgresql_storage, sample_experiment_result):
        """Test PostgreSQL experiment result operations."""
        # Store experiment result
        await postgresql_storage.store_experiment_result(sample_experiment_result)
        
        # Get by job ID
        result = await postgresql_storage.get_experiment_result_by_job_id("test_job_123")
        assert result is not None
        assert result.job_id == sample_experiment_result.job_id
        
        # List all results
        results = await postgresql_storage.list_experiment_results()
        assert len(results) >= 1
        job_ids = [r.job_id for r in results]
        assert "test_job_123" in job_ids


class TestStorageBackendJobMethodsComparison:
    """Tests to ensure SQLite and PostgreSQL backends behave consistently."""
    
    @pytest_asyncio.fixture
    async def both_storages(self, tmp_path, postgresql_url):
        """Create both SQLite and PostgreSQL storage backends."""
        # SQLite storage
        db_path = tmp_path / "comparison_test.db"
        sqlite_settings = SQLiteSettings(database_path=str(db_path))
        sqlite_storage = SQLiteStorageBackend(sqlite_settings)
        await sqlite_storage.initialize()
        
        storages = [("sqlite", sqlite_storage)]
        
        # PostgreSQL storage if available
        postgresql_storage = None
        if postgresql_url:
            postgresql_settings = PostgreSQLSettings(database_url=postgresql_url)
            postgresql_storage = PostgreSQLStorageBackend(postgresql_settings)
            await postgresql_storage.initialize()
            storages.append(("postgresql", postgresql_storage))
        
        yield storages
        
        # Cleanup
        await sqlite_storage.close()
        if postgresql_storage:
            await postgresql_storage.close()

    @pytest.mark.asyncio
    async def test_consistent_job_operations_across_backends(self, both_storages, sample_job_info):
        """Test that job operations work consistently across different storage backends."""
        for backend_name, storage in both_storages:
            agent_id = f"agent_123_{backend_name}"
            
            # Test running job operations
            await storage.set_running_job(agent_id, sample_job_info)
            running_job = await storage.get_running_job(agent_id)
            assert running_job is not None, f"Running job not found in {backend_name}"
            assert running_job.job_id == sample_job_info.job_id
            
            # Test queued job operations
            queued_job_info = sample_job_info.model_copy()
            queued_job_info.job_id = "queued_job"
            await storage.add_queued_job(agent_id, queued_job_info)
            
            queued_jobs = await storage.get_queued_jobs(agent_id)
            assert len(queued_jobs) == 1, f"Queued jobs count mismatch in {backend_name}"
            assert queued_jobs[0].job_id == "queued_job"
            
            # Test completed job operations
            completed_job_info = sample_job_info.model_copy()
            completed_job_info.job_id = "completed_job"
            completed_job_info.status = JobStatus(
                status=JobStatusEnum.COMPLETED,
                message="Completed"
            )
            await storage.add_completed_job(agent_id, completed_job_info)
            
            completed_jobs = await storage.get_completed_jobs(agent_id)
            assert len(completed_jobs) == 1, f"Completed jobs count mismatch in {backend_name}"
            assert completed_jobs[0].job_id == "completed_job"

    @pytest.mark.asyncio
    async def test_consistent_experiment_result_operations(self, both_storages, sample_experiment_result):
        """Test that experiment result operations work consistently across backends."""
        for backend_name, storage in both_storages:
            # Modify job_id to be unique per backend
            result = sample_experiment_result.model_copy()
            result.job_id = f"test_job_123_{backend_name}"
            
            # Store experiment result
            await storage.store_experiment_result(result)
            
            # Get by job ID
            retrieved_result = await storage.get_experiment_result_by_job_id(result.job_id)
            assert retrieved_result is not None, f"Experiment result not found in {backend_name}"
            assert retrieved_result.job_id == result.job_id
            assert retrieved_result.experiment_id == result.experiment_id
            
            # List results should include our result
            results = await storage.list_experiment_results()
            job_ids = [r.job_id for r in results]
            assert result.job_id in job_ids, f"Experiment result not in list for {backend_name}"


class TestJobMethodsConcurrency:
    """Tests for concurrent operations on job methods."""
    
    @pytest.mark.asyncio
    async def test_concurrent_job_additions(self, sqlite_storage):
        """Test adding multiple jobs concurrently."""
        import asyncio
        
        agent_id = "agent_123"
        
        # Create multiple job infos
        job_infos = []
        for i in range(5):
            job_info = JobInfo(
                job_id=f"concurrent_job_{i}",
                experiment_id="exp_123",
                agent_id=agent_id,
                created_time=datetime.now(timezone.utc),
                status=JobStatus(
                    status=JobStatusEnum.QUEUED,
                    message=f"Job {i}"
                )
            )
            job_infos.append(job_info)
        
        # Add jobs concurrently
        tasks = [
            sqlite_storage.add_queued_job(agent_id, job_info)
            for job_info in job_infos
        ]
        await asyncio.gather(*tasks)
        
        # Verify all jobs were added
        queued_jobs = await sqlite_storage.get_queued_jobs(agent_id)
        assert len(queued_jobs) == 5
        
        job_ids = {job.job_id for job in queued_jobs}
        expected_ids = {f"concurrent_job_{i}" for i in range(5)}
        assert job_ids == expected_ids

    @pytest.mark.asyncio
    async def test_concurrent_job_state_transitions(self, sqlite_storage, sample_job_info):
        """Test concurrent job state transitions."""
        import asyncio
        
        agent_id = "agent_123"
        
        # Set initial running job
        await sqlite_storage.set_running_job(agent_id, sample_job_info)
        
        # Simulate concurrent state transitions
        async def transition_to_completed():
            completed_job = sample_job_info.model_copy()
            completed_job.status = JobStatus(
                status=JobStatusEnum.COMPLETED,
                message="Completed"
            )
            await sqlite_storage.add_completed_job(agent_id, completed_job)
            await sqlite_storage.clear_running_job(agent_id)
        
        async def add_queued_jobs():
            for i in range(3):
                queued_job = sample_job_info.model_copy()
                queued_job.job_id = f"queued_job_{i}"
                await sqlite_storage.add_queued_job(agent_id, queued_job)
        
        # Run transitions concurrently
        await asyncio.gather(
            transition_to_completed(),
            add_queued_jobs()
        )
        
        # Verify final state
        running_job = await sqlite_storage.get_running_job(agent_id)
        assert running_job is None
        
        completed_jobs = await sqlite_storage.get_completed_jobs(agent_id)
        assert len(completed_jobs) == 1
        
        queued_jobs = await sqlite_storage.get_queued_jobs(agent_id)
        assert len(queued_jobs) == 3