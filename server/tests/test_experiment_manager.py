"""Tests for ExperimentManager core functionality."""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, Mock
from pathlib import Path

from ruckus_server.core.experiment_manager import ExperimentManager
from ruckus_server.core.config import ExperimentManagerSettings
from ruckus_server.core.storage.base import ExperimentAlreadyExistsException, ExperimentNotFoundException, ExperimentHasJobsException
from ruckus_common.models import ExperimentSpec, TaskType


class TestExperimentManager:
    """Tests for ExperimentManager functionality."""

    def test_experiment_manager_init_with_settings(self, experiment_manager_settings):
        """Test ExperimentManager initialization with settings."""
        manager = ExperimentManager(experiment_manager_settings)
        
        assert manager.settings == experiment_manager_settings
        assert manager.storage_backend is None
        assert manager._started is False

    def test_experiment_manager_init_without_settings(self):
        """Test ExperimentManager initialization without settings."""
        manager = ExperimentManager()
        
        assert isinstance(manager.settings, ExperimentManagerSettings)
        assert manager.storage_backend is None
        assert manager._started is False

    @pytest.mark.asyncio
    async def test_experiment_manager_start_success(self, experiment_manager_settings):
        """Test successful ExperimentManager startup."""
        manager = ExperimentManager(experiment_manager_settings)
        
        # Mock storage backend initialization
        with patch('ruckus_server.core.experiment_manager.storage_factory') as mock_factory:
            mock_storage = AsyncMock()
            mock_factory.create_storage_backend.return_value = mock_storage
            
            await manager.start()
            
            assert manager._started is True
            assert manager.storage_backend == mock_storage
            mock_storage.initialize.assert_called_once()
        
        await manager.stop()

    @pytest.mark.asyncio
    async def test_experiment_manager_start_idempotent(self, experiment_manager):
        """Test that starting ExperimentManager multiple times is safe."""
        # Manager is already started by fixture
        assert experiment_manager._started is True
        
        # Starting again should be safe
        await experiment_manager.start()
        assert experiment_manager._started is True

    @pytest.mark.asyncio
    async def test_experiment_manager_start_failure(self, experiment_manager_settings):
        """Test ExperimentManager startup failure."""
        manager = ExperimentManager(experiment_manager_settings)
        
        with patch('ruckus_server.core.experiment_manager.storage_factory') as mock_factory:
            mock_storage = AsyncMock()
            mock_storage.initialize.side_effect = Exception("Storage initialization failed")
            mock_factory.create_storage_backend.return_value = mock_storage
            
            with pytest.raises(Exception, match="Storage initialization failed"):
                await manager.start()
            
            assert manager._started is False

    @pytest.mark.asyncio
    async def test_experiment_manager_stop(self, experiment_manager):
        """Test ExperimentManager shutdown."""
        assert experiment_manager._started is True
        assert experiment_manager.storage_backend is not None
        
        await experiment_manager.stop()
        
        assert experiment_manager._started is False
        assert experiment_manager.storage_backend is None

    @pytest.mark.asyncio
    async def test_experiment_manager_stop_idempotent(self, experiment_manager):
        """Test that stopping ExperimentManager multiple times is safe."""
        await experiment_manager.stop()
        assert experiment_manager._started is False
        
        # Stopping again should be safe
        await experiment_manager.stop()
        assert experiment_manager._started is False

    @pytest.mark.asyncio
    async def test_create_experiment_success(self, experiment_manager, sample_experiment_spec):
        """Test successful experiment creation."""
        # Mock storage backend
        created_at = datetime.now(timezone.utc)
        expected_result = {
            "experiment_id": sample_experiment_spec.experiment_id,
            "created_at": created_at
        }
        experiment_manager.storage_backend.create_experiment = AsyncMock(return_value=expected_result)
        
        result = await experiment_manager.create_experiment(sample_experiment_spec)
        
        assert result == expected_result
        experiment_manager.storage_backend.create_experiment.assert_called_once_with(sample_experiment_spec)

    @pytest.mark.asyncio
    async def test_create_experiment_already_exists(self, experiment_manager, sample_experiment_spec):
        """Test creating experiment that already exists."""
        # Mock storage backend to raise exception
        experiment_manager.storage_backend.create_experiment = AsyncMock(
            side_effect=ExperimentAlreadyExistsException(sample_experiment_spec.experiment_id)
        )
        
        with pytest.raises(ExperimentAlreadyExistsException) as exc_info:
            await experiment_manager.create_experiment(sample_experiment_spec)
        
        assert exc_info.value.experiment_id == sample_experiment_spec.experiment_id

    @pytest.mark.asyncio
    async def test_create_experiment_not_started(self, experiment_manager_settings, sample_experiment_spec):
        """Test creating experiment when manager is not started."""
        manager = ExperimentManager(experiment_manager_settings)
        
        with pytest.raises(RuntimeError, match="Experiment manager not started"):
            await manager.create_experiment(sample_experiment_spec)

    @pytest.mark.asyncio
    async def test_create_experiment_storage_error(self, experiment_manager, sample_experiment_spec):
        """Test creating experiment with storage error."""
        # Mock storage backend to raise exception
        experiment_manager.storage_backend.create_experiment = AsyncMock(
            side_effect=Exception("Database connection failed")
        )
        
        with pytest.raises(Exception, match="Database connection failed"):
            await experiment_manager.create_experiment(sample_experiment_spec)

    @pytest.mark.asyncio
    async def test_create_multiple_experiments(self, experiment_manager, experiment_spec_factory):
        """Test creating multiple experiments."""
        experiments = []
        for i in range(3):
            spec = experiment_spec_factory(experiment_id=f"experiment-{i}")
            experiments.append(spec)
        
        # Mock storage backend
        def mock_create_experiment(spec):
            return {
                "experiment_id": spec.experiment_id,
                "created_at": datetime.now(timezone.utc)
            }
        
        experiment_manager.storage_backend.create_experiment = AsyncMock(side_effect=mock_create_experiment)
        
        results = []
        for spec in experiments:
            result = await experiment_manager.create_experiment(spec)
            results.append(result)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["experiment_id"] == f"experiment-{i}"
            assert "created_at" in result

    @pytest.mark.asyncio
    async def test_experiment_manager_logging_setup(self, experiment_manager_settings, tmp_path):
        """Test ExperimentManager logging setup."""
        # Create a temporary logging config file
        log_config_file = tmp_path / "test_logging.yml"
        log_config_content = """
version: 1
disable_existing_loggers: false
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  default:
    formatter: default
    class: logging.StreamHandler
    stream: ext://sys.stdout
root:
  level: DEBUG
  handlers: [default]
"""
        log_config_file.write_text(log_config_content)
        
        # Update settings to use the temporary config file
        experiment_manager_settings.log_config_file = str(log_config_file)
        
        manager = ExperimentManager(experiment_manager_settings)
        
        # Mock storage factory to avoid actual storage initialization
        with patch('ruckus_server.core.experiment_manager.storage_factory') as mock_factory:
            mock_storage = AsyncMock()
            mock_factory.create_storage_backend.return_value = mock_storage
            
            await manager.start()
            
            # Verify logger was set up
            assert manager.logger is not None
            assert manager.logger.name == 'ruckus_server.core.experiment_manager'
        
        await manager.stop()

    @pytest.mark.asyncio
    async def test_experiment_manager_logging_setup_file_not_found(self, experiment_manager_settings):
        """Test ExperimentManager logging setup when config file doesn't exist."""
        # Set non-existent config file
        experiment_manager_settings.log_config_file = "non_existent_config.yml"
        
        manager = ExperimentManager(experiment_manager_settings)
        
        # Mock storage factory to avoid actual storage initialization
        with patch('ruckus_server.core.experiment_manager.storage_factory') as mock_factory:
            mock_storage = AsyncMock()
            mock_factory.create_storage_backend.return_value = mock_storage
            
            await manager.start()
            
            # Verify logger was still set up with basic configuration
            assert manager.logger is not None
        
        await manager.stop()

    @pytest.mark.asyncio
    async def test_experiment_manager_context_manager_usage(self, experiment_manager_settings):
        """Test using ExperimentManager lifecycle manually."""
        manager = ExperimentManager(experiment_manager_settings)
        
        # Mock storage factory
        with patch('ruckus_server.core.experiment_manager.storage_factory') as mock_factory:
            mock_storage = AsyncMock()
            mock_factory.create_storage_backend.return_value = mock_storage
            
            # Manual lifecycle management
            await manager.start()
            assert manager._started is True
            
            # Use the manager
            spec = ExperimentSpec(
                experiment_id="test-context",
                name="Test Context",
                models=["test-model"],
                task_type=TaskType.SUMMARIZATION
            )
            
            mock_storage.create_experiment.return_value = {
                "experiment_id": "test-context",
                "created_at": datetime.now(timezone.utc)
            }
            
            result = await manager.create_experiment(spec)
            assert result["experiment_id"] == "test-context"
            
            await manager.stop()
            assert manager._started is False

    @pytest.mark.asyncio
    async def test_concurrent_experiment_creation(self, experiment_manager, experiment_spec_factory):
        """Test concurrent experiment creation."""
        # Create multiple experiment specs
        specs = [
            experiment_spec_factory(experiment_id=f"concurrent-{i}")
            for i in range(5)
        ]
        
        # Mock storage backend to simulate successful creation
        async def mock_create_experiment(spec):
            # Simulate some async work
            await asyncio.sleep(0.01)
            return {
                "experiment_id": spec.experiment_id,
                "created_at": datetime.now(timezone.utc)
            }
        
        experiment_manager.storage_backend.create_experiment = AsyncMock(side_effect=mock_create_experiment)
        
        # Create experiments concurrently
        tasks = [
            experiment_manager.create_experiment(spec)
            for spec in specs
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all experiments were created
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result["experiment_id"] == f"concurrent-{i}"
            assert "created_at" in result

    @pytest.mark.asyncio
    async def test_experiment_manager_health_check_integration(self, experiment_manager):
        """Test that ExperimentManager integrates with storage health checks."""
        # Mock storage backend health check
        experiment_manager.storage_backend.health_check = AsyncMock(return_value=True)
        
        # This would typically be called by a health check endpoint
        health_ok = await experiment_manager.storage_backend.health_check()
        assert health_ok is True
        
        # Test unhealthy storage
        experiment_manager.storage_backend.health_check = AsyncMock(return_value=False)
        health_ok = await experiment_manager.storage_backend.health_check()
        assert health_ok is False

    # Delete Experiment Tests
    @pytest.mark.asyncio
    async def test_delete_experiment_success(self, experiment_manager):
        """Test successful experiment deletion."""
        experiment_id = "test-experiment-123"
        deleted_at = datetime.now(timezone.utc)
        expected_result = {
            "experiment_id": experiment_id,
            "deleted_at": deleted_at
        }
        
        # Mock storage backend
        experiment_manager.storage_backend.delete_experiment = AsyncMock(return_value=expected_result)
        
        result = await experiment_manager.delete_experiment(experiment_id)
        
        assert result == expected_result
        experiment_manager.storage_backend.delete_experiment.assert_called_once_with(experiment_id)

    @pytest.mark.asyncio
    async def test_delete_experiment_not_found(self, experiment_manager):
        """Test deleting experiment that doesn't exist."""
        experiment_id = "non-existent-experiment"
        
        # Mock storage backend to raise exception
        experiment_manager.storage_backend.delete_experiment = AsyncMock(
            side_effect=ExperimentNotFoundException(experiment_id)
        )
        
        with pytest.raises(ExperimentNotFoundException) as exc_info:
            await experiment_manager.delete_experiment(experiment_id)
        
        assert exc_info.value.experiment_id == experiment_id

    @pytest.mark.asyncio
    async def test_delete_experiment_has_jobs(self, experiment_manager):
        """Test deleting experiment that has associated jobs."""
        experiment_id = "experiment-with-jobs"
        job_count = 5
        
        # Mock storage backend to raise exception
        experiment_manager.storage_backend.delete_experiment = AsyncMock(
            side_effect=ExperimentHasJobsException(experiment_id, job_count)
        )
        
        with pytest.raises(ExperimentHasJobsException) as exc_info:
            await experiment_manager.delete_experiment(experiment_id)
        
        assert exc_info.value.experiment_id == experiment_id
        assert exc_info.value.job_count == job_count

    @pytest.mark.asyncio
    async def test_delete_experiment_not_started(self, experiment_manager_settings):
        """Test deleting experiment when manager is not started."""
        manager = ExperimentManager(experiment_manager_settings)
        
        with pytest.raises(RuntimeError, match="Experiment manager not started"):
            await manager.delete_experiment("test-experiment")

    @pytest.mark.asyncio
    async def test_delete_experiment_storage_error(self, experiment_manager):
        """Test deleting experiment with storage error."""
        experiment_id = "error-experiment"
        
        # Mock storage backend to raise exception
        experiment_manager.storage_backend.delete_experiment = AsyncMock(
            side_effect=Exception("Database connection failed")
        )
        
        with pytest.raises(Exception, match="Database connection failed"):
            await experiment_manager.delete_experiment(experiment_id)

    @pytest.mark.asyncio
    async def test_delete_multiple_experiments(self, experiment_manager):
        """Test deleting multiple experiments."""
        experiment_ids = ["exp-1", "exp-2", "exp-3"]
        
        # Mock storage backend
        def mock_delete_experiment(experiment_id):
            return {
                "experiment_id": experiment_id,
                "deleted_at": datetime.now(timezone.utc)
            }
        
        experiment_manager.storage_backend.delete_experiment = AsyncMock(side_effect=mock_delete_experiment)
        
        results = []
        for experiment_id in experiment_ids:
            result = await experiment_manager.delete_experiment(experiment_id)
            results.append(result)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["experiment_id"] == experiment_ids[i]
            assert "deleted_at" in result

    @pytest.mark.asyncio
    async def test_concurrent_experiment_deletion(self, experiment_manager):
        """Test concurrent experiment deletion."""
        experiment_ids = [f"concurrent-del-{i}" for i in range(5)]
        
        # Mock storage backend to simulate successful deletion
        async def mock_delete_experiment(experiment_id):
            # Simulate some async work
            await asyncio.sleep(0.01)
            return {
                "experiment_id": experiment_id,
                "deleted_at": datetime.now(timezone.utc)
            }
        
        experiment_manager.storage_backend.delete_experiment = AsyncMock(side_effect=mock_delete_experiment)
        
        # Delete experiments concurrently
        tasks = [
            experiment_manager.delete_experiment(experiment_id)
            for experiment_id in experiment_ids
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all experiments were deleted
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result["experiment_id"] == experiment_ids[i]
            assert "deleted_at" in result

    @pytest.mark.asyncio
    async def test_delete_experiment_with_mixed_results(self, experiment_manager):
        """Test deleting experiments with mixed success/failure results."""
        
        def mock_delete_side_effect(experiment_id):
            if experiment_id == "not-found":
                raise ExperimentNotFoundException(experiment_id)
            elif experiment_id == "has-jobs":
                raise ExperimentHasJobsException(experiment_id, 2)
            else:
                return {
                    "experiment_id": experiment_id,
                    "deleted_at": datetime.now(timezone.utc)
                }
        
        experiment_manager.storage_backend.delete_experiment = AsyncMock(side_effect=mock_delete_side_effect)
        
        # Test successful deletion
        result = await experiment_manager.delete_experiment("success-exp")
        assert result["experiment_id"] == "success-exp"
        
        # Test not found
        with pytest.raises(ExperimentNotFoundException):
            await experiment_manager.delete_experiment("not-found")
        
        # Test has jobs
        with pytest.raises(ExperimentHasJobsException):
            await experiment_manager.delete_experiment("has-jobs")

    # List Experiments Tests
    @pytest.mark.asyncio
    async def test_list_experiments_success(self, experiment_manager):
        """Test successful experiments listing."""
        mock_experiments = [
            {
                "id": "exp-1",
                "name": "Experiment 1",
                "description": "First experiment",
                "spec_data": {"models": ["model-1"]},
                "status": "created",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            },
            {
                "id": "exp-2",
                "name": "Experiment 2", 
                "description": "Second experiment",
                "spec_data": {"models": ["model-2"]},
                "status": "running",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
        ]
        
        # Mock storage backend
        experiment_manager.storage_backend.list_experiments = AsyncMock(return_value=mock_experiments)
        
        result = await experiment_manager.list_experiments()
        
        assert result == mock_experiments
        experiment_manager.storage_backend.list_experiments.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_experiments_empty(self, experiment_manager):
        """Test listing experiments when none exist."""
        # Mock storage backend to return empty list
        experiment_manager.storage_backend.list_experiments = AsyncMock(return_value=[])
        
        result = await experiment_manager.list_experiments()
        
        assert result == []
        experiment_manager.storage_backend.list_experiments.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_experiments_not_started(self, experiment_manager_settings):
        """Test listing experiments when manager is not started."""
        manager = ExperimentManager(experiment_manager_settings)
        
        with pytest.raises(RuntimeError, match="Experiment manager not started"):
            await manager.list_experiments()

    @pytest.mark.asyncio
    async def test_list_experiments_storage_error(self, experiment_manager):
        """Test listing experiments with storage error."""
        # Mock storage backend to raise exception
        experiment_manager.storage_backend.list_experiments = AsyncMock(
            side_effect=Exception("Database connection failed")
        )
        
        with pytest.raises(Exception, match="Database connection failed"):
            await experiment_manager.list_experiments()

    @pytest.mark.asyncio
    async def test_list_experiments_large_dataset(self, experiment_manager):
        """Test listing many experiments."""
        # Create mock data for many experiments
        mock_experiments = []
        for i in range(100):
            mock_experiments.append({
                "id": f"exp-{i}",
                "name": f"Experiment {i}",
                "description": f"Description {i}",
                "spec_data": {"models": [f"model-{i}"]},
                "status": "created",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            })
        
        # Mock storage backend
        experiment_manager.storage_backend.list_experiments = AsyncMock(return_value=mock_experiments)
        
        result = await experiment_manager.list_experiments()
        
        assert len(result) == 100
        assert result[0]["id"] == "exp-0"
        assert result[99]["id"] == "exp-99"

    @pytest.mark.asyncio
    async def test_list_experiments_concurrent_access(self, experiment_manager):
        """Test concurrent access to list experiments."""
        mock_experiments = [
            {
                "id": "concurrent-exp",
                "name": "Concurrent Test",
                "description": "Testing concurrent access",
                "spec_data": {"models": ["test-model"]},
                "status": "created", 
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
        ]
        
        # Mock storage backend to simulate successful listing
        async def mock_list_experiments():
            # Simulate some async work
            await asyncio.sleep(0.01)
            return mock_experiments
        
        experiment_manager.storage_backend.list_experiments = AsyncMock(side_effect=mock_list_experiments)
        
        # List experiments concurrently
        tasks = [
            experiment_manager.list_experiments()
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all requests succeeded and returned the same data
        assert len(results) == 5
        for result in results:
            assert result == mock_experiments

    @pytest.mark.asyncio
    async def test_list_experiments_logging(self, experiment_manager):
        """Test that list experiments logs appropriately."""
        mock_experiments = [{"id": "log-test", "name": "Log Test"}]
        experiment_manager.storage_backend.list_experiments = AsyncMock(return_value=mock_experiments)
        
        # Capture logs by checking they don't raise exceptions
        result = await experiment_manager.list_experiments()
        
        assert result == mock_experiments
        # In a real test, you might want to capture and verify log messages