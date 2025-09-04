"""Tests for ExperimentManager.list_experiments method."""

import pytest
import asyncio
from unittest.mock import AsyncMock

from ruckus_server.core.experiment_manager import ExperimentManager
from ruckus_server.core.config import ExperimentManagerSettings
from ruckus_common.models import ExperimentSpec, TaskType


@pytest.fixture
def mock_storage_backend():
    """Create mock storage backend."""
    return AsyncMock()


@pytest.fixture
def experiment_manager(mock_storage_backend):
    """Create experiment manager with mock storage."""
    settings = ExperimentManagerSettings()
    manager = ExperimentManager(settings)
    manager.storage_backend = mock_storage_backend
    manager._started = True
    manager.logger = AsyncMock()
    return manager


@pytest.fixture
def sample_experiment_specs():
    """Create sample experiment specs."""
    return [
        ExperimentSpec(
            experiment_id="test-experiment-1",
            name="Test Experiment 1",
            description="First test experiment",
            models=["test-model-1"],
            task_type=TaskType.SUMMARIZATION,
            tags=["type-test", "version-1.0"],
            base_parameters={"learning_rate": 0.01, "epochs": 10}
        ),
        ExperimentSpec(
            experiment_id="test-experiment-2",
            name="Test Experiment 2",
            description="Second test experiment",
            models=["test-model-2"],
            task_type=TaskType.CLASSIFICATION,
            tags=["type-production", "version-2.0"],
            base_parameters={"learning_rate": 0.001, "epochs": 20}
        )
    ]


class TestListExperimentsSuccess:
    """Test successful list experiments scenarios."""
    
    @pytest.mark.asyncio
    async def test_list_experiments_success(self, experiment_manager, mock_storage_backend, sample_experiment_specs):
        """Test successful experiments listing."""
        # Setup
        mock_storage_backend.list_experiments.return_value = sample_experiment_specs
        
        # Execute
        result = await experiment_manager.list_experiments()
        
        # Verify
        assert result == sample_experiment_specs
        assert len(result) == 2
        assert all(isinstance(spec, ExperimentSpec) for spec in result)
        assert result[0].experiment_id == "test-experiment-1"
        assert result[1].experiment_id == "test-experiment-2"
        
        mock_storage_backend.list_experiments.assert_called_once()
        experiment_manager.logger.info.assert_any_call("Listing all experiments")
        experiment_manager.logger.info.assert_any_call("Found 2 experiments")
    
    @pytest.mark.asyncio
    async def test_list_experiments_empty(self, experiment_manager, mock_storage_backend):
        """Test listing when no experiments exist."""
        # Setup
        mock_storage_backend.list_experiments.return_value = []
        
        # Execute
        result = await experiment_manager.list_experiments()
        
        # Verify
        assert result == []
        assert len(result) == 0
        
        mock_storage_backend.list_experiments.assert_called_once()
        experiment_manager.logger.info.assert_any_call("Listing all experiments")
        experiment_manager.logger.info.assert_any_call("Found 0 experiments")
    
    @pytest.mark.asyncio
    async def test_list_experiments_complex_data(self, experiment_manager, mock_storage_backend):
        """Test listing experiments with complex data structures."""
        # Setup
        complex_spec = ExperimentSpec(
            experiment_id="complex-experiment",
            name="Complex Experiment",
            description="An experiment with complex parameters",
            models=["transformer-model"],
            task_type=TaskType.GENERATION,
            tags=["complexity-high", "framework-pytorch", "version-1.5"],
            base_parameters={
                "model": {
                    "architecture": "transformer",
                    "layers": [64, 32, 16],
                    "config": {"dropout": 0.1, "activation": "relu"}
                },
                "training": {
                    "optimizer": {"name": "adam", "lr": 1e-4},
                    "schedule": {"type": "cosine", "warmup": 1000},
                    "epochs": 100
                },
                "data": {
                    "source": "/path/to/data",
                    "preprocessing": ["normalize", "tokenize"],
                    "augmentation": True
                }
            }
        )
        
        mock_storage_backend.list_experiments.return_value = [complex_spec]
        
        # Execute
        result = await experiment_manager.list_experiments()
        
        # Verify
        assert len(result) == 1
        assert isinstance(result[0], ExperimentSpec)
        spec = result[0]
        assert spec.experiment_id == "complex-experiment"
        assert spec.base_parameters["model"]["architecture"] == "transformer"
        assert spec.base_parameters["training"]["optimizer"]["name"] == "adam"
        assert len(spec.base_parameters["model"]["layers"]) == 3
        assert "framework-pytorch" in spec.tags
    
    @pytest.mark.asyncio
    async def test_list_experiments_large_dataset(self, experiment_manager, mock_storage_backend):
        """Test listing many experiments."""
        # Setup
        large_experiment_list = []
        for i in range(100):
            spec = ExperimentSpec(
                experiment_id=f"experiment-{i}",
                name=f"Experiment {i}",
                description=f"Test experiment number {i}",
                models=[f"model-{i}"],
                task_type=TaskType.SUMMARIZATION,
                tags=[f"index-{i}", "batch-large_test"],
                base_parameters={"value": i * 10, "active": i % 2 == 0}
            )
            large_experiment_list.append(spec)
        
        mock_storage_backend.list_experiments.return_value = large_experiment_list
        
        # Execute
        result = await experiment_manager.list_experiments()
        
        # Verify
        assert len(result) == 100
        assert all(isinstance(spec, ExperimentSpec) for spec in result)
        assert result[0].experiment_id == "experiment-0"
        assert result[99].experiment_id == "experiment-99"
        assert result[50].base_parameters["value"] == 500
        
        experiment_manager.logger.info.assert_any_call("Found 100 experiments")


class TestListExperimentsErrors:
    """Test error scenarios."""
    
    @pytest.mark.asyncio
    async def test_list_experiments_not_started(self):
        """Test listing when manager is not started."""
        # Setup
        manager = ExperimentManager()
        manager._started = False
        
        # Execute & Verify
        with pytest.raises(RuntimeError) as exc_info:
            await manager.list_experiments()
        
        assert "not started" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_list_experiments_storage_error(self, experiment_manager, mock_storage_backend):
        """Test listing with storage backend error."""
        # Setup
        mock_storage_backend.list_experiments.side_effect = Exception("Database connection failed")
        
        # Execute & Verify
        with pytest.raises(Exception) as exc_info:
            await experiment_manager.list_experiments()
        
        assert "Database connection failed" in str(exc_info.value)
        mock_storage_backend.list_experiments.assert_called_once()
        experiment_manager.logger.info.assert_called_with("Listing all experiments")
        experiment_manager.logger.error.assert_called_with("Failed to list experiments: Database connection failed")
    
    @pytest.mark.asyncio
    async def test_list_experiments_generic_exception(self, experiment_manager, mock_storage_backend):
        """Test listing with generic exception."""
        # Setup
        mock_storage_backend.list_experiments.side_effect = ValueError("Invalid data format")
        
        # Execute & Verify
        with pytest.raises(ValueError) as exc_info:
            await experiment_manager.list_experiments()
        
        assert "Invalid data format" in str(exc_info.value)
        experiment_manager.logger.error.assert_called_with("Failed to list experiments: Invalid data format")


class TestListExperimentsEdgeCases:
    """Test edge cases and data validation."""
    
    @pytest.mark.asyncio
    async def test_list_experiments_with_none_descriptions(self, experiment_manager, mock_storage_backend):
        """Test listing experiments with None descriptions."""
        # Setup
        spec_with_none = ExperimentSpec(
            experiment_id="none-desc-test",
            name="None Description Test",
            description=None,
            models=["test-model"],
            task_type=TaskType.SUMMARIZATION,
            tags=[],
            base_parameters={}
        )
        
        mock_storage_backend.list_experiments.return_value = [spec_with_none]
        
        # Execute
        result = await experiment_manager.list_experiments()
        
        # Verify
        assert len(result) == 1
        assert isinstance(result[0], ExperimentSpec)
        assert result[0].description is None
    
    @pytest.mark.asyncio
    async def test_list_experiments_with_empty_containers(self, experiment_manager, mock_storage_backend):
        """Test listing experiments with empty tags and parameters."""
        # Setup
        empty_spec = ExperimentSpec(
            experiment_id="empty-test",
            name="Empty Test",
            description="Testing empty containers",
            models=["test-model"],
            task_type=TaskType.SUMMARIZATION,
            tags=[],
            base_parameters={}
        )
        
        mock_storage_backend.list_experiments.return_value = [empty_spec]
        
        # Execute
        result = await experiment_manager.list_experiments()
        
        # Verify
        assert len(result) == 1
        spec = result[0]
        assert spec.tags == []
        assert spec.base_parameters == {}
    
    @pytest.mark.asyncio
    async def test_list_experiments_mixed_data_types(self, experiment_manager, mock_storage_backend):
        """Test listing experiments with various data types."""
        # Setup
        spec_with_types = ExperimentSpec(
            experiment_id="types-test",
            name="Data Types Test",
            description="Testing various data types",
            models=["test-model"],
            task_type=TaskType.SUMMARIZATION,
            tags=["string-test", "integer-42", "float-3.14159", "boolean-True", "null-None"],
            base_parameters={
                "nested": {"deep": {"value": [1, 2, 3]}},
                "unicode": "测试数据",
                "scientific": 1e-5,
                "negative": -100
            }
        )
        
        mock_storage_backend.list_experiments.return_value = [spec_with_types]
        
        # Execute
        result = await experiment_manager.list_experiments()
        
        # Verify
        assert len(result) == 1
        spec = result[0]
        assert spec.tags["integer"] == 42
        assert spec.tags["float"] == 3.14159
        assert spec.tags["boolean"] is True
        assert spec.tags["null"] is None
        assert spec.parameters["unicode"] == "测试数据"
        assert spec.parameters["scientific"] == 1e-5


class TestListExperimentsPerformance:
    """Test performance and concurrency scenarios."""
    
    @pytest.mark.asyncio
    async def test_list_experiments_concurrent_access(self, experiment_manager, mock_storage_backend, sample_experiment_specs):
        """Test concurrent access to list experiments."""
        # Setup
        mock_storage_backend.list_experiments.return_value = sample_experiment_specs
        
        # Execute multiple concurrent calls
        tasks = [experiment_manager.list_experiments() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # Verify
        assert len(results) == 5
        for result in results:
            assert result == sample_experiment_specs
            assert len(result) == 2
            assert all(isinstance(spec, ExperimentSpec) for spec in result)
        
        # Verify storage was called for each request
        assert mock_storage_backend.list_experiments.call_count == 5
    
    @pytest.mark.asyncio
    async def test_list_experiments_repeated_calls(self, experiment_manager, mock_storage_backend, sample_experiment_specs):
        """Test repeated calls to list experiments."""
        # Setup
        mock_storage_backend.list_experiments.return_value = sample_experiment_specs
        
        # Execute multiple sequential calls
        results = []
        for _ in range(5):
            result = await experiment_manager.list_experiments()
            results.append(result)
        
        # Verify
        assert len(results) == 5
        for result in results:
            assert result == sample_experiment_specs
            assert all(isinstance(spec, ExperimentSpec) for spec in result)
        
        # Verify storage was called for each request
        assert mock_storage_backend.list_experiments.call_count == 5


class TestListExperimentsLogging:
    """Test logging behavior."""
    
    @pytest.mark.asyncio
    async def test_list_experiments_logging_success(self, experiment_manager, mock_storage_backend, sample_experiment_specs):
        """Test logging on successful list experiments."""
        # Setup
        mock_storage_backend.list_experiments.return_value = sample_experiment_specs
        
        # Execute
        await experiment_manager.list_experiments()
        
        # Verify logging calls
        assert experiment_manager.logger.info.call_count >= 2
        experiment_manager.logger.info.assert_any_call("Listing all experiments")
        experiment_manager.logger.info.assert_any_call("Found 2 experiments")
    
    @pytest.mark.asyncio
    async def test_list_experiments_logging_empty(self, experiment_manager, mock_storage_backend):
        """Test logging when no experiments found."""
        # Setup
        mock_storage_backend.list_experiments.return_value = []
        
        # Execute
        await experiment_manager.list_experiments()
        
        # Verify logging
        experiment_manager.logger.info.assert_any_call("Listing all experiments")
        experiment_manager.logger.info.assert_any_call("Found 0 experiments")
    
    @pytest.mark.asyncio
    async def test_list_experiments_logging_error(self, experiment_manager, mock_storage_backend):
        """Test logging on error."""
        # Setup
        mock_storage_backend.list_experiments.side_effect = Exception("Test error")
        
        # Execute & Verify
        with pytest.raises(Exception):
            await experiment_manager.list_experiments()
        
        # Verify logging
        experiment_manager.logger.info.assert_called_with("Listing all experiments")
        experiment_manager.logger.error.assert_called_with("Failed to list experiments: Test error")