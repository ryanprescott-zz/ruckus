"""Tests for ExperimentManager.get_experiment method."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from ruckus_server.core.experiment_manager import ExperimentManager
from ruckus_server.core.config import ExperimentManagerSettings
from ruckus_server.core.storage.base import ExperimentNotFoundException
from ruckus_common.models import ExperimentSpec


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
def sample_experiment_spec():
    """Create sample experiment spec."""
    from ruckus_common.models import TaskType
    return ExperimentSpec(
        experiment_id="test-experiment-123",
        name="Test Experiment",
        description="A test experiment",
        models=["test-model"],
        task_type=TaskType.SUMMARIZATION,
        tags=["env-test"],
        base_parameters={"param1": "value1"}
    )




class TestGetExperimentSuccess:
    """Test successful get experiment scenarios."""
    
    @pytest.mark.asyncio
    async def test_get_experiment_success(self, experiment_manager, mock_storage_backend, sample_experiment_spec):
        """Test successful experiment retrieval."""
        # Setup
        mock_storage_backend.get_experiment.return_value = sample_experiment_spec
        
        # Execute
        result = await experiment_manager.get_experiment("test-experiment-123")
        
        # Verify
        assert result == sample_experiment_spec
        assert isinstance(result, ExperimentSpec)
        assert result.experiment_id == "test-experiment-123"
        mock_storage_backend.get_experiment.assert_called_once_with("test-experiment-123")
        experiment_manager.logger.info.assert_any_call("Retrieving experiment test-experiment-123")
        experiment_manager.logger.info.assert_any_call("Experiment test-experiment-123 retrieved successfully")
    
    @pytest.mark.asyncio
    async def test_get_experiment_with_complex_data(self, experiment_manager, mock_storage_backend):
        """Test getting experiment with complex data structure."""
        # Setup
        complex_spec = ExperimentSpec(
            experiment_id="complex-exp",
            name="Complex Experiment",
            description="Complex test",
            models=["test-model"],
            task_type=TaskType.GENERATION,
            tags=["type-ml", "framework-pytorch"],
            base_parameters={
                "model": {"layers": [64, 32], "activation": "relu"},
                "training": {"epochs": 100, "lr": 0.001}
            }
        )
        
        mock_storage_backend.get_experiment.return_value = complex_spec
        
        # Execute
        result = await experiment_manager.get_experiment("complex-exp")
        
        # Verify
        assert result == complex_spec
        assert isinstance(result, ExperimentSpec)
        assert result.base_parameters["model"]["layers"] == [64, 32]
        mock_storage_backend.get_experiment.assert_called_once_with("complex-exp")


class TestGetExperimentNotFound:
    """Test experiment not found scenarios."""
    
    @pytest.mark.asyncio
    async def test_get_experiment_not_found(self, experiment_manager, mock_storage_backend):
        """Test getting non-existent experiment."""
        # Setup
        mock_storage_backend.get_experiment.side_effect = ExperimentNotFoundException("nonexistent")
        
        # Execute & Verify
        with pytest.raises(ExperimentNotFoundException) as exc_info:
            await experiment_manager.get_experiment("nonexistent")
        
        assert exc_info.value.experiment_id == "nonexistent"
        mock_storage_backend.get_experiment.assert_called_once_with("nonexistent")
        experiment_manager.logger.info.assert_called_with("Retrieving experiment nonexistent")
        experiment_manager.logger.error.assert_called_with("Experiment not found: Experiment nonexistent not found")
    
    @pytest.mark.asyncio
    async def test_get_experiment_not_found_propagation(self, experiment_manager, mock_storage_backend):
        """Test that ExperimentNotFoundException is properly propagated."""
        # Setup
        exception = ExperimentNotFoundException("missing-exp")
        mock_storage_backend.get_experiment.side_effect = exception
        
        # Execute & Verify
        with pytest.raises(ExperimentNotFoundException) as exc_info:
            await experiment_manager.get_experiment("missing-exp")
        
        # Verify it's the same exception instance
        assert exc_info.value is exception
        assert str(exc_info.value) == "Experiment missing-exp not found"


class TestGetExperimentErrors:
    """Test error scenarios."""
    
    @pytest.mark.asyncio
    async def test_get_experiment_not_started(self):
        """Test getting experiment when manager not started."""
        # Setup
        manager = ExperimentManager()
        manager._started = False
        
        # Execute & Verify
        with pytest.raises(RuntimeError) as exc_info:
            await manager.get_experiment("test-exp")
        
        assert "not started" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_experiment_storage_error(self, experiment_manager, mock_storage_backend):
        """Test getting experiment with storage backend error."""
        # Setup
        mock_storage_backend.get_experiment.side_effect = Exception("Storage connection failed")
        
        # Execute & Verify
        with pytest.raises(Exception) as exc_info:
            await experiment_manager.get_experiment("test-exp")
        
        assert "Storage connection failed" in str(exc_info.value)
        mock_storage_backend.get_experiment.assert_called_once_with("test-exp")
        experiment_manager.logger.info.assert_called_with("Retrieving experiment test-exp")
        experiment_manager.logger.error.assert_called_with("Failed to retrieve experiment test-exp: Storage connection failed")
    
    @pytest.mark.asyncio
    async def test_get_experiment_generic_exception(self, experiment_manager, mock_storage_backend):
        """Test getting experiment with generic exception."""
        # Setup
        mock_storage_backend.get_experiment.side_effect = ValueError("Invalid data format")
        
        # Execute & Verify  
        with pytest.raises(ValueError) as exc_info:
            await experiment_manager.get_experiment("test-exp")
        
        assert "Invalid data format" in str(exc_info.value)
        experiment_manager.logger.error.assert_called_with("Failed to retrieve experiment test-exp: Invalid data format")


class TestGetExperimentValidation:
    """Test input validation and edge cases."""
    
    @pytest.mark.asyncio
    async def test_get_experiment_empty_id(self, experiment_manager, mock_storage_backend):
        """Test getting experiment with empty ID."""
        # Setup
        mock_storage_backend.get_experiment.return_value = None
        
        # Execute
        await experiment_manager.get_experiment("")
        
        # Verify storage was called with empty string
        mock_storage_backend.get_experiment.assert_called_once_with("")
    
    @pytest.mark.asyncio
    async def test_get_experiment_special_characters(self, experiment_manager, mock_storage_backend, sample_experiment_spec):
        """Test getting experiment with special characters in ID."""
        # Setup
        special_id = "exp-123_test.v2"
        sample_experiment_spec.experiment_id = special_id
        mock_storage_backend.get_experiment.return_value = sample_experiment_spec
        
        # Execute
        result = await experiment_manager.get_experiment(special_id)
        
        # Verify
        assert result.experiment_id == special_id
        assert isinstance(result, ExperimentSpec)
        mock_storage_backend.get_experiment.assert_called_once_with(special_id)
    
    @pytest.mark.asyncio
    async def test_get_experiment_unicode_id(self, experiment_manager, mock_storage_backend, sample_experiment_spec):
        """Test getting experiment with unicode characters in ID."""
        # Setup
        unicode_id = "exp-测试-123"
        sample_experiment_spec.experiment_id = unicode_id
        mock_storage_backend.get_experiment.return_value = sample_experiment_spec
        
        # Execute
        result = await experiment_manager.get_experiment(unicode_id)
        
        # Verify
        assert result.experiment_id == unicode_id
        assert isinstance(result, ExperimentSpec)
        mock_storage_backend.get_experiment.assert_called_once_with(unicode_id)


class TestGetExperimentLogging:
    """Test logging behavior."""
    
    @pytest.mark.asyncio
    async def test_get_experiment_logging_success(self, experiment_manager, mock_storage_backend, sample_experiment_spec):
        """Test logging on successful get experiment."""
        # Setup
        sample_experiment_spec.experiment_id = "test-exp"
        mock_storage_backend.get_experiment.return_value = sample_experiment_spec
        
        # Execute
        await experiment_manager.get_experiment("test-exp")
        
        # Verify logging calls
        assert experiment_manager.logger.info.call_count == 2
        experiment_manager.logger.info.assert_any_call("Retrieving experiment test-exp")
        experiment_manager.logger.info.assert_any_call("Experiment test-exp retrieved successfully")
    
    @pytest.mark.asyncio
    async def test_get_experiment_logging_not_found(self, experiment_manager, mock_storage_backend):
        """Test logging on experiment not found."""
        # Setup
        mock_storage_backend.get_experiment.side_effect = ExperimentNotFoundException("missing")
        
        # Execute & Verify
        with pytest.raises(ExperimentNotFoundException):
            await experiment_manager.get_experiment("missing")
        
        # Verify logging
        experiment_manager.logger.info.assert_called_once_with("Retrieving experiment missing")
        experiment_manager.logger.error.assert_called_once_with("Experiment not found: Experiment missing not found")
    
    @pytest.mark.asyncio
    async def test_get_experiment_logging_error(self, experiment_manager, mock_storage_backend):
        """Test logging on generic error."""
        # Setup
        mock_storage_backend.get_experiment.side_effect = Exception("Database error")
        
        # Execute & Verify
        with pytest.raises(Exception):
            await experiment_manager.get_experiment("test-exp")
        
        # Verify logging
        experiment_manager.logger.info.assert_called_once_with("Retrieving experiment test-exp")
        experiment_manager.logger.error.assert_called_once_with("Failed to retrieve experiment test-exp: Database error")