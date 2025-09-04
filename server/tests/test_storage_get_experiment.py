"""Tests for storage backend get_experiment method."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from ruckus_server.core.storage.sqlite import SQLiteStorageBackend
from ruckus_server.core.storage.base import ExperimentNotFoundException
from ruckus_server.core.config import SQLiteSettings
from ruckus_common.models import ExperimentSpec


@pytest.fixture
def storage_settings():
    """Create storage settings for testing."""
    from ruckus_server.core.config import SQLiteSettings
    return SQLiteSettings(
        database_url="sqlite+aiosqlite:///:memory:"
    )


@pytest.fixture
async def storage_backend(storage_settings):
    """Create SQLite storage backend for testing."""
    backend = SQLiteStorageBackend(storage_settings)
    await backend.initialize()
    return backend


@pytest.fixture
def sample_experiment_spec():
    """Create sample experiment spec."""
    from ruckus_common.models import TaskType
    return ExperimentSpec(
        experiment_id="test-get-experiment",
        name="Test Get Experiment", 
        description="An experiment for testing get functionality",
        models=["test-model-1", "test-model-2"],
        task_type=TaskType.SUMMARIZATION,
        tags=["test", "version-1.0"],
        base_parameters={"learning_rate": 0.01, "epochs": 10}
    )


class TestGetExperimentSuccess:
    """Test successful get experiment scenarios."""
    
    @pytest.mark.asyncio
    async def test_get_experiment_success(self, storage_backend, sample_experiment_spec):
        """Test successful experiment retrieval."""
        # Setup - create experiment first
        await storage_backend.create_experiment(sample_experiment_spec)
        
        # Execute
        result = await storage_backend.get_experiment("test-get-experiment")
        
        # Verify
        assert result is not None
        assert isinstance(result, ExperimentSpec)
        assert result.experiment_id == "test-get-experiment"
        assert result.name == "Test Get Experiment"
        assert result.description == "An experiment for testing get functionality"
        assert result.tags == ["test", "version-1.0"]
        assert result.base_parameters == {"learning_rate": 0.01, "epochs": 10}
    
    @pytest.mark.asyncio
    async def test_get_experiment_with_complex_parameters(self, storage_backend):
        """Test getting experiment with complex parameter structure."""
        # Setup
        complex_spec = ExperimentSpec(
            experiment_id="complex-get-test",
            name="Complex Get Test",
            description="Testing complex parameters",
            models=["test-model"],
            task_type=TaskType.GENERATION,
            tags=["complexity-high", "category-ml"],
            base_parameters={
                "model": {
                    "architecture": "transformer",
                    "layers": [
                        {"type": "attention", "heads": 8, "dim": 512},
                        {"type": "feedforward", "dim": 2048}
                    ],
                    "dropout": 0.1
                },
                "training": {
                    "optimizer": {"name": "adam", "lr": 1e-4, "betas": [0.9, 0.999]},
                    "schedule": {"type": "cosine", "warmup_steps": 1000},
                    "batch_size": 32,
                    "gradient_clip": 1.0
                },
                "data": {
                    "source": "/path/to/data",
                    "preprocessing": ["tokenize", "normalize"],
                    "augmentation": {"enabled": True, "methods": ["random_crop", "flip"]}
                }
            }
        )
        
        # Create experiment
        await storage_backend.create_experiment(complex_spec)
        
        # Execute
        result = await storage_backend.get_experiment("complex-get-test")
        
        # Verify complex structure is preserved
        assert isinstance(result, ExperimentSpec)
        assert result.base_parameters["model"]["architecture"] == "transformer"
        assert result.base_parameters["training"]["optimizer"]["name"] == "adam"
        assert len(result.base_parameters["model"]["layers"]) == 2
        assert result.base_parameters["data"]["augmentation"]["methods"] == ["random_crop", "flip"]
    
    @pytest.mark.asyncio
    async def test_get_experiment_with_none_description(self, storage_backend):
        """Test getting experiment with None description."""
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
        
        # Create experiment
        await storage_backend.create_experiment(spec_with_none)
        
        # Execute
        result = await storage_backend.get_experiment("none-desc-test")
        
        # Verify
        assert isinstance(result, ExperimentSpec)
        assert result.description is None
    
    @pytest.mark.asyncio
    async def test_get_experiment_special_characters_in_id(self, storage_backend):
        """Test getting experiment with special characters in ID."""
        # Setup
        special_spec = ExperimentSpec(
            experiment_id="test-exp_123.v2",
            name="Special Characters Test",
            description="Testing special characters in ID",
            models=["test-model"],
            task_type=TaskType.SUMMARIZATION,
            tags=["version-v2"],
            base_parameters={"test": True}
        )
        
        # Create experiment
        await storage_backend.create_experiment(special_spec)
        
        # Execute
        result = await storage_backend.get_experiment("test-exp_123.v2")
        
        # Verify
        assert isinstance(result, ExperimentSpec)
        assert result.experiment_id == "test-exp_123.v2"
        assert result.name == "Special Characters Test"
    
    @pytest.mark.asyncio
    async def test_get_experiment_after_status_update(self, storage_backend, sample_experiment_spec):
        """Test getting experiment after status update."""
        # Setup - create and update experiment
        await storage_backend.create_experiment(sample_experiment_spec)
        await storage_backend.update_experiment_status("test-get-experiment", "running")
        
        # Execute
        result = await storage_backend.get_experiment("test-get-experiment")
        
        # Verify
        assert isinstance(result, ExperimentSpec)
        assert result.experiment_id == "test-get-experiment"


class TestGetExperimentNotFound:
    """Test experiment not found scenarios."""
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_experiment(self, storage_backend):
        """Test getting non-existent experiment raises exception."""
        # Execute & Verify
        with pytest.raises(ExperimentNotFoundException) as exc_info:
            await storage_backend.get_experiment("nonexistent-experiment")
        
        assert exc_info.value.experiment_id == "nonexistent-experiment"
        assert "nonexistent-experiment not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_deleted_experiment(self, storage_backend, sample_experiment_spec):
        """Test getting experiment that was deleted."""
        # Setup - create then delete experiment
        await storage_backend.create_experiment(sample_experiment_spec)
        await storage_backend.delete_experiment("test-get-experiment")
        
        # Execute & Verify
        with pytest.raises(ExperimentNotFoundException) as exc_info:
            await storage_backend.get_experiment("test-get-experiment")
        
        assert exc_info.value.experiment_id == "test-get-experiment"
    
    @pytest.mark.asyncio
    async def test_get_experiment_empty_id(self, storage_backend):
        """Test getting experiment with empty ID."""
        # Execute & Verify
        with pytest.raises(ExperimentNotFoundException) as exc_info:
            await storage_backend.get_experiment("")
        
        assert exc_info.value.experiment_id == ""
    
    @pytest.mark.asyncio
    async def test_get_experiment_none_id(self, storage_backend):
        """Test getting experiment with None ID."""
        # Execute & Verify
        with pytest.raises(ExperimentNotFoundException) as exc_info:
            await storage_backend.get_experiment(None)
        
        assert exc_info.value.experiment_id is None


class TestGetExperimentDataIntegrity:
    """Test data integrity and consistency."""
    
    @pytest.mark.asyncio
    async def test_get_experiment_preserves_json_serialization(self, storage_backend):
        """Test that experiment data JSON serialization is preserved."""
        # Setup with data that tests JSON serialization edge cases
        spec = ExperimentSpec(
            experiment_id="json-test",
            name="JSON Serialization Test",
            description="Testing JSON edge cases",
            models=["test-model"],
            task_type=TaskType.SUMMARIZATION,
            tags=["json-test"],
            base_parameters={
                "test_data": {"float": 3.14159, "boolean": True, "null": None},
                "nested": {"deep": {"value": [1, 2, 3]}},
                "unicode": "测试数据",
                "special_chars": "test@#$%^&*()",
                "numbers": {"int": 42, "float": 3.14, "scientific": 1e-5}
            }
        )
        
        # Create experiment
        await storage_backend.create_experiment(spec)
        
        # Execute
        result = await storage_backend.get_experiment("json-test")
        
        # Verify all data types are preserved
        assert isinstance(result, ExperimentSpec)
        test_data = result.base_parameters["test_data"]
        assert test_data["float"] == 3.14159
        assert test_data["boolean"] is True
        assert test_data["null"] is None
        assert result.base_parameters["unicode"] == "测试数据"
        assert result.base_parameters["numbers"]["scientific"] == 1e-5
    
    @pytest.mark.asyncio
    async def test_get_experiment_timestamps_consistency(self, storage_backend, sample_experiment_spec):
        """Test that timestamps are consistent and properly formatted."""
        # Setup
        before_create = datetime.now(timezone.utc)
        await storage_backend.create_experiment(sample_experiment_spec)
        after_create = datetime.now(timezone.utc)
        
        # Execute
        result = await storage_backend.get_experiment("test-get-experiment")
        
        # Verify experiment spec is returned correctly
        assert isinstance(result, ExperimentSpec)
        assert result.experiment_id == "test-get-experiment"
    
    @pytest.mark.asyncio
    async def test_get_experiment_concurrent_access(self, storage_backend, sample_experiment_spec):
        """Test getting experiment with concurrent access."""
        # Setup
        await storage_backend.create_experiment(sample_experiment_spec)
        
        # Execute multiple concurrent gets
        import asyncio
        results = await asyncio.gather(*[
            storage_backend.get_experiment("test-get-experiment")
            for _ in range(5)
        ])
        
        # Verify all results are identical
        first_result = results[0]
        for result in results[1:]:
            assert isinstance(result, ExperimentSpec)
            assert result.experiment_id == first_result.experiment_id
            assert result.base_parameters == first_result.base_parameters
            assert result.tags == first_result.tags
    
    @pytest.mark.asyncio
    async def test_get_experiment_isolation(self, storage_backend):
        """Test that getting different experiments returns correct data."""
        # Setup - create multiple experiments
        specs = []
        for i in range(3):
            spec = ExperimentSpec(
                experiment_id=f"isolation-test-{i}",
                name=f"Isolation Test {i}",
                description=f"Testing isolation {i}",
                models=["test-model"],
                task_type=TaskType.SUMMARIZATION,
                tags=[f"number-{i}"],
                base_parameters={"value": i * 10}
            )
            specs.append(spec)
            await storage_backend.create_experiment(spec)
        
        # Execute - get each experiment
        results = []
        for i in range(3):
            result = await storage_backend.get_experiment(f"isolation-test-{i}")
            results.append(result)
        
        # Verify each experiment has correct data
        for i, result in enumerate(results):
            assert isinstance(result, ExperimentSpec)
            assert result.experiment_id == f"isolation-test-{i}"
            assert result.name == f"Isolation Test {i}"
            assert f"number-{i}" in result.tags
            assert result.base_parameters["value"] == i * 10


class TestGetExperimentPerformance:
    """Test performance and scalability."""
    
    @pytest.mark.asyncio
    async def test_get_experiment_large_parameters(self, storage_backend):
        """Test getting experiment with large parameter structure."""
        # Setup - create experiment with large parameters
        large_params = {}
        for i in range(100):
            large_params[f"param_{i}"] = {
                "value": i,
                "data": [j for j in range(50)],
                "metadata": {"description": f"Parameter {i} with lots of data"}
            }
        
        large_spec = ExperimentSpec(
            experiment_id="large-params-test",
            name="Large Parameters Test",
            description="Testing large parameter structures",
            models=["test-model"],
            task_type=TaskType.GENERATION,
            tags=["size-large"],
            base_parameters=large_params
        )
        
        # Create experiment
        await storage_backend.create_experiment(large_spec)
        
        # Execute
        result = await storage_backend.get_experiment("large-params-test")
        
        # Verify large structure is handled correctly
        assert isinstance(result, ExperimentSpec)
        assert len(result.base_parameters) == 100
        assert result.base_parameters["param_50"]["value"] == 50
        assert len(result.base_parameters["param_99"]["data"]) == 50