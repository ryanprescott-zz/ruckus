"""Tests for model discovery functionality."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, patch

from ruckus_agent.utils.model_discovery import ModelDiscovery
from ruckus_agent.core.models import ModelInfo


class TestModelDiscovery:
    """Test model discovery functionality."""
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create a temporary models directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_huggingface_model(self, temp_models_dir):
        """Create a mock HuggingFace model structure."""
        model_dir = temp_models_dir / "test-model"
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
        
        # Create tokenizer_config.json
        tokenizer_config = {
            "tokenizer_class": "LlamaTokenizer",
            "vocab_size": 32000
        }
        with open(model_dir / "tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f)
        
        # Create model files
        (model_dir / "pytorch_model.bin").write_text("fake model weights")
        (model_dir / "tokenizer.model").write_text("fake tokenizer")
        (model_dir / "tokenizer.json").write_text("{}")
        
        return model_dir
    
    @pytest.fixture
    def mock_safetensors_model(self, temp_models_dir):
        """Create a mock SafeTensors model structure."""
        model_dir = temp_models_dir / "safetensors-model"
        model_dir.mkdir()
        
        # Create config.json
        config = {
            "model_type": "mistral",
            "architectures": ["MistralForCausalLM"],
            "vocab_size": 32000,
            "hidden_size": 4096,
            "torch_dtype": "bfloat16"
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)
        
        # Create safetensors files
        (model_dir / "model-00001-of-00002.safetensors").write_text("fake safetensors")
        (model_dir / "model-00002-of-00002.safetensors").write_text("fake safetensors")
        
        return model_dir
    
    @pytest.fixture
    def mock_quantized_model(self, temp_models_dir):
        """Create a mock quantized model structure."""
        model_dir = temp_models_dir / "quantized-model"
        model_dir.mkdir()
        
        # Create config.json with quantization config
        config = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "quantization_config": {
                "quant_method": "gptq",
                "bits": 4
            }
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)
        
        # Create GPTQ files
        (model_dir / "gptq_model-4bit-128g.safetensors").write_text("quantized weights")
        
        return model_dir
    
    @pytest.mark.asyncio
    async def test_discover_single_model(self, temp_models_dir, mock_huggingface_model):
        """Test discovering a single HuggingFace model."""
        discovery = ModelDiscovery(str(temp_models_dir))
        
        models = await discovery.discover_all_models()
        
        assert len(models) == 1
        model = models[0]
        
        assert model.name == "test-model"
        assert model.path == str(mock_huggingface_model)
        assert model.model_type == "llama"
        assert model.architecture == "LlamaForCausalLM"
        assert model.vocab_size == 32000
        assert model.hidden_size == 4096
        assert model.num_layers == 32
        assert model.num_attention_heads == 32
        assert model.max_position_embeddings == 2048
        assert model.torch_dtype == "float16"
        assert model.tokenizer_type == "LlamaTokenizer"
        assert model.tokenizer_vocab_size == 32000
        assert model.format == "pytorch"
        assert "vllm" in model.framework_compatible
        assert "transformers" in model.framework_compatible
        assert model.size_gb > 0  # Should calculate file sizes
        
        # Check file categorization
        assert "config.json" in model.config_files
        assert "pytorch_model.bin" in model.model_files
        assert "tokenizer.model" in model.tokenizer_files
    
    @pytest.mark.asyncio
    async def test_discover_multiple_models(self, temp_models_dir, mock_huggingface_model, mock_safetensors_model):
        """Test discovering multiple models."""
        discovery = ModelDiscovery(str(temp_models_dir))
        
        models = await discovery.discover_all_models()
        
        assert len(models) == 2
        model_names = {model.name for model in models}
        assert "test-model" in model_names
        assert "safetensors-model" in model_names
        
        # Check safetensors model specifically
        safetensors_model = next(m for m in models if m.name == "safetensors-model")
        assert safetensors_model.format == "safetensors"
        assert safetensors_model.model_type == "mistral"
        assert safetensors_model.torch_dtype == "bfloat16"
    
    @pytest.mark.asyncio
    async def test_quantization_detection(self, temp_models_dir, mock_quantized_model):
        """Test detection of quantized models."""
        discovery = ModelDiscovery(str(temp_models_dir))
        
        models = await discovery.discover_all_models()
        
        assert len(models) == 1
        model = models[0]
        
        assert model.quantization == "gptq"
        assert "gptq_model-4bit-128g.safetensors" in model.model_files
    
    @pytest.mark.asyncio
    async def test_empty_directory(self, temp_models_dir):
        """Test discovery in empty directory."""
        discovery = ModelDiscovery(str(temp_models_dir))
        
        models = await discovery.discover_all_models()
        
        assert len(models) == 0
    
    @pytest.mark.asyncio
    async def test_nonexistent_directory(self):
        """Test discovery with nonexistent directory."""
        discovery = ModelDiscovery("/nonexistent/path")
        
        models = await discovery.discover_all_models()
        
        assert len(models) == 0
    
    @pytest.mark.asyncio
    async def test_invalid_model_directory(self, temp_models_dir):
        """Test handling of directories without config.json."""
        # Create directory without config.json
        invalid_dir = temp_models_dir / "invalid-model"
        invalid_dir.mkdir()
        (invalid_dir / "some_file.txt").write_text("not a model")
        
        discovery = ModelDiscovery(str(temp_models_dir))
        
        models = await discovery.discover_all_models()
        
        assert len(models) == 0
    
    @pytest.mark.asyncio
    async def test_corrupted_config_handling(self, temp_models_dir):
        """Test handling of corrupted config.json files."""
        model_dir = temp_models_dir / "corrupted-model"
        model_dir.mkdir()
        
        # Create invalid JSON
        with open(model_dir / "config.json", "w") as f:
            f.write("invalid json {")
        
        discovery = ModelDiscovery(str(temp_models_dir))
        
        models = await discovery.discover_all_models()
        
        assert len(models) == 0  # Should handle gracefully
    
    def test_determine_model_format(self, temp_models_dir):
        """Test model format determination."""
        discovery = ModelDiscovery(str(temp_models_dir))
        
        # Test different formats
        assert discovery._determine_model_format(["model.safetensors"]) == "safetensors"
        assert discovery._determine_model_format(["pytorch_model.bin"]) == "pytorch"
        assert discovery._determine_model_format(["model.gguf"]) == "gguf"
        assert discovery._determine_model_format(["model.pt"]) == "pytorch"
        assert discovery._determine_model_format(["random.txt"]) == "unknown"
        
        # Test priority (safetensors over pytorch)
        assert discovery._determine_model_format(["model.bin", "model.safetensors"]) == "safetensors"
    
    def test_framework_compatibility(self, temp_models_dir):
        """Test framework compatibility determination."""
        discovery = ModelDiscovery(str(temp_models_dir))
        
        # Test vLLM compatible model
        llama_config = {"model_type": "llama"}
        frameworks = discovery._determine_compatible_frameworks("safetensors", llama_config)
        assert "vllm" in frameworks
        assert "transformers" in frameworks
        assert "pytorch" in frameworks
        
        # Test non-vLLM compatible model
        unknown_config = {"model_type": "unknown_architecture"}
        frameworks = discovery._determine_compatible_frameworks("safetensors", unknown_config)
        assert "vllm" not in frameworks
        assert "transformers" in frameworks
        
        # Test GGUF model
        frameworks = discovery._determine_compatible_frameworks("gguf", {})
        assert "llamacpp" in frameworks
    
    def test_quantization_detection_methods(self, temp_models_dir):
        """Test different quantization detection methods."""
        discovery = ModelDiscovery(str(temp_models_dir))
        
        # Test GPTQ detection from filename
        assert discovery._detect_quantization(["model_gptq.safetensors"], {}) == "gptq"
        
        # Test AWQ detection
        assert discovery._detect_quantization(["model_awq.safetensors"], {}) == "awq"
        
        # Test GGUF quantization
        assert discovery._detect_quantization(["model_q4_0.gguf"], {}).startswith("gguf_")
        
        # Test config-based quantization
        config_with_quant = {
            "quantization_config": {
                "quant_method": "bitsandbytes"
            }
        }
        assert discovery._detect_quantization([], config_with_quant) == "bitsandbytes"
        
        # Test no quantization
        assert discovery._detect_quantization(["model.safetensors"], {}) is None
    
    @pytest.mark.asyncio
    async def test_file_size_calculation(self, temp_models_dir, mock_huggingface_model):
        """Test that file sizes are calculated correctly."""
        discovery = ModelDiscovery(str(temp_models_dir))
        
        models = await discovery.discover_all_models()
        
        assert len(models) == 1
        model = models[0]
        
        # Should have non-zero size
        assert model.size_gb > 0
        
        # Size should be reasonable (our test files are very small)
        assert model.size_gb < 0.001  # Less than 1MB in GB
    
    @pytest.mark.asyncio
    async def test_timestamp_fields(self, temp_models_dir, mock_huggingface_model):
        """Test that timestamp fields are populated."""
        discovery = ModelDiscovery(str(temp_models_dir))
        
        before_discovery = datetime.utcnow()
        models = await discovery.discover_all_models()
        after_discovery = datetime.utcnow()
        
        assert len(models) == 1
        model = models[0]
        
        # Check discovered_at timestamp
        assert before_discovery <= model.discovered_at <= after_discovery