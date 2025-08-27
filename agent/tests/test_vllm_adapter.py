"""Tests for VLLM adapter functionality."""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from ruckus_agent.adapters.vllm_adapter import VLLMAdapter
from ruckus_agent.core.models import ModelInfo


class TestVLLMAdapter:
    """Test VLLM adapter functionality."""
    
    @pytest.fixture
    def vllm_adapter(self):
        """Create a VLLM adapter for testing."""
        return VLLMAdapter()
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create a temporary models directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_model_info(self):
        """Create a mock ModelInfo object."""
        return ModelInfo(
            name="test-llama-7b",
            path="/ruckus/models/test-llama-7b",
            size_gb=13.5,
            format="safetensors",
            framework_compatible=["vllm", "transformers", "pytorch"],
            model_type="llama",
            architecture="LlamaForCausalLM",
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            max_position_embeddings=2048,
            torch_dtype="float16",
            tokenizer_type="LlamaTokenizer",
            tokenizer_vocab_size=32000,
            config_files=["config.json"],
            model_files=["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors"],
            tokenizer_files=["tokenizer.json", "tokenizer.model"],
            discovered_at=datetime.utcnow()
        )
    
    @pytest.fixture
    def mock_discovered_models(self, mock_model_info):
        """Mock discovered models list."""
        return [mock_model_info]
    
    def test_vllm_adapter_initialization(self, vllm_adapter):
        """Test VLLM adapter initialization."""
        assert vllm_adapter.engine is None
        assert vllm_adapter.model_name is None
        assert vllm_adapter.model_path is None
        assert vllm_adapter.model_info is None
        assert vllm_adapter.discovery is not None
    
    @pytest.mark.asyncio
    async def test_load_model_success(self, vllm_adapter, mock_discovered_models, mock_model_info):
        """Test successful model loading."""
        model_name = "test-llama-7b"
        
        # Mock the discovery system
        with patch.object(vllm_adapter.discovery, 'discover_all_models', return_value=mock_discovered_models):
            # Mock vLLM components
            mock_engine = MagicMock()
            mock_llm = MagicMock()
            mock_sampling_params = MagicMock()
            mock_engine_args = MagicMock()
            mock_async_llm_engine = MagicMock()
            mock_async_llm_engine.from_engine_args.return_value = mock_engine
            
            # Mock all the imports at once
            with patch.dict('sys.modules', {
                'vllm': mock_llm,
                'vllm.engine': MagicMock(),
                'vllm.engine.arg_utils': MagicMock(AsyncEngineArgs=mock_engine_args),
                'vllm.engine.async_llm_engine': MagicMock(AsyncLLMEngine=mock_async_llm_engine)
            }):
                await vllm_adapter.load_model(model_name)
                
                assert vllm_adapter.model_name == model_name
                assert vllm_adapter.model_path == mock_model_info.path
                assert vllm_adapter.model_info == mock_model_info
                assert vllm_adapter.engine == mock_engine
    
    @pytest.mark.asyncio
    async def test_load_model_not_found(self, vllm_adapter):
        """Test loading a model that doesn't exist in discovery."""
        model_name = "nonexistent-model"
        
        # Mock empty discovery results
        with patch.object(vllm_adapter.discovery, 'discover_all_models', return_value=[]):
            with pytest.raises(ValueError, match="Model nonexistent-model not found"):
                await vllm_adapter.load_model(model_name)
    
    @pytest.mark.asyncio
    async def test_load_model_vllm_incompatible(self, vllm_adapter, mock_model_info):
        """Test loading a model that's not compatible with vLLM."""
        # Create incompatible model
        incompatible_model = mock_model_info.model_copy()
        incompatible_model.framework_compatible = ["transformers", "pytorch"]  # No vLLM
        
        model_name = "test-llama-7b"
        
        with patch.object(vllm_adapter.discovery, 'discover_all_models', return_value=[incompatible_model]):
            # Mock vLLM components
            mock_engine = MagicMock()
            mock_llm = MagicMock()
            mock_engine_args = MagicMock()
            mock_async_llm_engine = MagicMock()
            mock_async_llm_engine.from_engine_args.return_value = mock_engine
            
            with patch.dict('sys.modules', {
                'vllm': mock_llm,
                'vllm.engine': MagicMock(),
                'vllm.engine.arg_utils': MagicMock(AsyncEngineArgs=mock_engine_args),
                'vllm.engine.async_llm_engine': MagicMock(AsyncLLMEngine=mock_async_llm_engine)
            }):
                # Should load but with warning (we test warning in logs)
                await vllm_adapter.load_model(model_name)
                
                assert vllm_adapter.model_name == model_name
                assert vllm_adapter.engine == mock_engine
    
    @pytest.mark.asyncio
    async def test_load_model_vllm_import_error(self, vllm_adapter, mock_discovered_models):
        """Test loading model when vLLM is not available."""
        model_name = "test-llama-7b"
        
        with patch.object(vllm_adapter.discovery, 'discover_all_models', return_value=mock_discovered_models):
            # Mock ImportError during the import statement inside load_model
            with patch('builtins.__import__', side_effect=ImportError("No module named 'vllm'")):
                with pytest.raises(ImportError, match="vLLM not available"):
                    await vllm_adapter.load_model(model_name)
                
                # Ensure cleanup on failure
                assert vllm_adapter.engine is None
                assert vllm_adapter.model_name is None
    
    @pytest.mark.asyncio
    async def test_load_model_vllm_runtime_error(self, vllm_adapter, mock_discovered_models):
        """Test loading model when vLLM engine fails to initialize."""
        model_name = "test-llama-7b"
        
        with patch.object(vllm_adapter.discovery, 'discover_all_models', return_value=mock_discovered_models):
            # Mock vLLM components with engine initialization failure
            mock_llm = MagicMock()
            mock_engine_args = MagicMock()
            mock_async_llm_engine = MagicMock()
            mock_async_llm_engine.from_engine_args.side_effect = RuntimeError("CUDA out of memory")
            
            with patch.dict('sys.modules', {
                'vllm': mock_llm,
                'vllm.engine': MagicMock(),
                'vllm.engine.arg_utils': MagicMock(AsyncEngineArgs=mock_engine_args),
                'vllm.engine.async_llm_engine': MagicMock(AsyncLLMEngine=mock_async_llm_engine)
            }):
                with pytest.raises(RuntimeError, match="Failed to initialize vLLM engine"):
                    await vllm_adapter.load_model(model_name)
                
                # Ensure cleanup on failure
                assert vllm_adapter.engine is None
                assert vllm_adapter.model_name is None
    
    @pytest.mark.asyncio
    async def test_unload_model(self, vllm_adapter):
        """Test model unloading."""
        # Set up loaded state
        vllm_adapter.engine = MagicMock()
        vllm_adapter.model_name = "test-model"
        vllm_adapter.model_path = "/path/to/model"
        vllm_adapter.model_info = MagicMock()
        
        await vllm_adapter.unload_model()
        
        assert vllm_adapter.engine is None
        assert vllm_adapter.model_name is None
        assert vllm_adapter.model_path is None
        assert vllm_adapter.model_info is None
    
    @pytest.mark.asyncio
    async def test_generate_no_model_loaded(self, vllm_adapter):
        """Test generation when no model is loaded."""
        with pytest.raises(RuntimeError, match="No model loaded"):
            await vllm_adapter.generate("Test prompt", {})
    
    @pytest.mark.asyncio
    async def test_generate_success(self, vllm_adapter):
        """Test successful text generation."""
        # Mock loaded model
        mock_engine = MagicMock()
        vllm_adapter.engine = mock_engine
        vllm_adapter.model_name = "test-model"
        
        # Mock vLLM SamplingParams
        with patch.dict('sys.modules', {'vllm': MagicMock()}):
            from unittest.mock import AsyncMock
            
            # Mock the async generator that vLLM returns
            mock_output = MagicMock()
            mock_output.finished = True
            mock_output.outputs = [MagicMock(text="Generated response")]
            
            async def mock_generator():
                yield mock_output
            
            mock_engine.generate.return_value = mock_generator()
            
            result = await vllm_adapter.generate("Test prompt", {"temperature": 0.7, "max_tokens": 100})
            
            assert result == "Generated response"
    
    @pytest.mark.asyncio
    async def test_generate_batch_no_model_loaded(self, vllm_adapter):
        """Test batch generation when no model is loaded."""
        with pytest.raises(RuntimeError, match="No model loaded"):
            await vllm_adapter.generate_batch(["Prompt 1", "Prompt 2"], {})
    
    @pytest.mark.asyncio
    async def test_generate_batch_success(self, vllm_adapter):
        """Test successful batch text generation."""
        # Mock loaded model
        mock_engine = MagicMock()
        vllm_adapter.engine = mock_engine
        vllm_adapter.model_name = "test-model"
        
        prompts = ["Prompt 1", "Prompt 2"]
        
        with patch.dict('sys.modules', {'vllm': MagicMock()}):
            # Mock generators for each prompt
            def create_mock_generator(response_text):
                async def mock_generator():
                    mock_output = MagicMock()
                    mock_output.finished = True
                    mock_output.outputs = [MagicMock(text=response_text)]
                    yield mock_output
                return mock_generator()
            
            # Mock engine.generate to return different generators for different calls
            mock_engine.generate.side_effect = [
                create_mock_generator("Response 1"),
                create_mock_generator("Response 2")
            ]
            
            results = await vllm_adapter.generate_batch(prompts, {"temperature": 0.5})
            
            assert len(results) == 2
            assert results == ["Response 1", "Response 2"]
    
    @pytest.mark.asyncio
    async def test_tokenize_no_model_loaded(self, vllm_adapter):
        """Test tokenization when no model is loaded."""
        with pytest.raises(RuntimeError, match="No model loaded"):
            await vllm_adapter.tokenize("Test text")
    
    @pytest.mark.asyncio
    async def test_tokenize_success(self, vllm_adapter):
        """Test successful tokenization."""
        # Mock loaded model with tokenizer
        mock_engine = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_engine.engine.tokenizer = mock_tokenizer
        
        vllm_adapter.engine = mock_engine
        vllm_adapter.model_name = "test-model"
        
        tokens = await vllm_adapter.tokenize("Test text")
        
        assert tokens == [1, 2, 3, 4, 5]
        mock_tokenizer.encode.assert_called_once_with("Test text")
    
    @pytest.mark.asyncio
    async def test_tokenize_fallback(self, vllm_adapter):
        """Test tokenization fallback when tokenizer fails."""
        # Mock loaded model with failing tokenizer
        mock_engine = MagicMock()
        mock_engine.engine.tokenizer.encode.side_effect = Exception("Tokenizer error")
        
        vllm_adapter.engine = mock_engine
        vllm_adapter.model_name = "test-model"
        
        tokens = await vllm_adapter.tokenize("hello world test")
        
        # Should fallback to word count-based tokens
        assert len(tokens) == 3  # 3 words
    
    @pytest.mark.asyncio
    async def test_count_tokens(self, vllm_adapter):
        """Test token counting."""
        # Mock tokenize method
        with patch.object(vllm_adapter, 'tokenize', return_value=[1, 2, 3, 4, 5]):
            count = await vllm_adapter.count_tokens("Test text")
            assert count == 5
    
    @pytest.mark.asyncio
    async def test_count_tokens_fallback(self, vllm_adapter):
        """Test token counting fallback."""
        # Mock tokenize to fail
        with patch.object(vllm_adapter, 'tokenize', side_effect=Exception("Tokenize failed")):
            count = await vllm_adapter.count_tokens("hello world test")
            assert count == 3  # Word count fallback
    
    def test_get_model_info_no_model(self, vllm_adapter):
        """Test getting model info when no model is loaded."""
        info = vllm_adapter.get_model_info()
        
        expected = {
            "model_name": None,
            "model_path": None,
            "engine": "vllm",
            "loaded": False,
        }
        assert info == expected
    
    def test_get_model_info_with_model(self, vllm_adapter, mock_model_info):
        """Test getting model info when model is loaded."""
        # Set up loaded state
        vllm_adapter.engine = MagicMock()
        vllm_adapter.model_name = "test-model"
        vllm_adapter.model_path = "/path/to/model"
        vllm_adapter.model_info = mock_model_info
        
        info = vllm_adapter.get_model_info()
        
        assert info["model_name"] == "test-model"
        assert info["model_path"] == "/path/to/model"
        assert info["engine"] == "vllm"
        assert info["loaded"] is True
        assert info["model_type"] == "llama"
        assert info["architecture"] == "LlamaForCausalLM"
        assert info["size_gb"] == 13.5
        assert info["format"] == "safetensors"
        assert info["framework_compatible"] == ["vllm", "transformers", "pytorch"]
    
    def test_get_capabilities(self, vllm_adapter):
        """Test getting vLLM capabilities."""
        capabilities = vllm_adapter.get_capabilities()
        
        expected_capabilities = {
            "streaming": True,
            "batch_processing": True,
            "continuous_batching": True,
            "tokenization": True,
            "quantization": True,
            "tensor_parallel": True,
            "paged_attention": True,
            "dynamic_batching": True,
        }
        
        assert capabilities == expected_capabilities
    
    @pytest.mark.asyncio
    async def test_get_metrics_no_model(self, vllm_adapter):
        """Test getting metrics when no model is loaded."""
        metrics = await vllm_adapter.get_metrics()
        
        expected = {
            "engine_loaded": False,
            "model_name": None,
            "model_path": None,
        }
        assert metrics == expected
    
    @pytest.mark.asyncio
    async def test_get_metrics_with_model(self, vllm_adapter, mock_model_info):
        """Test getting metrics when model is loaded."""
        # Set up loaded state
        mock_engine = MagicMock()
        mock_engine.get_stats.return_value = {"requests_processed": 100}
        
        vllm_adapter.engine = mock_engine
        vllm_adapter.model_name = "test-model"
        vllm_adapter.model_path = "/path/to/model"
        vllm_adapter.model_info = mock_model_info
        
        metrics = await vllm_adapter.get_metrics()
        
        assert metrics["engine_loaded"] is True
        assert metrics["model_name"] == "test-model"
        assert metrics["model_path"] == "/path/to/model"
        assert metrics["model_size_gb"] == 13.5
        assert metrics["model_type"] == "llama"
        assert metrics["engine_stats"]["requests_processed"] == 100
    
    @pytest.mark.asyncio
    async def test_get_metrics_engine_stats_unavailable(self, vllm_adapter, mock_model_info):
        """Test getting metrics when engine stats are unavailable."""
        # Set up loaded state without get_stats method
        mock_engine = MagicMock()
        del mock_engine.get_stats  # Remove get_stats method
        
        vllm_adapter.engine = mock_engine
        vllm_adapter.model_name = "test-model"
        vllm_adapter.model_info = mock_model_info
        
        metrics = await vllm_adapter.get_metrics()
        
        # Should work without engine stats
        assert metrics["engine_loaded"] is True
        assert "engine_stats" not in metrics
    
    @pytest.mark.asyncio
    async def test_load_model_with_kwargs(self, vllm_adapter, mock_discovered_models):
        """Test loading model with additional kwargs."""
        model_name = "test-llama-7b"
        kwargs = {
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.9,
            "max_num_batched_tokens": 4096
        }
        
        with patch.object(vllm_adapter.discovery, 'discover_all_models', return_value=mock_discovered_models):
            # Mock vLLM components  
            mock_engine = MagicMock()
            mock_llm = MagicMock()
            mock_engine_args_instance = MagicMock()
            mock_engine_args_class = MagicMock(return_value=mock_engine_args_instance)
            mock_async_llm_engine = MagicMock()
            mock_async_llm_engine.from_engine_args.return_value = mock_engine
            
            with patch.dict('sys.modules', {
                'vllm': mock_llm,
                'vllm.engine': MagicMock(),
                'vllm.engine.arg_utils': MagicMock(AsyncEngineArgs=mock_engine_args_class),
                'vllm.engine.async_llm_engine': MagicMock(AsyncLLMEngine=mock_async_llm_engine)
            }):
                await vllm_adapter.load_model(model_name, **kwargs)
                
                # Verify kwargs were passed to AsyncEngineArgs
                mock_engine_args_class.assert_called_once()
                call_args = mock_engine_args_class.call_args
                assert call_args[1]["tensor_parallel_size"] == 2
                assert call_args[1]["gpu_memory_utilization"] == 0.9
                assert call_args[1]["max_num_batched_tokens"] == 4096