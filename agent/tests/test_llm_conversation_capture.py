"""Tests for LLM conversation capture functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from ruckus_agent.adapters.vllm_adapter import VLLMAdapter
from ruckus_agent.core.models import ModelInfo
from ruckus_agent.core.agent import Agent
from ruckus_agent.core.config import Settings
from ruckus_common.models import (
    JobRequest, TaskType, LLMGenerationParams, PromptTemplate, PromptMessage, PromptRole, AgentType
)


class TestLLMConversationCapture:
    """Test LLM conversation capture functionality."""

    @pytest.fixture
    def vllm_adapter(self):
        """Create a VLLM adapter for testing."""
        return VLLMAdapter()

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
            discovered_at=datetime.now(timezone.utc)
        )

    @pytest.fixture
    def mock_discovered_models(self, mock_model_info):
        """Mock discovered models list."""
        return [mock_model_info]

    @pytest.fixture
    def sample_conversation_input(self):
        """Sample conversation input for testing."""
        return [
            PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
            PromptMessage(role=PromptRole.USER, content="What is the capital of France?")
        ]

    @pytest.fixture
    def expected_conversation_response(self):
        """Expected conversation response with assistant reply."""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]

    @pytest.mark.asyncio
    async def test_vllm_adapter_generate_with_conversation_capture(
        self, vllm_adapter, mock_discovered_models, sample_conversation_input, expected_conversation_response
    ):
        """Test that vLLM adapter can generate with conversation capture."""
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

            # Mock tokenizer for token counting
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.side_effect = [
                [1, 2, 3, 4, 5, 6, 7],  # system message tokens
                [8, 9, 10, 11, 12],      # user message tokens
                [13, 14, 15, 16, 17, 18] # assistant response tokens
            ]
            mock_engine.engine.tokenizer = mock_tokenizer

            with patch.dict('sys.modules', {
                'vllm': mock_llm,
                'vllm.engine': MagicMock(),
                'vllm.engine.arg_utils': MagicMock(AsyncEngineArgs=mock_engine_args),
                'vllm.engine.async_llm_engine': MagicMock(AsyncLLMEngine=mock_async_llm_engine)
            }):
                # Load model first
                await vllm_adapter.load_model(model_name)

                # Mock the async generator that vLLM returns
                mock_output = MagicMock()
                mock_output.finished = True
                mock_output.outputs = [MagicMock(text="The capital of France is Paris.")]

                async def mock_generator():
                    yield mock_output

                mock_engine.generate.return_value = mock_generator()

                # Test the new generate_with_conversation method
                conversation_result = await vllm_adapter.generate_with_conversation(
                    conversation=sample_conversation_input,
                    parameters={"temperature": 0.7, "max_tokens": 100}
                )

                # Verify conversation structure
                assert "conversation" in conversation_result
                assert "input_tokens" in conversation_result
                assert "output_tokens" in conversation_result
                assert "total_tokens" in conversation_result

                # Verify conversation content
                conversation = conversation_result["conversation"]
                assert len(conversation) == 3  # system + user + assistant
                assert conversation[0]["role"] == "system"
                assert conversation[0]["content"] == "You are a helpful assistant."
                assert conversation[1]["role"] == "user"
                assert conversation[1]["content"] == "What is the capital of France?"
                assert conversation[2]["role"] == "assistant"
                assert conversation[2]["content"] == "The capital of France is Paris."

                # Verify token counts
                assert conversation_result["input_tokens"] == 12  # 7 + 5 tokens
                assert conversation_result["output_tokens"] == 6   # 6 tokens
                assert conversation_result["total_tokens"] == 18   # 12 + 6 tokens

    @pytest.mark.asyncio
    async def test_vllm_adapter_generate_with_conversation_no_system_message(self, vllm_adapter, mock_discovered_models):
        """Test conversation capture with only user message (no system message)."""
        model_name = "test-llama-7b"
        user_only_conversation = [
            PromptMessage(role=PromptRole.USER, content="Hello, how are you?")
        ]

        with patch.object(vllm_adapter.discovery, 'discover_all_models', return_value=mock_discovered_models):
            # Mock vLLM components
            mock_engine = MagicMock()
            mock_llm = MagicMock()
            mock_engine_args = MagicMock()
            mock_async_llm_engine = MagicMock()
            mock_async_llm_engine.from_engine_args.return_value = mock_engine

            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.side_effect = [
                [1, 2, 3, 4],           # user message tokens
                [5, 6, 7, 8, 9]         # assistant response tokens
            ]
            mock_engine.engine.tokenizer = mock_tokenizer

            with patch.dict('sys.modules', {
                'vllm': mock_llm,
                'vllm.engine': MagicMock(),
                'vllm.engine.arg_utils': MagicMock(AsyncEngineArgs=mock_engine_args),
                'vllm.engine.async_llm_engine': MagicMock(AsyncLLMEngine=mock_async_llm_engine)
            }):
                await vllm_adapter.load_model(model_name)

                # Mock response
                mock_output = MagicMock()
                mock_output.finished = True
                mock_output.outputs = [MagicMock(text="I'm doing well, thank you!")]

                async def mock_generator():
                    yield mock_output

                mock_engine.generate.return_value = mock_generator()

                # Test conversation generation
                conversation_result = await vllm_adapter.generate_with_conversation(
                    conversation=user_only_conversation,
                    parameters={"temperature": 0.7, "max_tokens": 50}
                )

                # Verify conversation structure
                conversation = conversation_result["conversation"]
                assert len(conversation) == 2  # user + assistant
                assert conversation[0]["role"] == "user"
                assert conversation[0]["content"] == "Hello, how are you?"
                assert conversation[1]["role"] == "assistant"
                assert conversation[1]["content"] == "I'm doing well, thank you!"

                # Verify token counts
                assert conversation_result["input_tokens"] == 4   # 4 tokens
                assert conversation_result["output_tokens"] == 5  # 5 tokens
                assert conversation_result["total_tokens"] == 9   # 4 + 5 tokens

    @pytest.mark.asyncio
    async def test_vllm_adapter_generate_with_conversation_token_counting_fallback(
        self, vllm_adapter, mock_discovered_models, sample_conversation_input
    ):
        """Test conversation capture with token counting fallback when tokenizer fails."""
        model_name = "test-llama-7b"

        with patch.object(vllm_adapter.discovery, 'discover_all_models', return_value=mock_discovered_models):
            mock_engine = MagicMock()
            mock_llm = MagicMock()
            mock_engine_args = MagicMock()
            mock_async_llm_engine = MagicMock()
            mock_async_llm_engine.from_engine_args.return_value = mock_engine

            # Mock tokenizer that fails
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.side_effect = Exception("Tokenizer error")
            mock_engine.engine.tokenizer = mock_tokenizer

            with patch.dict('sys.modules', {
                'vllm': mock_llm,
                'vllm.engine': MagicMock(),
                'vllm.engine.arg_utils': MagicMock(AsyncEngineArgs=mock_engine_args),
                'vllm.engine.async_llm_engine': MagicMock(AsyncLLMEngine=mock_async_llm_engine)
            }):
                await vllm_adapter.load_model(model_name)

                mock_output = MagicMock()
                mock_output.finished = True
                mock_output.outputs = [MagicMock(text="Paris is the capital.")]

                async def mock_generator():
                    yield mock_output

                mock_engine.generate.return_value = mock_generator()

                # Test conversation generation with tokenizer failure
                conversation_result = await vllm_adapter.generate_with_conversation(
                    conversation=sample_conversation_input,
                    parameters={"temperature": 0.7, "max_tokens": 100}
                )

                # Should still work with word-count fallback
                assert "conversation" in conversation_result
                assert "input_tokens" in conversation_result
                assert "output_tokens" in conversation_result
                assert "total_tokens" in conversation_result

                # Token counts should be based on word count fallback
                # "You are a helpful assistant." = 5 words
                # "What is the capital of France?" = 7 words
                # But actual word split might differ, let's check the actual count
                expected_words = len("You are a helpful assistant.".split()) + len("What is the capital of France?".split())
                assert conversation_result["input_tokens"] == expected_words
                assert conversation_result["output_tokens"] == 4  # 4 words
                assert conversation_result["total_tokens"] == expected_words + 4

    @pytest.mark.asyncio
    async def test_agent_llm_task_stores_conversation_in_output(self):
        """Test that agent LLM task execution stores conversation in output field."""
        # Create test agent
        settings = Settings(
            agent_type=AgentType.WHITE_BOX,
            model_path="/test/models",
            job_max_execution_hours=1
        )
        agent = Agent(settings)

        # Create LLM job request
        job_request = JobRequest(
            job_id="test-job-123",
            experiment_id="test-exp-456",
            agent_id="test-agent",
            model="test-llama-7b",
            task_type=TaskType.LLM_GENERATION,
            task_config={
                "prompt_template": {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What is the capital of France?"}
                    ]
                }
            },
            parameters={
                "temperature": 0.7,
                "max_tokens": 100
            },
            framework="vllm",
            timeout_seconds=300,
            runs_per_job=1
        )

        # Mock vLLM adapter
        mock_adapter = AsyncMock()
        mock_conversation_result = {
            "conversation": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ],
            "input_tokens": 12,
            "output_tokens": 6,
            "total_tokens": 18
        }
        mock_adapter.generate_with_conversation.return_value = mock_conversation_result
        mock_adapter.load_model = AsyncMock()
        mock_adapter.unload_model = AsyncMock()
        mock_adapter.get_metrics = AsyncMock(return_value={})

        # Patch the agent's LLM task execution to use our mock adapter
        with patch.object(agent, '_get_model_adapter_for_task', return_value=mock_adapter):
            with patch.object(agent, '_get_current_hardware_info', return_value={}):
                # Execute single run
                run_start = datetime.now(timezone.utc)
                result = await agent._execute_llm_task(job_request, run_id=0, is_cold_start=True, run_start=run_start)

                # Verify conversation is stored in output
                assert result.output is not None
                assert "conversation" in result.output
                assert "input_tokens" in result.output
                assert "output_tokens" in result.output
                assert "total_tokens" in result.output

                # Verify conversation content
                conversation = result.output["conversation"]
                assert len(conversation) == 3
                assert conversation[0]["role"] == "system"
                assert conversation[1]["role"] == "user"
                assert conversation[2]["role"] == "assistant"
                assert conversation[2]["content"] == "The capital of France is Paris."

                # Verify token counts
                assert result.output["input_tokens"] == 12
                assert result.output["output_tokens"] == 6
                assert result.output["total_tokens"] == 18

                # Verify the adapter was called correctly
                mock_adapter.generate_with_conversation.assert_called_once()
                call_args = mock_adapter.generate_with_conversation.call_args
                assert len(call_args[1]["conversation"]) == 2  # system + user messages

    @pytest.mark.asyncio
    async def test_agent_llm_task_handles_conversation_capture_errors(self):
        """Test that agent handles errors during conversation capture gracefully."""
        settings = Settings(
            agent_type=AgentType.WHITE_BOX,
            model_path="/test/models",
            job_max_execution_hours=1
        )
        agent = Agent(settings)

        job_request = JobRequest(
            job_id="test-job-error",
            experiment_id="test-exp-error",
            agent_id="test-agent",
            model="test-llama-7b",
            task_type=TaskType.LLM_GENERATION,
            task_config={
                "prompt_template": {
                    "messages": [
                        {"role": "user", "content": "Test prompt"}
                    ]
                }
            },
            parameters={"temperature": 0.7},
            framework="vllm",
            timeout_seconds=300,
            runs_per_job=1
        )

        # Mock adapter that raises an error
        mock_adapter = AsyncMock()
        mock_adapter.generate_with_conversation.side_effect = Exception("Model inference failed")

        with patch.object(agent, '_get_model_adapter_for_task', return_value=mock_adapter):
            with patch.object(agent, '_get_current_hardware_info', return_value={}):
                # Execute should raise the error
                run_start = datetime.now(timezone.utc)
                with pytest.raises(Exception, match="Model inference failed"):
                    await agent._execute_llm_task(job_request, run_id=0, is_cold_start=True, run_start=run_start)

    @pytest.mark.asyncio
    async def test_agent_integration_end_to_end_conversation_capture(self):
        """Test end-to-end conversation capture from job execution to result storage."""
        settings = Settings(
            agent_type=AgentType.WHITE_BOX,
            model_path="/test/models",
            job_max_execution_hours=1
        )
        agent = Agent(settings)

        # Start agent
        await agent.start()

        try:
            job_request = JobRequest(
                job_id="test-integration-job",
                experiment_id="test-integration-exp",
                agent_id="test-agent",
                model="test-llama-7b",
                task_type=TaskType.LLM_GENERATION,
                task_config={
                    "prompt_template": {
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "What is the capital of France?"}
                        ]
                    }
                },
                parameters={"temperature": 0.7, "max_tokens": 100},
                framework="vllm",
                timeout_seconds=300,
                runs_per_job=1
            )

            # Mock model adapter for the integration test
            mock_adapter = AsyncMock()
            mock_conversation_result = {
                "conversation": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "The capital of France is Paris."}
                ],
                "input_tokens": 12,
                "output_tokens": 6,
                "total_tokens": 18
            }
            mock_adapter.generate_with_conversation.return_value = mock_conversation_result
            mock_adapter.load_model = AsyncMock()
            mock_adapter.unload_model = AsyncMock()
            mock_adapter.get_metrics = AsyncMock(return_value={})

            with patch.object(agent, '_get_model_adapter_for_task', return_value=mock_adapter):
                with patch.object(agent, '_get_current_hardware_info', return_value={}):
                    # Execute job
                    job_result = await agent.execute_job(job_request)

                    # Verify job completed successfully
                    assert job_result.status.value == "completed"
                    assert job_result.error is None

                    # Verify conversation is stored in the result cache
                    cached_result = agent.result_cache.get(job_request.job_id)
                    assert cached_result is not None

                    # For single-run jobs, the result should be in the JobResult format
                    # The conversation should be accessible in the metrics or output
                    assert job_result.output is not None
                    assert "conversation" in job_result.output
                    assert len(job_result.output["conversation"]) == 3
                    assert job_result.output["conversation"][2]["content"] == "The capital of France is Paris."

        finally:
            await agent.stop()