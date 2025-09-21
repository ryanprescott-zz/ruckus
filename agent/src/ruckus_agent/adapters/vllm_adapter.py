"""vLLM adapter for high-performance inference."""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from .base import ModelAdapter
from ..utils.model_discovery import ModelDiscovery
from ..core.config import settings

logger = logging.getLogger(__name__)


class VLLMAdapter(ModelAdapter):
    """Adapter for vLLM inference engine."""

    def __init__(self):
        self.engine = None
        self.model_name = None
        self.model_path = None
        self.model_info = None
        self.discovery = ModelDiscovery(settings.model_path)
        logger.info("VLLMAdapter initialized")

    async def load_model(self, model_name: str, **kwargs) -> None:
        """Load a model with vLLM using discovered model information."""
        logger.info(f"VLLMAdapter loading model: {model_name}")
        
        try:
            # Find model in discovered models
            discovered_models = await self.discovery.discover_all_models()
            matching_model = None
            
            for model in discovered_models:
                if model.name == model_name or model.path.endswith(model_name):
                    matching_model = model
                    break
            
            if not matching_model:
                raise ValueError(f"Model {model_name} not found in discovered models")
            
            # Validate model is compatible with vLLM
            if "vllm" not in matching_model.framework_compatible:
                logger.warning(f"Model {model_name} may not be compatible with vLLM (compatible: {matching_model.framework_compatible})")
            
            # Store model information
            self.model_name = model_name
            self.model_path = matching_model.path
            self.model_info = matching_model
            
            # Load model with vLLM
            logger.info(f"Loading model from path: {self.model_path}")
            logger.debug(f"Model info: {matching_model.model_type}, {matching_model.size_gb:.2f}GB, {matching_model.format}")
            
            try:
                # Import vLLM here to handle import errors gracefully
                from vllm import LLM, SamplingParams
                from vllm.engine.arg_utils import AsyncEngineArgs
                from vllm.engine.async_llm_engine import AsyncLLMEngine
                
                # Configure vLLM engine arguments
                engine_args = AsyncEngineArgs(
                    model=self.model_path,
                    **kwargs
                )
                
                # Create async engine
                self.engine = AsyncLLMEngine.from_engine_args(engine_args)
                
                logger.info(f"VLLMAdapter model loaded successfully: {model_name}")
                logger.debug(f"Model loaded with vLLM engine at {self.model_path}")
                
            except ImportError as e:
                raise ImportError(
                    f"❌ vLLM framework is not installed or failed to import: {e}. "
                    f"Please install vLLM with: pip install vllm"
                )
            except Exception as e:
                # Re-raise with more context
                raise RuntimeError(
                    f"❌ Failed to initialize vLLM engine for model '{model_name}': {e}. "
                    f"This could be due to insufficient GPU memory, incompatible model format, "
                    f"or vLLM configuration issues. Please check the model requirements and system resources."
                )
                
        except Exception as e:
            logger.error(f"VLLMAdapter failed to load model {model_name}: {e}")
            # Clean up on failure
            self.engine = None
            self.model_name = None
            self.model_path = None
            self.model_info = None
            raise

    async def unload_model(self) -> None:
        """Unload the current model."""
        logger.info(f"VLLMAdapter unloading model: {self.model_name}")
        
        # Clean up vLLM engine
        if self.engine:
            try:
                # Note: AsyncLLMEngine doesn't have explicit cleanup in older versions
                # In newer versions, you might need to call engine.shutdown()
                pass
            except Exception as e:
                logger.warning(f"Error during engine cleanup: {e}")
        
        # Clear references
        self.engine = None
        self.model_name = None
        self.model_path = None
        self.model_info = None
        logger.info("VLLMAdapter model unloaded")

    async def generate(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Generate with vLLM."""
        if not self.engine:
            raise RuntimeError("No model loaded. Call load_model() first.")
            
        logger.debug(f"VLLMAdapter generating for prompt length: {len(prompt)}")
        
        try:
            from vllm import SamplingParams
            
            # Create sampling parameters from input parameters
            sampling_params = SamplingParams(
                temperature=parameters.get('temperature', 0.7),
                max_tokens=parameters.get('max_tokens', 100),
                top_p=parameters.get('top_p', 1.0),
                top_k=parameters.get('top_k', -1),
            )
            
            # Generate using vLLM async engine
            request_id = f"req_{asyncio.get_event_loop().time()}"
            
            # Add request to engine
            results_generator = self.engine.generate(
                prompt, 
                sampling_params, 
                request_id
            )
            
            # Collect results
            final_output = ""
            async for request_output in results_generator:
                if request_output.finished:
                    # Get the generated text
                    if request_output.outputs:
                        final_output = request_output.outputs[0].text
                    break
            
            logger.debug(f"VLLMAdapter generated output length: {len(final_output)}")
            return final_output
            
        except Exception as e:
            logger.error(f"VLLMAdapter generation failed: {e}")
            raise

    async def generate_batch(self, prompts: List[str], parameters: Dict[str, Any]) -> List[str]:
        """Batch generation with vLLM."""
        if not self.engine:
            raise RuntimeError("No model loaded. Call load_model() first.")
            
        logger.info(f"VLLMAdapter batch generation for {len(prompts)} prompts")
        
        try:
            from vllm import SamplingParams
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=parameters.get('temperature', 0.7),
                max_tokens=parameters.get('max_tokens', 100),
                top_p=parameters.get('top_p', 1.0),
                top_k=parameters.get('top_k', -1),
            )
            
            # Generate batch requests
            request_ids = [f"req_{i}_{asyncio.get_event_loop().time()}" for i in range(len(prompts))]
            
            # Submit all requests
            generators = []
            for prompt, request_id in zip(prompts, request_ids):
                generator = self.engine.generate(prompt, sampling_params, request_id)
                generators.append(generator)
            
            # Collect results
            results = []
            for generator in generators:
                final_output = ""
                async for request_output in generator:
                    if request_output.finished:
                        if request_output.outputs:
                            final_output = request_output.outputs[0].text
                        break
                results.append(final_output)
            
            logger.info(f"VLLMAdapter batch generation completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"VLLMAdapter batch generation failed: {e}")
            raise

    async def tokenize(self, text: str) -> List[int]:
        """Tokenize text using the model's tokenizer."""
        if not self.engine:
            raise RuntimeError("No model loaded. Call load_model() first.")
            
        try:
            # Access tokenizer through engine
            tokenizer = self.engine.engine.tokenizer
            token_ids = tokenizer.encode(text)
            return token_ids
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            # Fallback to simple word splitting
            return list(range(len(text.split())))

    async def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            tokens = await self.tokenize(text)
            return len(tokens)
        except Exception:
            # Fallback to word count
            return len(text.split())

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        base_info = {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "engine": "vllm",
            "loaded": self.engine is not None,
        }
        
        # Add discovered model metadata if available
        if self.model_info:
            base_info.update({
                "model_type": self.model_info.model_type,
                "architecture": self.model_info.architecture,
                "size_gb": self.model_info.size_gb,
                "format": self.model_info.format,
                "vocab_size": self.model_info.vocab_size,
                "hidden_size": self.model_info.hidden_size,
                "num_layers": self.model_info.num_layers,
                "num_attention_heads": self.model_info.num_attention_heads,
                "max_position_embeddings": self.model_info.max_position_embeddings,
                "torch_dtype": self.model_info.torch_dtype,
                "quantization": self.model_info.quantization,
                "tokenizer_type": self.model_info.tokenizer_type,
                "framework_compatible": self.model_info.framework_compatible,
                "discovered_at": self.model_info.discovered_at.isoformat() if self.model_info.discovered_at else None,
            })
        
        return base_info

    def get_capabilities(self) -> Dict[str, bool]:
        """Get vLLM capabilities."""
        return {
            "streaming": True,
            "batch_processing": True,
            "continuous_batching": True,
            "tokenization": True,
            "quantization": True,
            "tensor_parallel": True,
            "paged_attention": True,
            "dynamic_batching": True,
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get vLLM runtime metrics."""
        metrics = {
            "engine_loaded": self.engine is not None,
            "model_name": self.model_name,
            "model_path": self.model_path,
        }
        
        if self.model_info:
            metrics.update({
                "model_size_gb": self.model_info.size_gb,
                "model_type": self.model_info.model_type,
                "quantization": self.model_info.quantization,
            })
        
        # Add engine-specific metrics if available
        if self.engine:
            try:
                # Try to get engine stats if available
                # Note: This depends on vLLM version and may not always be available
                engine_stats = getattr(self.engine, 'get_stats', lambda: {})()
                if engine_stats:
                    metrics.update({
                        "engine_stats": engine_stats
                    })
            except Exception as e:
                logger.debug(f"Could not retrieve engine stats: {e}")
        
        return metrics

    async def generate_with_conversation(self, conversation: List[Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate with full conversation capture for LLM tasks.

        Args:
            conversation: List of PromptMessage objects containing the conversation
            parameters: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            Dict containing:
                - conversation: Full conversation including assistant response
                - input_tokens: Number of input tokens
                - output_tokens: Number of output tokens
                - total_tokens: Total token count
        """
        if not self.engine:
            raise RuntimeError("No model loaded. Call load_model() first.")

        logger.debug(f"VLLMAdapter generating conversation with {len(conversation)} messages")

        try:
            # Convert conversation to prompt format expected by vLLM
            # For now, we'll concatenate all messages into a single prompt
            # TODO: Use proper chat template when available
            prompt_parts = []
            input_messages = []

            for msg in conversation:
                # Handle both dict format and PromptMessage objects
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    role = msg.role.value if hasattr(msg.role, 'value') else msg.role
                    content = msg.content
                else:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')

                # Store original message
                input_messages.append({"role": role, "content": content})

                # Build prompt (simple concatenation for now)
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")

            # Add assistant prompt to get response
            prompt_parts.append("Assistant:")
            full_prompt = "\n".join(prompt_parts)

            logger.debug(f"Generated prompt for vLLM (length: {len(full_prompt)})")

            # Count input tokens by summing individual message tokens
            input_tokens = 0
            try:
                for msg_data in input_messages:
                    msg_tokens = await self.count_tokens(msg_data["content"])
                    input_tokens += msg_tokens
            except Exception as e:
                logger.warning(f"Failed to count input tokens: {e}, using word count fallback")
                input_tokens = sum(len(msg["content"].split()) for msg in input_messages)

            # Generate response using existing generate method
            response_text = await self.generate(full_prompt, parameters)

            # Count output tokens
            try:
                output_tokens = await self.count_tokens(response_text)
            except Exception as e:
                logger.warning(f"Failed to count output tokens: {e}, using word count fallback")
                output_tokens = len(response_text.split())

            # Build complete conversation including assistant response
            complete_conversation = input_messages.copy()
            complete_conversation.append({
                "role": "assistant",
                "content": response_text.strip()
            })

            # Calculate total tokens
            total_tokens = input_tokens + output_tokens

            result = {
                "conversation": complete_conversation,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "model_response": response_text.strip()  # Quick access to just the response
            }

            logger.debug(f"Conversation generation completed: {input_tokens} input + {output_tokens} output = {total_tokens} total tokens")
            return result

        except Exception as e:
            logger.error(f"VLLMAdapter conversation generation failed: {e}")
            raise
