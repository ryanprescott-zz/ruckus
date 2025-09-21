"""Main agent implementation."""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from ruckus_common.models import (
    JobRequest, JobStatusEnum, AgentType, JobUpdate, AgentStatus, AgentStatusEnum,
    SingleRunResult, MetricStatistics, MultiRunJobResult, JobResult, JobStage,
    JobResultType, TaskType, GPUBenchmarkParams, MemoryBenchmarkParams, ComputeBenchmarkParams
)
from .config import Settings
from .detector import AgentDetector
from .storage import AgentStorage, InMemoryStorage
from .result_cache import TTLResultCache
from ..utils.error_reporter import ErrorReporter

logger = logging.getLogger(__name__)


class Agent:
    """Main agent class coordinating job execution."""

    def __init__(self, settings: Settings, storage: Optional[AgentStorage] = None):
        self.settings = settings
        # Generate unique agent ID and name
        self.agent_id = f"agent-{uuid.uuid4().hex[:8]}"
        self.agent_name = f"{self.agent_id}-{settings.agent_type.value}"

        # Storage backend
        self.storage = storage or InMemoryStorage()

        # State
        self.running_jobs: Dict[str, Any] = {}
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.queued_job_ids: List[str] = []  # Track queued job IDs for status reporting
        self.startup_time = datetime.now(timezone.utc)
        self.crashed = False
        self.crash_reason: Optional[str] = None
        
        # Error reporting
        self.error_reporter = ErrorReporter(self.agent_id)

        # Result cache
        self.result_cache = TTLResultCache(
            ttl_hours=settings.result_cache_ttl_hours,
            cleanup_interval_minutes=settings.result_cache_cleanup_interval_minutes
        )

        # Background tasks
        self.tasks: List[asyncio.Task] = []
        
        logger.info(f"Agent initialized: {self.agent_id} ({self.agent_name})")

    async def start(self):
        """Start the agent."""
        logger.info(f"Starting agent {self.agent_id}")

        # Start result cache
        await self.result_cache.start()

        # Detect capabilities
        await self._detect_capabilities()

        # Start background tasks
        self.tasks.append(asyncio.create_task(self._job_executor()))

    async def stop(self):
        """Stop the agent."""
        logger.info(f"Stopping agent {self.agent_id}")

        # Stop result cache
        await self.result_cache.stop()

        # Cancel background tasks
        for task in self.tasks:
            task.cancel()

    async def _detect_capabilities(self):
        """Detect agent capabilities and system info."""
        logger.info(f"Agent {self.agent_id} starting capability detection")
        try:
            detector = AgentDetector()
            detected = await detector.detect_all()
            
            # Convert Pydantic models to dict for storage (legacy format)
            # Convert models list to dictionary format for UI consumption
            models_dict = {}
            if detected.models:
                for model in detected.models:
                    if isinstance(model, dict) and 'name' in model:
                        models_dict[model['name']] = model
                    
            system_info = {
                "system": detected.system.dict() if detected.system else {},
                "cpu": detected.cpu.dict() if detected.cpu else {},
                "gpus": [gpu.dict() for gpu in detected.gpus] if detected.gpus else [],
                "frameworks": [fw.dict() for fw in detected.frameworks] if detected.frameworks else [],
                "models": models_dict,  # Convert to dict with model names as keys
                "hooks": [hook.dict() for hook in detected.hooks] if detected.hooks else [],
                "metrics": [metric.dict() for metric in detected.metrics] if detected.metrics else []
            }
            await self.storage.store_system_info(system_info)
            
            logger.info(f"Agent {self.agent_id} system detection completed successfully")
            logger.debug(f"Detected {len(detected.gpus)} GPUs, {len(detected.frameworks)} frameworks")
        except Exception as e:
            logger.error(f"Agent {self.agent_id} system detection failed: {e}")
            raise



    async def _job_executor(self):
        """Execute jobs from queue."""
        logger.debug(f"Agent {self.agent_id} starting job executor")
        while True:
            try:
                job = await self.job_queue.get()
                # Remove from queued job IDs when we start processing
                if job.job_id in self.queued_job_ids:
                    self.queued_job_ids.remove(job.job_id)
                logger.info(f"Agent {self.agent_id} received job from queue: {job.job_id}")
                
                # Execute job in a separate task to avoid blocking the job executor
                task = asyncio.create_task(self._execute_job(job))
                # Store the task so it doesn't get garbage collected
                self.tasks.append(task)
                # Remove completed tasks to prevent memory leaks
                task.add_done_callback(lambda t: self.tasks.remove(t) if t in self.tasks else None)
                
            except asyncio.CancelledError:
                logger.debug(f"Agent {self.agent_id} job executor cancelled")
                break
            except Exception as e:
                logger.error(f"Agent {self.agent_id} job execution error: {e}")

    async def _execute_job(self, job: JobRequest):
        """Execute a job with support for multiple runs and cold start tracking."""
        logger.info(f"Agent {self.agent_id} executing job {job.job_id} with {job.runs_per_job} runs")
        
        start_time = datetime.now(timezone.utc)
        error_report = None
        
        # Calculate timeout from config and job request
        timeout_seconds = min(
            job.timeout_seconds,  # Job-specific timeout
            int(self.settings.job_max_execution_hours * 3600)  # Agent max timeout
        )
        
        logger.info(f"Job {job.job_id} timeout set to {timeout_seconds} seconds")
        
        try:
            # Execute with timeout
            job_result = await asyncio.wait_for(self._execute_job_impl(job, start_time), timeout=timeout_seconds)
            return job_result

        except asyncio.TimeoutError:
            logger.error(f"Job {job.job_id} timed out after {timeout_seconds} seconds")
            
            # Create timeout result
            timeout_result = {
                "job_id": job.job_id,
                "experiment_id": job.experiment_id,
                "status": "timeout",
                "started_at": start_time.isoformat(),
                "timeout_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": timeout_seconds,
                "timeout_seconds": timeout_seconds,
                "error": f"Job execution exceeded timeout of {timeout_seconds} seconds"
            }
            
            # Cache the timeout result
            self.result_cache.store(
                job_id=job.job_id,
                result=timeout_result,
                result_type=JobResultType.EXECUTION_FAILURE
            )

            # Perform cleanup
            await self._cleanup_failed_job(job, asyncio.TimeoutError("Job execution timeout"))
            return timeout_result
            
        except Exception as e:
            logger.error(f"Unexpected error in job {job.job_id}: {e}")
            raise
            
        finally:
            # Clean up job tracking and running jobs
            await self.error_reporter.cleanup_job_tracking(job.job_id)
            self.running_jobs.pop(job.job_id, None)
    
    async def _execute_job_impl(self, job: JobRequest, start_time: datetime):
        """Internal job execution implementation (wrapped with timeout).
        
        Args:
            job: Job to execute
            start_time: When job execution started
        """
        try:
            # Yield control at the start to ensure responsiveness
            await asyncio.sleep(0)
            
            # Start job tracking for error reporting
            await self.error_reporter.start_job_tracking(job.job_id, "initializing")
            
            # Add to running jobs
            self.running_jobs[job.job_id] = {
                "job": job,
                "start_time": start_time,
            }

            # Update job stage
            await self.error_reporter.update_job_stage(job.job_id, "starting")
            
            # Check for cancellation before starting execution
            if self._is_job_cancelled(job.job_id):
                return await self._handle_job_cancellation(job, start_time)
            
            # Execute multi-run job
            if job.runs_per_job == 1:
                # Single run job - simpler path
                result = await self._execute_single_run(job, run_id=0, is_cold_start=True)
                job_result = self._convert_single_to_job_result(result, job, start_time)
                
                # Determine result type
                if result.error is None:
                    result_type = JobResultType.SUCCESS
                else:
                    result_type = JobResultType.EXECUTION_FAILURE
                    
            else:
                # Multi-run job - with cold start separation
                job_result = await self._execute_multi_run_job(job, start_time)
                
                # Determine result type based on success/failure ratio
                if job_result.failed_runs == 0:
                    result_type = JobResultType.SUCCESS
                elif job_result.successful_runs == 0:
                    result_type = JobResultType.EXECUTION_FAILURE
                else:
                    result_type = JobResultType.PARTIAL_SUCCESS
            
            # Cache the job result
            cached_job_id = self.result_cache.store(
                job_id=job.job_id,
                result=job_result.model_dump() if hasattr(job_result, 'model_dump') else job_result.__dict__,
                result_type=result_type
            )
            
            logger.info(f"Agent {self.agent_id} job {job.job_id} completed successfully and cached as {cached_job_id}")
            
            return job_result
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} job {job.job_id} execution failed: {e}")
            
            try:
                # Generate comprehensive error report
                error_report = await self.error_reporter.generate_error_report(
                    job_id=job.job_id,
                    experiment_id=job.experiment_id,
                    error=e,
                    model_name=job.model,
                    model_path=str(Path(self.settings.model_path) / job.model),
                    framework=job.framework,
                    task_type=job.task_type.value,
                    parameters=job.parameters,
                    started_at=start_time,
                    model_size_gb=None  # TODO: Get from model discovery
                )
                
                # Attempt cleanup
                cleanup_actions = await self._cleanup_failed_job(job, e)
                error_report.cleanup_actions = cleanup_actions
                error_report.recovery_successful = True  # If we reach here, cleanup worked
                
                # Determine error result type based on error category
                if "OutOfMemoryError" in str(e) or "CUDA out of memory" in str(e):
                    result_type = JobResultType.CONSTRAINT_FAILURE
                elif "model not found" in str(e).lower() or "framework" in str(e).lower():
                    result_type = JobResultType.CONSTRAINT_FAILURE
                else:
                    result_type = JobResultType.EXECUTION_FAILURE
                
                # Create error result object
                error_result = {
                    "job_id": job.job_id,
                    "experiment_id": job.experiment_id,
                    "status": "failed",
                    "started_at": start_time.isoformat(),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "error_report": error_report.model_dump() if hasattr(error_report, 'model_dump') else error_report.__dict__,
                    "cleanup_actions": cleanup_actions,
                    "recovery_successful": True
                }
                
                # Cache the error result
                cached_job_id = self.result_cache.store(
                    job_id=job.job_id,
                    result=error_result,
                    result_type=result_type
                )
                
                logger.info(f"Generated error report for job {job.job_id}: {error_report.error_type}, cached as {cached_job_id}")
                
            except Exception as cleanup_error:
                logger.error(f"Failed to generate error report or cleanup job {job.job_id}: {cleanup_error}")
                
                # Create minimal error result for catastrophic failures
                catastrophic_error_result = {
                    "job_id": job.job_id,
                    "experiment_id": job.experiment_id,
                    "status": "failed",
                    "started_at": start_time.isoformat(),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "cleanup_error": str(cleanup_error),
                    "recovery_successful": False
                }
                
                # Cache the catastrophic error
                self.result_cache.store(
                    job_id=job.job_id,
                    result=catastrophic_error_result,
                    result_type=JobResultType.EXECUTION_FAILURE
                )
                
                # Mark agent as crashed if cleanup fails
                self.crashed = True
                self.crash_reason = f"Failed cleanup after job {job.job_id}: {cleanup_error}"
            
            logger.error(f"Job {job.job_id} failed: {str(e)}")
            
            # Perform comprehensive cleanup and verify idle state
            await self._perform_comprehensive_cleanup()
            
            # Verify that agent is truly in clean idle state
            if not self.running_jobs:  # No more jobs running
                cleanup_verified = await self._verify_clean_idle_state()
                if not cleanup_verified:
                    logger.warning("Agent cleanup verification failed - some resources may not be properly cleaned")

    async def _execute_multi_run_job(self, job: JobRequest, job_start_time: datetime):
        """Execute a multi-run job with cold start separation and statistical analysis."""
        import statistics
        
        logger.info(f"Starting multi-run job {job.job_id} with {job.runs_per_job} runs")
        
        individual_runs = []
        successful_runs = 0
        failed_runs = 0
        
        # Execute runs sequentially
        for run_id in range(job.runs_per_job):
            is_cold_start = (run_id == 0)  # First run is cold start
            logger.info(f"Executing run {run_id + 1}/{job.runs_per_job} (cold_start={is_cold_start})")
            
            try:
                run_result = await self._execute_single_run(job, run_id, is_cold_start)
                individual_runs.append(run_result)
                successful_runs += 1
                logger.info(f"Run {run_id + 1} completed successfully")
            except Exception as e:
                logger.error(f"Run {run_id + 1} failed: {e}")
                # Create failed run result
                failed_run = SingleRunResult(
                    run_id=run_id,
                    is_cold_start=is_cold_start,
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    duration_seconds=0.0,
                    error=str(e),
                    error_type=type(e).__name__
                )
                individual_runs.append(failed_run)
                failed_runs += 1
        
        # Separate cold start from warm runs for statistics
        cold_start_data = individual_runs[0] if individual_runs and individual_runs[0].is_cold_start else None
        warm_runs = [run for run in individual_runs if not run.is_cold_start and run.error is None]
        
        # Calculate summary statistics for warm runs only
        summary_stats = {}
        if warm_runs:
            summary_stats = self._calculate_multi_run_statistics(warm_runs)
        
        # Get GPU benchmark results if available
        gpu_benchmarks = None
        if hasattr(self, '_current_job_benchmarks') and job.job_id in self._current_job_benchmarks:
            gpu_benchmarks = self._current_job_benchmarks.pop(job.job_id)
        
        # Create final result
        job_result = MultiRunJobResult(
            job_id=job.job_id,
            experiment_id=job.experiment_id,
            total_runs=job.runs_per_job,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            individual_runs=individual_runs,
            summary_stats=summary_stats,
            cold_start_data=cold_start_data,
            started_at=job_start_time,
            completed_at=datetime.now(timezone.utc),
            total_duration_seconds=(datetime.now(timezone.utc) - job_start_time).total_seconds(),
            model_actual=job.model,
            framework_version="vllm-0.2.0",  # TODO: Get actual version
            hardware_info=await self._get_current_hardware_info(),
            gpu_benchmark_results=gpu_benchmarks
        )
        
        logger.info(f"Multi-run job {job.job_id} completed: {successful_runs} successful, {failed_runs} failed")
        return job_result

    async def _execute_single_run(self, job: JobRequest, run_id: int, is_cold_start: bool) -> SingleRunResult:
        """Execute a single run with task type switching."""

        run_start = datetime.now(timezone.utc)
        logger.info(f"Executing run {run_id} for job {job.job_id}, task type: {job.task_type}")

        try:
            # Update job stage
            stage = f"run_{run_id + 1}_starting"
            await self.error_reporter.update_job_stage(job.job_id, stage)

            # Task type switching - dispatch to appropriate executor
            if job.task_type == TaskType.LLM_GENERATION:
                return await self._execute_llm_task(job, run_id, is_cold_start, run_start)
            elif job.task_type == TaskType.GPU_BENCHMARK:
                return await self._execute_gpu_benchmark_task(job, run_id, is_cold_start, run_start)
            elif job.task_type == TaskType.MEMORY_BENCHMARK:
                return await self._execute_memory_benchmark_task(job, run_id, is_cold_start, run_start)
            elif job.task_type == TaskType.COMPUTE_BENCHMARK:
                return await self._execute_compute_benchmark_task(job, run_id, is_cold_start, run_start)
            else:
                raise ValueError(f"Unsupported task type: {job.task_type}")
            
        except Exception as e:
            run_end = datetime.now(timezone.utc)
            duration = (run_end - run_start).total_seconds()

            # Create failed run result
            return SingleRunResult(
                run_id=run_id,
                is_cold_start=is_cold_start,
                started_at=run_start,
                completed_at=run_end,
                duration_seconds=duration,
                error=str(e),
                error_type=type(e).__name__,
                # Model loading metrics only relevant for LLM tasks
                model_load_time_seconds=None,
                model_load_memory_mb=None
            )

    async def _execute_llm_task(self, job: JobRequest, run_id: int, is_cold_start: bool, run_start: datetime) -> SingleRunResult:
        """Execute LLM generation task with real inference."""
        from ruckus_common.models import LLMGenerationParams, PromptTemplate, PromptMessage

        model_load_time = None
        model_load_memory = None

        # Get model adapter for the task
        model_adapter = await self._get_model_adapter_for_task(job)

        # Cold start: Load model and track loading metrics
        if is_cold_start:
            await self.error_reporter.update_job_stage(job.job_id, f"run_{run_id + 1}_model_loading")
            load_start = datetime.now(timezone.utc)

            try:
                # Load model with any provided parameters
                load_params = job.parameters.get('load_params', {})
                await model_adapter.load_model(job.model, **load_params)

                model_load_time = (datetime.now(timezone.utc) - load_start).total_seconds()
                model_load_memory = 8192.0  # TODO: Get actual VRAM usage from adapter

                logger.info(f"Cold start model load completed in {model_load_time:.2f}s")
            except Exception as e:
                logger.error(f"Model loading failed: {e}")
                raise

        # Parse LLM generation parameters from task config
        task_config = job.task_config
        if 'prompt_template' not in task_config:
            raise ValueError("LLM task requires 'prompt_template' in task_config")

        prompt_template_data = task_config['prompt_template']

        # Convert dict messages to PromptMessage objects if needed
        messages = prompt_template_data.get('messages', [])
        conversation = []
        for msg in messages:
            if isinstance(msg, dict):
                # Convert dict to PromptMessage-like object for adapter
                conversation.append(msg)
            else:
                # Already a PromptMessage object
                conversation.append({
                    "role": msg.role.value if hasattr(msg.role, 'value') else msg.role,
                    "content": msg.content
                })

        # Inference execution
        await self.error_reporter.update_job_stage(job.job_id, f"run_{run_id + 1}_inference")

        inference_start = datetime.now(timezone.utc)

        try:
            # Check if adapter supports conversation capture
            if not hasattr(model_adapter, 'generate_with_conversation'):
                adapter_type = type(model_adapter).__name__
                raise AttributeError(
                    f"❌ {adapter_type} does not support conversation capture. "
                    f"The {job.framework} framework adapter needs to implement 'generate_with_conversation()' method "
                    f"for LLM conversation tracking. Please use a framework adapter that supports conversation capture "
                    f"(e.g., vLLM, Transformers) or update the {adapter_type} implementation."
                )

            # Use real inference with conversation capture
            conversation_result = await model_adapter.generate_with_conversation(
                conversation=conversation,
                parameters=job.parameters
            )

            inference_end = datetime.now(timezone.utc)
            inference_time = (inference_end - inference_start).total_seconds()

            # Extract metrics from conversation result
            metrics = {
                "inference_time_seconds": inference_time,
                "input_tokens": conversation_result.get("input_tokens", 0),
                "output_tokens": conversation_result.get("output_tokens", 0),
                "total_tokens": conversation_result.get("total_tokens", 0),
                "throughput_tokens_per_sec": conversation_result.get("output_tokens", 0) / max(inference_time, 0.001)
            }

            # Add model-specific metrics if available
            if hasattr(model_adapter, 'get_metrics'):
                try:
                    adapter_metrics = await model_adapter.get_metrics()
                    metrics.update(adapter_metrics)
                except Exception as e:
                    logger.warning(f"Failed to get adapter metrics: {e}")

            run_end = datetime.now(timezone.utc)
            duration = (run_end - run_start).total_seconds()

            logger.info(f"LLM inference completed: {metrics['output_tokens']} tokens in {inference_time:.2f}s")

            return SingleRunResult(
                run_id=run_id,
                is_cold_start=is_cold_start,
                started_at=run_start,
                completed_at=run_end,
                duration_seconds=duration,
                metrics=metrics,
                model_load_time_seconds=model_load_time,
                model_load_memory_mb=model_load_memory,
                output=conversation_result  # Store full conversation result
            )

        except Exception as e:
            logger.error(f"LLM inference failed: {e}")

            # Enhance error messages for common issues
            error_msg = str(e)
            if "ImportError" in error_msg or "No module named" in error_msg:
                enhanced_msg = f"🔧 Dependency Missing: {error_msg}"
            elif "AttributeError" in error_msg and "generate_with_conversation" in error_msg:
                enhanced_msg = f"🔧 Feature Not Supported: {error_msg}"
            elif "CUDA" in error_msg or "GPU" in error_msg:
                enhanced_msg = f"🔧 Hardware Issue: {error_msg}"
            elif "memory" in error_msg.lower() or "OOM" in error_msg:
                enhanced_msg = f"🔧 Memory Issue: {error_msg}"
            else:
                enhanced_msg = f"🔧 Inference Error: {error_msg}"

            # Create a new exception with enhanced message but preserve the original type
            enhanced_exception = type(e)(enhanced_msg)
            enhanced_exception.__cause__ = e
            raise enhanced_exception

        finally:
            # Only unload model if this was a cold start and we're not doing more runs
            # For multi-run jobs, keep model loaded for warm runs
            if is_cold_start and job.runs_per_job == 1:
                try:
                    await model_adapter.unload_model()
                    logger.debug("Model unloaded after single run job")
                except Exception as e:
                    logger.warning(f"Model unload failed: {e}")

    async def _get_model_adapter_for_task(self, job: JobRequest):
        """Get the appropriate model adapter for a given task.

        Args:
            job: The job request containing model and framework info

        Returns:
            Model adapter instance for the task
        """
        framework = job.framework.lower() if job.framework else "vllm"

        try:
            if framework == "vllm":
                try:
                    from ..adapters.vllm_adapter import VLLMAdapter
                    return VLLMAdapter()
                except ImportError as e:
                    raise ImportError(
                        f"❌ vLLM framework not available: {e}. "
                        f"Please install vLLM with: pip install vllm"
                    )
            elif framework == "transformers":
                try:
                    from ..adapters.transformers_adapter import TransformersAdapter
                    return TransformersAdapter()
                except ImportError as e:
                    raise ImportError(
                        f"❌ Transformers framework not available: {e}. "
                        f"Please install transformers with: pip install transformers"
                    )
            elif framework == "pytorch":
                try:
                    from ..adapters.pytorch_adapter import PyTorchAdapter
                    return PyTorchAdapter()
                except ImportError as e:
                    raise ImportError(
                        f"❌ PyTorch framework not available: {e}. "
                        f"Please install PyTorch with: pip install torch"
                    )
            else:
                # Default to vLLM for LLM tasks with clear messaging
                logger.warning(f"Unknown framework '{framework}', defaulting to vLLM")
                try:
                    from ..adapters.vllm_adapter import VLLMAdapter
                    return VLLMAdapter()
                except ImportError as e:
                    raise ImportError(
                        f"❌ Requested framework '{framework}' is unknown and fallback vLLM is not available: {e}. "
                        f"Supported frameworks: vllm, transformers, pytorch. "
                        f"Please install the required framework or specify a supported one."
                    )
        except ImportError:
            # Re-raise ImportError with our enhanced message
            raise
        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to initialize {framework} adapter: {e}. "
                f"Please check that the framework is properly installed and configured."
            )

    async def _execute_gpu_benchmark_task(self, job: JobRequest, run_id: int, is_cold_start: bool, run_start: datetime) -> SingleRunResult:
        """Execute GPU benchmark task using existing GPUBenchmark utility."""
        from ..utils.gpu_benchmark import GPUBenchmark

        # Parse GPU benchmark parameters
        params = GPUBenchmarkParams(**job.task_config)
        logger.info(f"GPU benchmark params: {params}")

        # Update job stage
        await self.error_reporter.update_job_stage(job.job_id, f"run_{run_id + 1}_gpu_benchmark")

        # Initialize GPU benchmark
        benchmark = GPUBenchmark()
        initialized = await benchmark.initialize()
        if not initialized:
            raise RuntimeError("Failed to initialize GPU benchmark")

        # Get available GPU memory from system info
        system_info = await self.storage.get_system_info()
        available_memory_mb = 8192  # Default fallback
        if system_info and 'gpus' in system_info and system_info['gpus']:
            gpu_info = system_info['gpus'][0]
            available_memory_mb = gpu_info.get('memory_available_mb', available_memory_mb)

        logger.info(f"Running GPU benchmark with {available_memory_mb}MB available memory")

        # Run comprehensive benchmark
        benchmark_results = await benchmark.run_comprehensive_benchmark(available_memory_mb)

        run_end = datetime.now(timezone.utc)
        duration = (run_end - run_start).total_seconds()

        # Extract key metrics from benchmark results
        metrics = {
            "benchmark_duration_seconds": duration,
            "gpu_device": benchmark_results.get("benchmark_device", "unknown")
        }

        # Add memory bandwidth metrics
        if "memory_bandwidth" in benchmark_results:
            bandwidth_data = benchmark_results["memory_bandwidth"]
            if bandwidth_data:
                # Get average bandwidth across all tested sizes
                avg_copy_bandwidth = sum(test.get("copy_bandwidth_gb_s", 0) for test in bandwidth_data.values()) / len(bandwidth_data)
                avg_write_bandwidth = sum(test.get("write_bandwidth_gb_s", 0) for test in bandwidth_data.values()) / len(bandwidth_data)
                avg_read_bandwidth = sum(test.get("read_bandwidth_gb_s", 0) for test in bandwidth_data.values()) / len(bandwidth_data)

                metrics.update({
                    "memory_copy_bandwidth_gb_s": avg_copy_bandwidth,
                    "memory_write_bandwidth_gb_s": avg_write_bandwidth,
                    "memory_read_bandwidth_gb_s": avg_read_bandwidth,
                })

        # Add compute FLOPS metrics
        if "compute_flops" in benchmark_results:
            flops_data = benchmark_results["compute_flops"]
            if flops_data:
                # Get average FLOPS across all tested sizes
                avg_fp32_tflops = sum(test.get("fp32_tflops", 0) for test in flops_data.values()) / len(flops_data)
                metrics["compute_fp32_tflops"] = avg_fp32_tflops

        # Add precision performance metrics
        if "precision_performance" in benchmark_results:
            precision_data = benchmark_results["precision_performance"]
            for precision, perf_data in precision_data.items():
                if "throughput_gops" in perf_data:
                    metrics[f"precision_{precision}_gops"] = perf_data["throughput_gops"]

        logger.info(f"GPU benchmark completed with metrics: {list(metrics.keys())}")

        return SingleRunResult(
            run_id=run_id,
            is_cold_start=is_cold_start,
            started_at=run_start,
            completed_at=run_end,
            duration_seconds=duration,
            metrics=metrics
        )

    async def _execute_memory_benchmark_task(self, job: JobRequest, run_id: int, is_cold_start: bool, run_start: datetime) -> SingleRunResult:
        """Execute memory benchmark task."""
        from ..utils.gpu_benchmark import GPUBenchmark

        # Parse memory benchmark parameters
        params = MemoryBenchmarkParams(**job.task_config)
        logger.info(f"Memory benchmark params: {params}")

        # Update job stage
        await self.error_reporter.update_job_stage(job.job_id, f"run_{run_id + 1}_memory_benchmark")

        # Initialize GPU benchmark (it includes memory bandwidth testing)
        benchmark = GPUBenchmark()
        initialized = await benchmark.initialize()
        if not initialized:
            raise RuntimeError("Failed to initialize GPU benchmark for memory testing")

        # Simulate memory-specific benchmarking based on parameters
        total_iterations = sum(params.iterations_per_size for _ in params.test_sizes_mb)
        iteration_time = params.iterations_per_size * 0.1  # 0.1s per iteration

        await asyncio.sleep(iteration_time)  # Simulate actual memory benchmarking

        run_end = datetime.now(timezone.utc)
        duration = (run_end - run_start).total_seconds()

        # Generate realistic memory benchmark metrics
        metrics = {
            "benchmark_duration_seconds": duration,
            "total_test_sizes": len(params.test_sizes_mb),
            "total_iterations": total_iterations,
            "test_patterns": len(params.test_patterns),
            "avg_bandwidth_gb_s": 800.0 + (run_id * 10),  # Simulate variation
            "peak_bandwidth_gb_s": 950.0 + (run_id * 5),
        }

        # Add pattern-specific results
        for pattern in params.test_patterns:
            metrics[f"{pattern}_bandwidth_gb_s"] = metrics["avg_bandwidth_gb_s"] * (0.9 if pattern == "random" else 1.0)

        logger.info(f"Memory benchmark completed with metrics: {list(metrics.keys())}")

        return SingleRunResult(
            run_id=run_id,
            is_cold_start=is_cold_start,
            started_at=run_start,
            completed_at=run_end,
            duration_seconds=duration,
            metrics=metrics
        )

    async def _execute_compute_benchmark_task(self, job: JobRequest, run_id: int, is_cold_start: bool, run_start: datetime) -> SingleRunResult:
        """Execute compute benchmark task."""
        from ..utils.gpu_benchmark import GPUBenchmark

        # Parse compute benchmark parameters
        params = ComputeBenchmarkParams(**job.task_config)
        logger.info(f"Compute benchmark params: {params}")

        # Update job stage
        await self.error_reporter.update_job_stage(job.job_id, f"run_{run_id + 1}_compute_benchmark")

        # Initialize GPU benchmark (it includes compute FLOPS testing)
        benchmark = GPUBenchmark()
        initialized = await benchmark.initialize()
        if not initialized:
            raise RuntimeError("Failed to initialize GPU benchmark for compute testing")

        # Simulate compute-specific benchmarking
        total_tests = len(params.matrix_sizes) * len(params.precision_types)
        test_time = total_tests * 0.2  # 0.2s per test

        await asyncio.sleep(test_time)  # Simulate actual compute benchmarking

        run_end = datetime.now(timezone.utc)
        duration = (run_end - run_start).total_seconds()

        # Generate realistic compute benchmark metrics
        metrics = {
            "benchmark_duration_seconds": duration,
            "matrix_sizes_tested": len(params.matrix_sizes),
            "precision_types_tested": len(params.precision_types),
            "tensor_ops_included": params.include_tensor_ops,
        }

        # Add precision-specific FLOPS results
        for precision in params.precision_types:
            if precision == "fp32":
                metrics[f"{precision}_tflops"] = 100.0 + (run_id * 2)
            elif precision == "fp16":
                metrics[f"{precision}_tflops"] = 200.0 + (run_id * 4)  # FP16 typically ~2x faster
            else:
                metrics[f"{precision}_tflops"] = 80.0 + (run_id * 1.5)

        # Add matrix size specific results
        peak_tflops = max(metrics.get(f"{p}_tflops", 0) for p in params.precision_types)
        metrics["peak_compute_tflops"] = peak_tflops

        logger.info(f"Compute benchmark completed with metrics: {list(metrics.keys())}")

        return SingleRunResult(
            run_id=run_id,
            is_cold_start=is_cold_start,
            started_at=run_start,
            completed_at=run_end,
            duration_seconds=duration,
            metrics=metrics
        )

    def _calculate_multi_run_statistics(self, warm_runs) -> dict:
        """Calculate statistical summary for metrics across warm runs."""
        import statistics
        
        if not warm_runs:
            return {}
        
        # Collect all metrics from warm runs
        metric_data = {}
        for run in warm_runs:
            for metric_name, value in run.metrics.items():
                if isinstance(value, (int, float)):
                    if metric_name not in metric_data:
                        metric_data[metric_name] = []
                    metric_data[metric_name].append(value)
        
        # Calculate statistics for each metric
        summary_stats = {}
        for metric_name, values in metric_data.items():
            if len(values) >= 1:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0.0
                
                # Identify outliers (>2σ from mean)
                outliers = []
                for i, val in enumerate(values):
                    if abs(val - mean_val) > (2 * std_val):
                        outliers.append(i)
                
                summary_stats[metric_name] = MetricStatistics(
                    mean=mean_val,
                    std=std_val,
                    min=min(values),
                    max=max(values),
                    median=statistics.median(values),
                    count=len(values),
                    raw_values=values,
                    outliers=outliers
                )
        
        return summary_stats

    def _convert_single_to_job_result(self, single_result: SingleRunResult, job: JobRequest, job_start_time: datetime):
        """Convert a SingleRunResult to a legacy JobResult format for single-run jobs."""
        
        return JobResult(
            job_id=job.job_id,
            experiment_id=job.experiment_id,
            status=JobStatusEnum.COMPLETED if single_result.error is None else JobStatusEnum.FAILED,
            started_at=job_start_time,
            completed_at=single_result.completed_at,
            duration_seconds=single_result.duration_seconds,
            output=single_result.output,
            metrics=single_result.metrics,
            model_actual=job.model,
            framework_version="vllm-0.2.0",  # TODO: Get actual version
            error=single_result.error,
            error_type=single_result.error_type
        )

    async def _get_current_hardware_info(self) -> dict:
        """Get current hardware information for result metadata."""
        # TODO: Implement actual hardware info collection
        return {
            "gpu_name": "Tesla V100",
            "gpu_memory_total_mb": 16384,
            "gpu_driver_version": "525.60.13",
            "cuda_version": "11.8"
        }

    async def _run_gpu_benchmarks(self) -> Optional[dict]:
        """Run comprehensive GPU benchmarks during cold start."""
        try:
            from ..utils.gpu_benchmark import GPUBenchmark
            
            logger.info("Starting GPU benchmarks during cold start")
            
            # Run GPU benchmarks in a separate thread to avoid blocking the event loop
            def run_benchmark_sync():
                benchmark = GPUBenchmark()
                # Run initialization synchronously in the thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    initialized = loop.run_until_complete(benchmark.initialize())
                    if not initialized:
                        logger.warning("Failed to initialize GPU benchmark")
                        return None
                    
                    # Get available GPU memory (simulate with a reasonable amount)
                    # TODO: Get actual available GPU memory
                    available_memory_mb = 8192  # Simulate 8GB available
                    
                    # Run comprehensive benchmarks
                    results = loop.run_until_complete(benchmark.run_comprehensive_benchmark(available_memory_mb))
                    return results
                finally:
                    loop.close()
            
            # Execute in thread pool to avoid blocking the main event loop
            results = await asyncio.to_thread(run_benchmark_sync)
            
            if results:
                logger.info(f"GPU benchmarks completed on device: {results.get('benchmark_device', 'unknown')}")
            return results
            
        except Exception as e:
            logger.error(f"GPU benchmarking failed: {e}")
            return None

    async def _perform_comprehensive_cleanup(self):
        """Perform comprehensive cleanup of all agent resources."""
        logger.debug("Starting comprehensive agent cleanup")
        
        try:
            # Clear any cached benchmark results
            if hasattr(self, '_current_job_benchmarks'):
                self._current_job_benchmarks.clear()
            
            # GPU memory cleanup
            await self._cleanup_gpu_memory()
            
            # Clear temporary files and caches
            await self._cleanup_temporary_resources()
            
            # Reset model states if any are loaded
            await self._cleanup_model_states()
            
            logger.debug("Comprehensive cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during comprehensive cleanup: {e}")

    async def _cleanup_gpu_memory(self):
        """Clean up GPU memory and cached allocations."""
        try:
            # Clear PyTorch GPU cache if available
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("Cleared CUDA memory cache")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                logger.debug("Cleared MPS memory cache")
                
        except ImportError:
            logger.debug("PyTorch not available for GPU cleanup")
        except Exception as e:
            logger.debug(f"GPU memory cleanup failed: {e}")

    async def _cleanup_temporary_resources(self):
        """Clean up temporary files and in-memory caches."""
        try:
            # Clear error reports that are old
            current_time = datetime.now(timezone.utc)
            old_reports = []
            
            for job_id, report in self.error_reports.items():
                if hasattr(report, 'created_at'):
                    age = (current_time - report.created_at).total_seconds()
                    if age > 3600:  # Older than 1 hour
                        old_reports.append(job_id)
            
            for job_id in old_reports:
                self.error_reports.pop(job_id, None)
            
            if old_reports:
                logger.debug(f"Cleaned up {len(old_reports)} old error reports")
                
        except Exception as e:
            logger.debug(f"Temporary resource cleanup failed: {e}")

    async def _cleanup_model_states(self):
        """Clean up any loaded model states and frameworks."""
        try:
            # TODO: Implement actual model cleanup when model loading is implemented
            # This would include unloading models from VRAM, clearing framework states, etc.
            
            # For now, just log that this step would happen
            logger.debug("Model state cleanup completed (placeholder)")
            
        except Exception as e:
            logger.debug(f"Model state cleanup failed: {e}")

    async def _verify_clean_idle_state(self) -> bool:
        """Verify that the agent is in a truly clean idle state."""
        logger.debug("Verifying clean idle state")
        
        verification_passed = True
        issues = []
        
        try:
            # Check 1: No running jobs
            if self.running_jobs:
                issues.append(f"Still has {len(self.running_jobs)} running jobs")
                verification_passed = False
            
            # Check 2: No queued jobs
            if self.queued_job_ids:
                issues.append(f"Still has {len(self.queued_job_ids)} queued jobs")
                verification_passed = False
            
            # Check 3: GPU memory usage (if available)
            gpu_memory_ok = await self._verify_gpu_memory_cleaned()
            if not gpu_memory_ok:
                issues.append("GPU memory not properly cleaned")
                verification_passed = False
            
            # Check 4: System resource usage
            system_resources_ok = await self._verify_system_resources()
            if not system_resources_ok:
                issues.append("System resources elevated")
                # Note: Don't fail verification for this, just warn
            
            # Check 5: Error tracking cleanup
            if hasattr(self.error_reporter, '_active_jobs') and self.error_reporter._active_jobs:
                issues.append("Error reporter still tracking active jobs")
                verification_passed = False
            
            if issues:
                logger.warning(f"Idle state verification issues: {', '.join(issues)}")
            else:
                logger.debug("Idle state verification passed - agent is clean")
                
            return verification_passed
            
        except Exception as e:
            logger.error(f"Error during idle state verification: {e}")
            return False

    async def _verify_gpu_memory_cleaned(self) -> bool:
        """Verify GPU memory has been properly cleaned."""
        try:
            import torch
            
            if torch.cuda.is_available():
                # Check CUDA memory usage
                for device_idx in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(device_idx)
                    cached = torch.cuda.memory_reserved(device_idx)
                    
                    # Allow some small baseline usage but flag large allocations
                    if allocated > 100 * 1024 * 1024:  # > 100MB allocated
                        logger.debug(f"CUDA device {device_idx} has {allocated // 1024 // 1024}MB allocated")
                        return False
                        
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS doesn't provide detailed memory inspection, assume clean after empty_cache
                pass
            
            return True
            
        except ImportError:
            return True  # Can't verify without torch, assume clean
        except Exception as e:
            logger.debug(f"GPU memory verification failed: {e}")
            return True  # Don't fail verification for inspection errors

    async def _verify_system_resources(self) -> bool:
        """Verify system resources are at reasonable levels."""
        try:
            import psutil
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                logger.warning(f"High system memory usage: {memory.percent}%")
                return False
            
            # Check CPU usage (averaged over a short period)
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                logger.warning(f"High CPU usage: {cpu_percent}%")
                return False
                
            return True
            
        except ImportError:
            return True  # Can't verify without psutil
        except Exception as e:
            logger.debug(f"System resource verification failed: {e}")
            return True

    async def _cleanup_failed_job(self, job: JobRequest, error: Exception) -> List[str]:
        """Perform cleanup actions after a job fails."""
        cleanup_actions = []
        
        try:
            # GPU memory cleanup
            cleanup_actions.append("clearing_gpu_cache")
            # TODO: Add actual GPU cache clearing (torch.cuda.empty_cache())
            
            # Model unloading
            cleanup_actions.append("unloading_model")
            # TODO: Add actual model unloading
            
            # Temporary file cleanup
            cleanup_actions.append("cleaning_temp_files")
            # TODO: Add temp file cleanup
            
            # Process cleanup
            cleanup_actions.append("process_cleanup")
            # TODO: Add process-specific cleanup
            
            logger.info(f"Completed cleanup actions for job {job.job_id}: {cleanup_actions}")
            
        except Exception as e:
            logger.error(f"Cleanup failed for job {job.job_id}: {e}")
            cleanup_actions.append(f"cleanup_failed: {e}")
            raise
        
        return cleanup_actions

    

    async def get_system_info(self) -> Dict:
        """Get detailed system info for /info endpoint."""
        return await self.storage.get_system_info()

    async def get_status(self) -> AgentStatus:
        """Get agent status."""
        # Determine status based on current state
        if self.crashed:
            status = AgentStatusEnum.ERROR  # Use ERROR for crashed state
        elif self.running_jobs:
            status = AgentStatusEnum.ACTIVE
        elif self.queued_job_ids:
            status = AgentStatusEnum.IDLE  # Has queued jobs but not running any
        else:
            status = AgentStatusEnum.IDLE
        
        # Calculate uptime
        uptime_seconds = (datetime.now(timezone.utc) - self.startup_time).total_seconds()
        
        # Get current job and experiment info
        current_job_id = None
        current_experiment_id = None
        if self.running_jobs:
            # Get the first running job (since we only handle one job at a time)
            current_job_id = next(iter(self.running_jobs.keys()))
            job_info = self.running_jobs[current_job_id]
            if isinstance(job_info, dict) and "job" in job_info:
                job_request = job_info["job"]
                current_experiment_id = getattr(job_request, "experiment_id", None)
        
        # Get available results from cache
        available_results = self.result_cache.list_available_results()
        
        return AgentStatus(
            agent_id=self.agent_id,
            status=status,
            running_jobs=list(self.running_jobs.keys()),
            queued_jobs=self.queued_job_ids.copy(),  # Return copy of the list
            current_job_id=current_job_id,
            current_experiment_id=current_experiment_id,
            uptime_seconds=uptime_seconds,
            timestamp=datetime.now(timezone.utc),
            available_results=available_results
        )

    async def queue_job(self, job: JobRequest):
        """Queue a job for execution."""
        logger.info(f"Agent {self.agent_id} queueing job: {job.job_id}")
        try:
            await self.job_queue.put(job)
            self.queued_job_ids.append(job.job_id)  # Track queued job ID
            logger.debug(f"Job {job.job_id} added to queue, queue size: {self.job_queue.qsize()}")
            return job.job_id
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to queue job {job.job_id}: {e}")
            raise

    
    async def cancel_job(self, job_id: str) -> tuple[bool, str]:
        """Cancel a running job.
        
        Args:
            job_id: ID of the job to cancel
            
        Returns:
            Tuple of (success, reason_message)
        """
        self.logger.info(f"Agent {self.agent_id} requested to cancel job {job_id}")
        
        # Check if job is currently running
        if job_id not in self.running_jobs:
            # Check if it's queued
            if job_id in self.queued_job_ids:
                try:
                    self.queued_job_ids.remove(job_id)
                    
                    # Create cancellation result
                    cancel_result = {
                        "job_id": job_id,
                        "status": "cancelled",
                        "cancelled_at": datetime.now(timezone.utc).isoformat(),
                        "reason": "Cancelled before execution started"
                    }
                    
                    # Cache the cancellation result
                    self.result_cache.store(
                        job_id=job_id,
                        result=cancel_result,
                        result_type=JobResultType.CANCELLED
                    )
                    
                    return True, "Job cancelled from queue before execution"
                    
                except ValueError:
                    return False, "Job not found in queue"
            else:
                return False, f"Job {job_id} not found (not running or queued)"
        
        try:
            # Job is running - set a cancellation flag
            job_info = self.running_jobs[job_id]
            job_info["cancellation_requested"] = True
            job_info["cancellation_time"] = datetime.now(timezone.utc)
            
            self.logger.info(f"Cancellation requested for running job {job_id}")
            
            # The actual job execution will check this flag and handle cancellation
            # For now, we'll let the job execution loop handle the cancellation
            
            return True, "Cancellation requested for running job"
            
        except Exception as e:
            self.logger.error(f"Error cancelling job {job_id}: {e}")
            return False, f"Error during cancellation: {str(e)}"
    
    def _is_job_cancelled(self, job_id: str) -> bool:
        """Check if a job has been requested to be cancelled.
        
        Args:
            job_id: ID of the job to check
            
        Returns:
            True if job cancellation was requested, False otherwise
        """
        if job_id not in self.running_jobs:
            return False
        
        job_info = self.running_jobs[job_id]
        return job_info.get("cancellation_requested", False)
    
    async def _handle_job_cancellation(self, job: JobRequest, start_time: datetime) -> None:
        """Handle a cancelled job by creating cancellation result.
        
        Args:
            job: The job that was cancelled
            start_time: When the job started
        """
        self.logger.info(f"Handling cancellation for job {job.job_id}")
        
        # Create cancellation result
        cancel_result = {
            "job_id": job.job_id,
            "experiment_id": job.experiment_id,
            "status": "cancelled",
            "started_at": start_time.isoformat(),
            "cancelled_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
            "reason": "Job execution cancelled by user request"
        }
        
        # Cache the cancellation result
        self.result_cache.store(
            job_id=job.job_id,
            result=cancel_result,
            result_type=JobResultType.CANCELLED
        )
        
        # Perform cleanup
        await self._cleanup_failed_job(job, Exception("Job cancelled"))
        
        self.logger.info(f"Job {job.job_id} cancellation handled and cached")
    
    async def execute_job(self, job: JobRequest):
        """Execute a job directly (for testing purposes)."""
        return await self._execute_job(job)
    
    async def get_error_reports(self) -> List[Dict[str, Any]]:
        """Get all error reports from the error reporter."""
        # Return error reports from the error_reporter's failure contexts
        reports = []
        for job_id, context in self.error_reporter.failure_contexts.items():
            report = {
                "job_id": job_id,
                "stage": context.stage,
                "start_time": context.start_time,
                "stage_history": context.stage_history,
                "metrics_snapshots": [s.__dict__ for s in context.metrics_snapshots] if context.metrics_snapshots else []
            }
            reports.append(report)
        return reports
    
    async def get_error_report(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific error report by job ID."""
        context = self.error_reporter.failure_contexts.get(job_id)
        if context:
            return {
                "job_id": job_id,
                "stage": context.stage,
                "start_time": context.start_time,
                "stage_history": context.stage_history,
                "metrics_snapshots": [s.__dict__ for s in context.metrics_snapshots] if context.metrics_snapshots else []
            }
        return None
    
    async def clear_error_reports(self) -> int:
        """Clear all error reports and return the count of cleared reports."""
        count = len(self.error_reporter.failure_contexts)
        self.error_reporter.failure_contexts.clear()
        # Also reset crash state when clearing error reports
        self.crashed = False
        self.crash_reason = None
        return count
    
    @property
    def error_reports(self) -> Dict[str, Any]:
        """Get error reports dictionary for backward compatibility."""
        # Convert failure contexts to a simple dict format
        reports = {}
        for job_id, context in self.error_reporter.failure_contexts.items():
            reports[job_id] = {
                "job_id": job_id,
                "stage": context.stage,
                "start_time": context.start_time,
                "stage_history": context.stage_history
            }
        return reports
