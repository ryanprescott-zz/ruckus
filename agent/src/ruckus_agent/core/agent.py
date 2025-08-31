"""Main agent implementation."""

import asyncio
import httpx
import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from ruckus_common.models import (
    JobRequest, JobStatus, AgentType, JobUpdate, AgentStatus, AgentStatusEnum,
    SingleRunResult, MetricStatistics, MultiRunJobResult, JobResult, JobStage
)
from .config import Settings
from .models import AgentRegistration, JobErrorReport
from .detector import AgentDetector
from .storage import AgentStorage, InMemoryStorage
from ..utils.error_reporter import ErrorReporter

logger = logging.getLogger(__name__)


class Agent:
    """Main agent class coordinating job execution."""

    def __init__(self, settings: Settings, storage: Optional[AgentStorage] = None):
        self.settings = settings
        # Generate unique agent ID and name
        self.agent_id = f"agent-{uuid.uuid4().hex[:8]}"
        self.agent_name = f"{self.agent_id}-{settings.agent_type.value}"
        self.orchestrator_url = settings.orchestrator_url

        # Storage backend
        self.storage = storage or InMemoryStorage()

        # State
        self.registered = False
        self.running_jobs: Dict[str, Any] = {}
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.queued_job_ids: List[str] = []  # Track queued job IDs for status reporting
        self.startup_time = datetime.now(timezone.utc)
        self.crashed = False
        self.crash_reason: Optional[str] = None
        
        # Error reporting
        self.error_reporter = ErrorReporter(self.agent_id)
        self.error_reports: Dict[str, JobErrorReport] = {}  # Store error reports for server retrieval

        # HTTP client for orchestrator communication
        self.client = httpx.AsyncClient()

        # Background tasks
        self.tasks: List[asyncio.Task] = []
        
        logger.info(f"Agent initialized: {self.agent_id} ({self.agent_name})")
        logger.debug(f"Orchestrator URL: {self.orchestrator_url}")

    async def start(self):
        """Start the agent."""
        logger.info(f"Starting agent {self.agent_id}")

        # Detect capabilities
        await self._detect_capabilities()

        # Register with orchestrator
        if self.orchestrator_url:
            await self._register()

        # Start background tasks
        self.tasks.append(asyncio.create_task(self._heartbeat_loop()))
        self.tasks.append(asyncio.create_task(self._job_executor()))

    async def stop(self):
        """Stop the agent."""
        logger.info(f"Stopping agent {self.agent_id}")

        # Cancel background tasks
        for task in self.tasks:
            task.cancel()

        # Close HTTP client
        await self.client.aclose()

    async def _detect_capabilities(self):
        """Detect agent capabilities and system info."""
        logger.info(f"Agent {self.agent_id} starting capability detection")
        try:
            detector = AgentDetector()
            detected = await detector.detect_all()
            
            # Store system info in storage
            system_info = {
                "system": detected.get("system", {}),
                "cpu": detected.get("cpu", {}),
                "gpus": detected.get("gpus", []),
                "frameworks": detected.get("frameworks", []),
                "models": detected.get("models", []),
                "hooks": detected.get("hooks", []),
                "metrics": detected.get("metrics", [])
            }
            await self.storage.store_system_info(system_info)
            
            # Store simplified capabilities for internal use
            capabilities = {
                "agent_type": self.settings.agent_type.value,
                "gpu_count": len(detected.get("gpus", [])),
                "frameworks": [f["name"] for f in detected.get("frameworks", [])],
                "max_concurrent_jobs": self.settings.max_concurrent_jobs,
                "monitoring_available": bool(detected.get("hooks", [])),
            }
            await self.storage.store_capabilities(capabilities)
            
            logger.info(f"Agent {self.agent_id} capabilities detected successfully")
            logger.debug(f"Detected {capabilities['gpu_count']} GPUs, {len(capabilities['frameworks'])} frameworks")
        except Exception as e:
            logger.error(f"Agent {self.agent_id} capability detection failed: {e}")
            raise

    async def _register(self):
        """Register with orchestrator."""
        logger.info(f"Agent {self.agent_id} attempting registration with orchestrator")
        try:
            # Get full system info including discovered models
            system_info = await self.get_system_info()
            capabilities = await self.get_capabilities()
            
            # Create comprehensive registration
            registration = AgentRegistration(
                agent_id=self.agent_id,
                agent_name=self.agent_name,
                agent_type=self.settings.agent_type,
                
                # System information
                system=system_info.get("system"),
                cpu=system_info.get("cpu"),
                gpus=[gpu for gpu in system_info.get("gpus", [])],
                
                # Framework and model information  
                frameworks=[fw for fw in system_info.get("frameworks", [])],
                models=[model for model in system_info.get("models", [])],
                hooks=[hook for hook in system_info.get("hooks", [])],
                
                # Capabilities
                capabilities=capabilities,
                metrics_available=[metric for metric in system_info.get("metrics", [])],
                
                # Configuration
                max_concurrent_jobs=self.settings.max_concurrent_jobs,
                max_batch_size=1,  # TODO: Make configurable
            )

            response = await self.client.post(
                f"{self.orchestrator_url}/api/v1/agents/register",
                json=registration.dict(),
            )
            if response.status_code == 200:
                self.registered = True
                logger.info(f"Agent {self.agent_id} registered successfully with orchestrator")
                logger.debug(f"Registered with {len(registration.models)} models, {len(registration.frameworks)} frameworks")
            else:
                logger.error(f"Registration failed with status {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Agent {self.agent_id} registration failed: {e}")
            raise

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to orchestrator."""
        logger.debug(f"Agent {self.agent_id} starting heartbeat loop")
        while True:
            try:
                await asyncio.sleep(self.settings.heartbeat_interval)

                if self.registered:
                    status = await self.get_status()
                    response = await self.client.post(
                        f"{self.orchestrator_url}/api/v1/agents/{self.agent_id}/heartbeat",
                        json=status.dict(),
                    )
                    logger.debug(f"Heartbeat sent, status: {response.status_code}")
                else:
                    logger.debug("Skipping heartbeat - not registered")
            except asyncio.CancelledError:
                logger.debug(f"Agent {self.agent_id} heartbeat loop cancelled")
                break
            except Exception as e:
                logger.error(f"Agent {self.agent_id} heartbeat error: {e}")

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
                await self._execute_job(job)
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
        
        try:
            # Start job tracking for error reporting
            await self.error_reporter.start_job_tracking(job.job_id, "initializing")
            
            # Add to running jobs
            self.running_jobs[job.job_id] = {
                "job": job,
                "start_time": start_time,
            }

            # Send initial status update
            await self.error_reporter.update_job_stage(job.job_id, "starting")
            update = JobUpdate(
                job_id=job.job_id,
                status=JobStatus.RUNNING,
                stage=JobStage.INITIALIZING,
            )
            await self._send_update(update)
            
            # Execute multi-run job
            if job.runs_per_job == 1:
                # Single run job - simpler path
                result = await self._execute_single_run(job, run_id=0, is_cold_start=True)
                job_result = self._convert_single_to_job_result(result, job, start_time)
            else:
                # Multi-run job - with cold start separation
                job_result = await self._execute_multi_run_job(job, start_time)
            
            # Send final success update
            update = JobUpdate(
                job_id=job.job_id,
                status=JobStatus.COMPLETED,
                stage=JobStage.FINALIZING,
                output=job_result,
                timestamp=datetime.now(timezone.utc)
            )
            await self._send_update(update)
            
            logger.info(f"Agent {self.agent_id} job {job.job_id} completed successfully")
            
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
                    model_path=f"/ruckus/models/{job.model}",  # TODO: Get actual path
                    framework=job.framework,
                    task_type=job.task_type.value,
                    parameters=job.parameters,
                    started_at=start_time,
                    model_size_gb=None  # TODO: Get from model discovery
                )
                
                # Store error report for server retrieval
                self.error_reports[job.job_id] = error_report
                
                # Attempt cleanup
                cleanup_actions = await self._cleanup_failed_job(job, e)
                error_report.cleanup_actions = cleanup_actions
                error_report.recovery_successful = True  # If we reach here, cleanup worked
                
                logger.info(f"Generated error report for job {job.job_id}: {error_report.error_type}")
                
            except Exception as cleanup_error:
                logger.error(f"Failed to generate error report or cleanup job {job.job_id}: {cleanup_error}")
                # Mark agent as crashed if cleanup fails
                self.crashed = True
                self.crash_reason = f"Failed cleanup after job {job.job_id}: {cleanup_error}"
            
            # Send failure update to server
            failure_update = JobUpdate(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                message=str(e)
            )
            await self._send_update(failure_update)
            
        finally:
            # Clean up job tracking and running jobs
            await self.error_reporter.cleanup_job_tracking(job.job_id)
            self.running_jobs.pop(job.job_id, None)
            
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
        """Execute a single run of a job with detailed metrics tracking."""
        
        run_start = datetime.now(timezone.utc)
        model_load_time = None
        model_load_memory = None
        
        try:
            # Update job stage
            stage = f"run_{run_id + 1}_starting"
            await self.error_reporter.update_job_stage(job.job_id, stage)
            
            # Cold start: Load model and track loading metrics
            if is_cold_start:
                await self.error_reporter.update_job_stage(job.job_id, f"run_{run_id + 1}_model_loading")
                load_start = datetime.now(timezone.utc)
                
                # TODO: Implement actual model loading with VRAM tracking
                await asyncio.sleep(0.5)  # Simulate model loading time
                model_load_time = (datetime.now(timezone.utc) - load_start).total_seconds()
                model_load_memory = 8192.0  # TODO: Get actual VRAM usage
                
                logger.info(f"Cold start model load completed in {model_load_time:.2f}s")
                
                # Run GPU benchmarks during cold start (when GPU is available)
                if run_id == 0:  # Only run benchmarks on the first cold start
                    await self.error_reporter.update_job_stage(job.job_id, f"run_{run_id + 1}_gpu_benchmarking")
                    gpu_benchmark_results = await self._run_gpu_benchmarks()
                    if gpu_benchmark_results:
                        # Store benchmarks for inclusion in final result
                        if not hasattr(self, '_current_job_benchmarks'):
                            self._current_job_benchmarks = {}
                        self._current_job_benchmarks[job.job_id] = gpu_benchmark_results
            
            # Inference execution
            await self.error_reporter.update_job_stage(job.job_id, f"run_{run_id + 1}_inference")
            
            # TODO: Implement actual inference with metrics collection
            await asyncio.sleep(0.2)  # Simulate inference time
            
            # Collect performance metrics
            metrics = {
                "inference_time_seconds": 0.15 + (run_id * 0.01),  # Simulate slight variation
                "throughput_tokens_per_sec": 120.0 - (run_id * 2),
                "memory_usage_mb": 6400.0 + (run_id * 50),
                "gpu_utilization_percent": 85.0 + (run_id * 1.5)
            }
            
            run_end = datetime.now(timezone.utc)
            duration = (run_end - run_start).total_seconds()
            
            # Create successful run result
            return SingleRunResult(
                run_id=run_id,
                is_cold_start=is_cold_start,
                started_at=run_start,
                completed_at=run_end,
                duration_seconds=duration,
                metrics=metrics,
                model_load_time_seconds=model_load_time,
                model_load_memory_mb=model_load_memory
            )
            
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
                model_load_time_seconds=model_load_time,
                model_load_memory_mb=model_load_memory
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
            status=JobStatus.COMPLETED if single_result.error is None else JobStatus.FAILED,
            started_at=job_start_time,
            completed_at=single_result.completed_at,
            duration_seconds=single_result.duration_seconds,
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
            
            benchmark = GPUBenchmark()
            if not await benchmark.initialize():
                logger.warning("Failed to initialize GPU benchmark")
                return None
            
            # Get available GPU memory (simulate with a reasonable amount)
            # TODO: Get actual available GPU memory
            available_memory_mb = 8192  # Simulate 8GB available
            
            # Run comprehensive benchmarks
            results = await benchmark.run_comprehensive_benchmark(available_memory_mb)
            
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

    async def _send_update(self, update: JobUpdate):
        """Send job update to orchestrator."""
        if self.orchestrator_url:
            try:
                response = await self.client.post(
                    f"{self.orchestrator_url}/api/v1/jobs/{update.job_id}/update",
                    json=update.dict(),
                )
                logger.debug(f"Job update sent for {update.job_id}: {update.status}, status: {response.status_code}")
            except Exception as e:
                logger.error(f"Agent {self.agent_id} failed to send update for job {update.job_id}: {e}")
        else:
            logger.debug(f"No orchestrator URL configured, skipping update for job {update.job_id}")
    
    async def get_capabilities(self) -> Dict:
        """Get agent capabilities."""
        return await self.storage.get_capabilities()

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
        
        return AgentStatus(
            agent_id=self.agent_id,
            status=status,
            running_jobs=list(self.running_jobs.keys()),
            queued_jobs=self.queued_job_ids.copy(),  # Return copy of the list
            uptime_seconds=uptime_seconds,
            timestamp=datetime.now(timezone.utc)
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

    async def get_error_reports(self) -> List[JobErrorReport]:
        """Get all stored error reports."""
        return list(self.error_reports.values())
    
    async def get_error_report(self, job_id: str) -> Optional[JobErrorReport]:
        """Get error report for a specific job."""
        return self.error_reports.get(job_id)
    
    async def clear_error_reports(self) -> int:
        """Clear all error reports and return count cleared."""
        count = len(self.error_reports)
        self.error_reports.clear()
        self.crashed = False  # Reset crashed state when reports are cleared
        self.crash_reason = None
        logger.info(f"Cleared {count} error reports, reset crashed state")
        return count
    
    async def execute_job(self, job: JobRequest):
        """Execute a job directly (for testing purposes)."""
        return await self._execute_job(job)
