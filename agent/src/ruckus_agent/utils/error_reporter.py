"""Error reporting and failure diagnostics utilities."""

import asyncio
import logging
import subprocess
import traceback
import psutil
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from ..core.models import SystemMetricsSnapshot, JobErrorReport, JobFailureContext

logger = logging.getLogger(__name__)


class SystemMetricsCollector:
    """Collect comprehensive system metrics for error reporting."""
    
    @staticmethod
    async def capture_snapshot() -> SystemMetricsSnapshot:
        """Capture a comprehensive snapshot of current system metrics."""
        logger.debug("Capturing system metrics snapshot")
        
        snapshot = SystemMetricsSnapshot()
        
        try:
            # GPU metrics via nvidia-smi
            await SystemMetricsCollector._capture_gpu_metrics(snapshot)
            
            # System metrics via psutil
            await SystemMetricsCollector._capture_system_metrics(snapshot)
            
            # Process metrics
            await SystemMetricsCollector._capture_process_metrics(snapshot)
            
            logger.debug("System metrics snapshot captured successfully")
            
        except Exception as e:
            logger.error(f"Failed to capture complete metrics snapshot: {e}")
            # Return partial snapshot - better than nothing
        
        return snapshot
    
    @staticmethod
    async def _capture_gpu_metrics(snapshot: SystemMetricsSnapshot) -> None:
        """Capture GPU metrics using nvidia-smi."""
        try:
            # Run nvidia-smi with detailed query
            cmd = [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                lines = stdout.decode().strip().split('\n')
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        try:
                            # Parse values, handling 'N/A' entries
                            memory_used = int(parts[0]) if parts[0] != 'N/A' else 0
                            memory_total = int(parts[1]) if parts[1] != 'N/A' else 0
                            gpu_util = float(parts[2]) if parts[2] != 'N/A' else 0.0
                            temp = float(parts[3]) if parts[3] != 'N/A' else 0.0
                            power = float(parts[4]) if parts[4] != 'N/A' else 0.0
                            
                            snapshot.gpu_memory_used_mb.append(memory_used)
                            snapshot.gpu_memory_total_mb.append(memory_total)
                            snapshot.gpu_utilization_percent.append(gpu_util)
                            snapshot.gpu_temperature_c.append(temp)
                            snapshot.gpu_power_draw_w.append(power)
                            
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Failed to parse nvidia-smi line '{line}': {e}")
                            
            else:
                logger.warning(f"nvidia-smi failed: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"Failed to capture GPU metrics: {e}")
    
    @staticmethod
    async def _capture_system_metrics(snapshot: SystemMetricsSnapshot) -> None:
        """Capture system-level metrics."""
        try:
            # Memory info
            memory = psutil.virtual_memory()
            snapshot.system_memory_used_gb = (memory.total - memory.available) / (1024**3)
            snapshot.system_memory_total_gb = memory.total / (1024**3)
            
            # CPU utilization (average over short interval)
            snapshot.cpu_utilization_percent = psutil.cpu_percent(interval=0.1)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            snapshot.disk_usage_gb = disk.used / (1024**3)
            
        except Exception as e:
            logger.error(f"Failed to capture system metrics: {e}")
    
    @staticmethod
    async def _capture_process_metrics(snapshot: SystemMetricsSnapshot) -> None:
        """Capture current process metrics."""
        try:
            current_process = psutil.Process()
            
            # Memory usage
            memory_info = current_process.memory_info()
            snapshot.process_memory_mb = memory_info.rss / (1024**2)
            
            # CPU usage
            snapshot.process_cpu_percent = current_process.cpu_percent()
            
        except Exception as e:
            logger.error(f"Failed to capture process metrics: {e}")


class ErrorReporter:
    """Generate comprehensive error reports for job failures."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.metrics_collector = SystemMetricsCollector()
        self.failure_contexts: Dict[str, JobFailureContext] = {}
    
    async def start_job_tracking(self, job_id: str, stage: str = "initializing") -> JobFailureContext:
        """Start tracking a job for potential failure reporting."""
        logger.debug(f"Starting failure tracking for job {job_id} at stage '{stage}'")
        
        context = JobFailureContext(
            job_id=job_id,
            stage=stage,
            start_time=datetime.now(timezone.utc)
        )
        
        # Capture initial metrics
        initial_snapshot = await self.metrics_collector.capture_snapshot()
        context.metrics_snapshots.append(initial_snapshot)
        context.stage_history.append(stage)
        
        self.failure_contexts[job_id] = context
        return context
    
    async def update_job_stage(self, job_id: str, stage: str) -> None:
        """Update the current stage of a tracked job."""
        if job_id in self.failure_contexts:
            context = self.failure_contexts[job_id]
            context.stage = stage
            context.stage_history.append(f"{stage} ({datetime.now(timezone.utc)})")
            
            # Capture metrics at stage transition
            snapshot = await self.metrics_collector.capture_snapshot()
            context.metrics_snapshots.append(snapshot)
            
            logger.debug(f"Updated job {job_id} to stage '{stage}'")
    
    async def generate_error_report(
        self,
        job_id: str,
        experiment_id: str,
        error: Exception,
        model_name: str,
        model_path: str,
        framework: str,
        task_type: str,
        parameters: Dict[str, Any],
        started_at: datetime,
        model_size_gb: Optional[float] = None
    ) -> JobErrorReport:
        """Generate a comprehensive error report for a failed job."""
        
        logger.info(f"Generating error report for job {job_id}")
        
        # Capture failure-time metrics
        failure_snapshot = await self.metrics_collector.capture_snapshot()
        
        # Get failure context if available
        context = self.failure_contexts.get(job_id)
        before_snapshot = None
        if context and context.metrics_snapshots:
            before_snapshot = context.metrics_snapshots[-1]  # Most recent pre-failure snapshot
        
        # Determine error type
        error_type = self._classify_error(error, failure_snapshot)
        
        # Check for CUDA out of memory
        cuda_oom = "out of memory" in str(error).lower() or "cuda" in str(error).lower()
        
        # Capture additional diagnostic info
        nvidia_smi_output = await self._capture_nvidia_smi_full()
        available_vram = self._estimate_available_vram(failure_snapshot)
        
        # Calculate failure duration
        failed_at = datetime.now(timezone.utc)
        duration = (failed_at - started_at).total_seconds()
        
        # Create error report
        error_report = JobErrorReport(
            job_id=job_id,
            experiment_id=experiment_id,
            agent_id=self.agent_id,
            
            # Error details
            error_type=error_type,
            error_message=str(error),
            error_traceback=self._format_error_traceback(error),
            
            # Job context
            model_name=model_name,
            model_path=model_path,
            framework=framework,
            task_type=task_type,
            parameters=parameters,
            
            # Timing
            started_at=started_at,
            failed_at=failed_at,
            duration_before_failure_seconds=duration,
            
            # System state
            metrics_at_failure=failure_snapshot,
            metrics_before_failure=before_snapshot,
            
            # Diagnostic info
            model_size_gb=model_size_gb,
            available_vram_mb=available_vram,
            cuda_out_of_memory=cuda_oom,
            nvidia_smi_output=nvidia_smi_output
        )
        
        logger.info(f"Error report generated for job {job_id}: {error_type}")
        return error_report
    
    async def cleanup_job_tracking(self, job_id: str) -> None:
        """Clean up tracking for a completed/failed job."""
        if job_id in self.failure_contexts:
            del self.failure_contexts[job_id]
            logger.debug(f"Cleaned up tracking for job {job_id}")
    
    def _format_error_traceback(self, error: Exception) -> str:
        """Format error traceback, handling both active exceptions and standalone errors."""
        try:
            # First try to get the current traceback if we're in an exception context
            current_tb = traceback.format_exc()
            if current_tb != "NoneType: None\n":
                return current_tb
            
            # If no active exception, create one from the error
            error_type = type(error).__name__
            error_msg = str(error)
            return f"{error_type}: {error_msg}\n"
        except Exception:
            # Fallback
            return f"{type(error).__name__}: {str(error)}\n"
    
    def _classify_error(self, error: Exception, snapshot: SystemMetricsSnapshot) -> str:
        """Classify the type of error based on exception and system state."""
        error_str = str(error).lower()
        
        # CUDA/GPU errors
        if "cuda" in error_str and "out of memory" in error_str:
            return "cuda_out_of_memory"
        elif "cuda" in error_str:
            return "cuda_error"
        
        # Memory errors
        if "memory" in error_str or "oom" in error_str:
            return "out_of_memory"
        
        # Model loading errors
        if "load" in error_str or "checkpoint" in error_str or "safetensors" in error_str:
            return "model_loading_error"
        
        # Timeout errors
        if "timeout" in error_str or isinstance(error, asyncio.TimeoutError):
            return "timeout"
        
        # Import/dependency errors
        if isinstance(error, ImportError) or "import" in error_str:
            return "dependency_error"
        
        # File/path errors
        if isinstance(error, (FileNotFoundError, PermissionError)) or "file" in error_str:
            return "file_error"
        
        # Network errors
        if "connection" in error_str or "network" in error_str:
            return "network_error"
        
        # Default
        return "unknown_error"
    
    async def _capture_nvidia_smi_full(self) -> Optional[str]:
        """Capture full nvidia-smi output for diagnostics."""
        try:
            result = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                return stdout.decode()
            else:
                return f"nvidia-smi failed: {stderr.decode()}"
                
        except Exception as e:
            return f"Failed to run nvidia-smi: {e}"
    
    def _estimate_available_vram(self, snapshot: SystemMetricsSnapshot) -> Optional[int]:
        """Estimate available VRAM from snapshot."""
        if snapshot.gpu_memory_total_mb and snapshot.gpu_memory_used_mb:
            # Return available VRAM for GPU with most available memory
            available_per_gpu = [
                total - used for total, used in 
                zip(snapshot.gpu_memory_total_mb, snapshot.gpu_memory_used_mb)
            ]
            return max(available_per_gpu) if available_per_gpu else None
        return None