"""Resource utilization metrics."""

from typing import Dict, Any, Optional
import psutil
from .base import MetricCollector


class ResourceCollector(MetricCollector):
    """Collect resource utilization metrics."""

    def __init__(self):
        super().__init__("resource")
        self.initial_memory = None
        self.peak_memory = None

    def start(self):
        """Start resource monitoring."""
        super().start()
        self.initial_memory = psutil.virtual_memory().used
        self.peak_memory = self.initial_memory

    async def collect(self) -> Dict[str, Any]:
        """Collect resource metrics."""
        metrics = {}

        # Memory metrics
        current_memory = psutil.virtual_memory()
        metrics["memory_used_mb"] = current_memory.used / (1024 * 1024)
        metrics["memory_percent"] = current_memory.percent

        if self.initial_memory:
            memory_delta = current_memory.used - self.initial_memory
            metrics["memory_delta_mb"] = memory_delta / (1024 * 1024)

            # Track peak
            if current_memory.used > self.peak_memory:
                self.peak_memory = current_memory.used
            metrics["memory_peak_mb"] = self.peak_memory / (1024 * 1024)

        # CPU metrics
        metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)

        # GPU metrics (if available)
        gpu_metrics = await self._collect_gpu_metrics()
        if gpu_metrics:
            metrics.update(gpu_metrics)

        self.metrics.update(metrics)
        return metrics

    async def _collect_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect GPU metrics if available."""
        try:
            import pynvml
            pynvml.nvmlInit()

            metrics = {}
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics[f"gpu_{i}_memory_used_mb"] = mem_info.used / (1024 * 1024)
                metrics[f"gpu_{i}_memory_total_mb"] = mem_info.total / (1024 * 1024)

                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics[f"gpu_{i}_utilization_percent"] = util.gpu

                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics[f"gpu_{i}_temperature_c"] = temp

                # Power
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
                metrics[f"gpu_{i}_power_w"] = power

            return metrics

        except Exception:
            return None