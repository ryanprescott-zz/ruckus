"""Performance metric collection."""

import time
from typing import Dict, Any, Optional
from datetime import datetime
from .base import MetricCollector


class PerformanceCollector(MetricCollector):
    """Collect performance metrics."""

    def __init__(self):
        super().__init__("performance")
        self.start_time = None
        self.first_token_time = None
        self.token_times = []

    def mark_first_token(self):
        """Mark when first token is generated."""
        if self.start_time and not self.first_token_time:
            self.first_token_time = time.perf_counter()

    def mark_token(self):
        """Mark token generation."""
        self.token_times.append(time.perf_counter())

    async def collect(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        metrics = {}

        # Latency
        if self.start_time:
            end_time = time.perf_counter()
            metrics["total_latency_ms"] = (end_time - self.start_time) * 1000

            # Time to first token
            if self.first_token_time:
                metrics["time_to_first_token_ms"] = (
                                                            self.first_token_time - self.start_time
                                                    ) * 1000

            # Throughput
            if len(self.token_times) > 1:
                total_time = self.token_times[-1] - self.token_times[0]
                if total_time > 0:
                    metrics["tokens_per_second"] = len(self.token_times) / total_time

        self.metrics.update(metrics)
        return metrics

    def start(self):
        """Start performance measurement."""
        super().start()
        self.start_time = time.perf_counter()
        self.first_token_time = None
        self.token_times = []