import time
import threading
from typing import Dict, Any, Optional, List
from collections import deque
from .metrics_collector import collector


class InferenceMonitor:
    """Monitor inference latency, throughput, and system metrics."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies: deque = deque(maxlen=window_size)
        self.throughput_timestamps: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()

        self.total_inferences = 0
        self.start_time = time.time()

    def record_inference(self, latency_ms: float):
        """Record an inference with its latency."""
        with self._lock:
            self.latencies.append(latency_ms)
            self.throughput_timestamps.append(time.time())
            self.total_inferences += 1

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency percentiles."""
        with self._lock:
            if not self.latencies:
                return {}

            sorted_latencies = sorted(self.latencies)
            n = len(sorted_latencies)

            return {
                "p50": sorted_latencies[int(n * 0.5)],
                "p95": sorted_latencies[int(n * 0.95)] if n > 1 else sorted_latencies[0],
                "p99": sorted_latencies[int(n * 0.99)] if n > 1 else sorted_latencies[0],
                "mean": sum(sorted_latencies) / n,
                "min": sorted_latencies[0],
                "max": sorted_latencies[-1],
            }

    def get_throughput(self) -> float:
        """Get current throughput (inferences per second)."""
        with self._lock:
            if len(self.throughput_timestamps) < 2:
                return 0.0

            time_diff = self.throughput_timestamps[-1] - self.throughput_timestamps[0]
            if time_diff == 0:
                return 0.0

            return (len(self.throughput_timestamps) - 1) / time_diff

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics."""
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        metrics = {
            "cpu_usage": cpu_percent / 100.0,
            "memory_used_gb": memory.used / (1024 ** 3),
            "memory_total_gb": memory.total / (1024 ** 3),
            "memory_percent": memory.percent / 100.0,
        }

        try:
            import torch
            if torch.cuda.is_available():
                metrics["gpu_memory_used"] = torch.cuda.memory_allocated() / (1024 ** 3)
                metrics["gpu_memory_reserved"] = torch.cuda.memory_reserved() / (1024 ** 3)
        except ImportError:
            pass

        collector.set("cpu_usage", metrics["cpu_usage"])
        if "gpu_memory_used" in metrics:
            collector.set("gpu_memory_used", metrics["gpu_memory_used"])

        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get complete inference summary."""
        latency_stats = self.get_latency_stats()
        system_metrics = self.get_system_metrics()

        collector.set("inference_latency_p50", latency_stats.get("p50", 0))
        collector.set("inference_latency_p95", latency_stats.get("p95", 0))
        collector.set("inference_latency_p99", latency_stats.get("p99", 0))
        collector.set("throughput", self.get_throughput())

        return {
            "total_inferences": self.total_inferences,
            "uptime_seconds": time.time() - self.start_time,
            "latency": latency_stats,
            "throughput": self.get_throughput(),
            "system": system_metrics,
        }

    def reset(self):
        """Reset monitoring data."""
        with self._lock:
            self.latencies.clear()
            self.throughput_timestamps.clear()
            self.total_inferences = 0
            self.start_time = time.time()


inference_monitor = InferenceMonitor()