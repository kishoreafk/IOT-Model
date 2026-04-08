import time
import threading
from typing import Dict, Any, Optional, List
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MetricDefinition:
    """Metric definition with metadata."""
    name: str
    type: str
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Thread-safe Prometheus-compatible metrics collector."""

    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._metric_definitions: Dict[str, MetricDefinition] = {}
        self._lock = threading.Lock()
        self._start_time = time.time()

        self._init_default_metrics()

    def _init_default_metrics(self):
        """Initialize default metrics."""
        default_metrics = [
            MetricDefinition("accuracy", "gauge", "Model accuracy score"),
            MetricDefinition("precision", "gauge", "Model precision score"),
            MetricDefinition("recall", "gauge", "Model recall score"),
            MetricDefinition("f1_score", "gauge", "Model F1 score"),
            MetricDefinition("novelty_detection_rate", "gauge", "Rate of novel detections"),
            MetricDefinition("false_novelty_rate", "gauge", "False novelty detection rate"),
            MetricDefinition("ece", "gauge", "Expected Calibration Error"),
            MetricDefinition("mce", "gauge", "Maximum Calibration Error"),
            MetricDefinition("inference_latency_p50", "gauge", "P50 inference latency (ms)"),
            MetricDefinition("inference_latency_p95", "gauge", "P95 inference latency (ms)"),
            MetricDefinition("inference_latency_p99", "gauge", "P99 inference latency (ms)"),
            MetricDefinition("throughput", "gauge", "Inference throughput (images/sec)"),
            MetricDefinition("signature_success_rate", "gauge", "RSA signature success rate"),
            MetricDefinition("encryption_success_rate", "gauge", "Encryption success rate"),
            MetricDefinition("mmd_drift", "gauge", "Maximum Mean Discrepancy drift score"),
            MetricDefinition("ks_drift", "gauge", "Kolmogorov-Smirnov drift score"),
            MetricDefinition("chi_sq_drift", "gauge", "Chi-squared drift score"),
            MetricDefinition("active_alerts", "gauge", "Number of active alerts"),
            MetricDefinition("total_embeddings", "gauge", "Total embeddings in index"),
            MetricDefinition("total_devices", "gauge", "Total connected devices"),
            MetricDefinition("total_clusters", "gauge", "Total number of clusters"),
            MetricDefinition("total_experts", "gauge", "Total number of MoE experts"),
            MetricDefinition("cpu_usage", "gauge", "CPU usage percentage"),
            MetricDefinition("gpu_memory_used", "gauge", "GPU memory used (GB)"),
            MetricDefinition("lora_adaptation_delta", "gauge", "LoRA adaptation accuracy delta"),
            MetricDefinition("training_loss", "gauge", "Current training loss"),
            MetricDefinition("validation_loss", "gauge", "Current validation loss"),
        ]

        for metric in default_metrics:
            self._metric_definitions[metric.name] = metric

    def inc(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] += value

    def set(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge value."""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value for histogram."""
        key = self._make_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)

    def get(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current value for a metric."""
        key = self._make_key(name, labels)
        with self._lock:
            if key in self._counters:
                return self._counters[key]
            if key in self._gauges:
                return self._gauges[key]
        return None

    def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics (p50, p95, p99)."""
        key = self._make_key(name, labels)
        with self._lock:
            values = self._histograms.get(key, [])
            if not values:
                return {}

            sorted_values = sorted(values)
            n = len(sorted_values)

            return {
                "count": n,
                "sum": sum(values),
                "min": min(values),
                "max": max(values),
                "p50": sorted_values[int(n * 0.5)],
                "p95": sorted_values[int(n * 0.95)] if n > 1 else sorted_values[0],
                "p99": sorted_values[int(n * 0.99)] if n > 1 else sorted_values[0],
                "mean": sum(values) / n,
            }

    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create metric key with labels."""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def list_metrics(self) -> List[str]:
        """List all registered metric names."""
        with self._lock:
            return list(set(
                k.split("{")[0] for k in list(self._counters.keys()) + list(self._gauges.keys())
            ))

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as dictionary."""
        with self._lock:
            result = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    k: self.get_histogram_stats(k.split("{")[0], None)
                    for k in self._histograms.keys()
                },
            }
        return result

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        lines.append(f"# HELP uptime_seconds Hub uptime in seconds")
        lines.append(f"# TYPE uptime_seconds gauge")
        lines.append(f"uptime_seconds {time.time() - self._start_time}")

        with self._lock:
            for key, value in self._counters.items():
                lines.append(f"{key} {value}")

            for key, value in self._gauges.items():
                lines.append(f"{key} {value}")

        return "\n".join(lines)

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


collector = MetricsCollector()