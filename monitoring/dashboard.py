import time
from typing import Dict, Any, List, Optional
from fastapi import APIRouter
from .metrics_collector import collector
from .model_monitor import ModelPerformanceMonitor, AdaptationMonitor
from .inference_monitor import InferenceMonitor
from .security_monitor import SecurityMonitor
from .drift_detector import DriftDetector
from .alerting import AlertManager


class MonitoringDashboard:
    """Monitoring dashboard orchestrator combining all monitoring components."""

    def __init__(self):
        self.model_monitor = ModelPerformanceMonitor()
        self.adaptation_monitor = AdaptationMonitor()
        self.inference_monitor = InferenceMonitor()
        self.security_monitor = SecurityMonitor()
        self.drift_detector = DriftDetector()
        self.alert_manager = AlertManager()

        self.start_time = time.time()

    def get_status(self) -> Dict[str, Any]:
        """Get overall monitoring system status."""
        return {
            "status": "operational",
            "uptime_seconds": time.time() - self.start_time,
            "components": {
                "model_monitor": True,
                "adaptation_monitor": True,
                "inference_monitor": True,
                "security_monitor": True,
                "drift_detector": True,
                "alert_manager": True,
            },
        }

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get complete system overview."""
        inference_summary = self.inference_monitor.get_summary()
        security_summary = self.security_monitor.get_summary()
        adaptation_summary = self.adaptation_monitor.get_summary()
        alert_status = self.alert_manager.get_status()
        model_metrics = self.model_monitor.compute_metrics()

        return {
            "status": "operational",
            "timestamp": time.time(),
            "model_performance": model_metrics,
            "inference": {
                "latency_p50": inference_summary["latency"].get("p50", 0),
                "latency_p95": inference_summary["latency"].get("p95", 0),
                "latency_p99": inference_summary["latency"].get("p99", 0),
                "throughput": inference_summary.get("throughput", 0),
            },
            "system": inference_summary.get("system", {}),
            "security": {
                "signature_rate": security_summary["signature_verification"]["success_rate"],
                "encryption_rate": security_summary["encryption_operations"]["success_rate"],
            },
            "adaptation": adaptation_summary,
            "alerts": alert_status,
        }

    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        metrics = self.model_monitor.compute_metrics()
        return {
            "accuracy": metrics.get("accuracy", 0),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "f1": metrics.get("f1", 0),
            "ece": metrics.get("ece", 0),
            "mce": metrics.get("mce", 0),
            "confusion_matrix": self.model_monitor.get_confusion_matrix().tolist(),
            "per_class": self.model_monitor.get_per_class_metrics(),
        }

    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get LoRA adaptation statistics."""
        return self.adaptation_monitor.get_summary()

    def get_inference_health(self) -> Dict[str, Any]:
        """Get inference health metrics."""
        return self.inference_monitor.get_summary()

    def get_security_audit(self) -> Dict[str, Any]:
        """Get security audit summary."""
        return self.security_monitor.get_summary()

    def get_drift_report(self) -> Dict[str, Any]:
        """Get drift detection report."""
        return self.drift_detector.get_status()

    def get_alerts(self) -> Dict[str, Any]:
        """Get all alerts (active and history)."""
        return {
            "active": self.alert_manager.get_active_alerts(),
            "history": self.alert_manager.get_alert_history(),
            "status": self.alert_manager.get_status(),
        }

    def get_calibration(self) -> Dict[str, Any]:
        """Get calibration metrics."""
        metrics = self.model_monitor.compute_metrics()
        return {
            "ece": metrics.get("ece", 0),
            "mce": metrics.get("mce", 0),
        }

    def get_novelty(self) -> Dict[str, Any]:
        """Get novelty detection metrics."""
        return {
            "novel_detection_rate": collector.get("novelty_detection_rate", 0),
            "false_novelty_rate": collector.get("false_novelty_rate", 0),
        }

    def get_confusion_matrix(self) -> List[List[int]]:
        """Get confusion matrix."""
        return self.model_monitor.get_confusion_matrix().tolist()

    def get_per_class(self) -> Dict[int, Dict[str, float]]:
        """Get per-class metrics."""
        return self.model_monitor.get_per_class_metrics()

    def check_drift(self, embeddings: Any = None, confidences: Any = None, cluster_counts: Any = None) -> Dict[str, Any]:
        """Manually trigger drift check."""
        import numpy as np
        emb = np.array(embeddings) if embeddings is not None else None
        conf = np.array(confidences) if confidences is not None else None
        counts = np.array(cluster_counts) if cluster_counts is not None else None
        return self.drift_detector.detect_all_drifts(emb, conf, counts)

    def evaluate_alerts(self):
        """Manually trigger alert evaluation."""
        return self.alert_manager.evaluate_rules()


dashboard = MonitoringDashboard()
router = APIRouter()


@router.get("/status")
async def monitoring_status():
    return dashboard.get_status()


@router.get("/dashboard")
async def monitoring_dashboard():
    return dashboard.get_dashboard_summary()


@router.get("/metrics")
async def monitoring_metrics():
    return {"content": dashboard.model_monitor.compute_metrics()}


@router.get("/metrics-json")
async def monitoring_metrics_json():
    return {"metrics": dashboard.model_monitor.compute_metrics()}


@router.get("/metrics-list")
async def monitoring_metrics_list():
    return {"metrics": list(dashboard.model_monitor.compute_metrics().keys())}


@router.get("/dashboard-summary")
async def dashboard_summary():
    return dashboard.get_dashboard_summary()


@router.get("/model-performance")
async def model_performance():
    return dashboard.get_model_performance()


@router.get("/adaptation-stats")
async def adaptation_stats():
    return dashboard.get_adaptation_stats()


@router.get("/inference-health")
async def inference_health():
    return dashboard.get_inference_health()


@router.get("/security-audit")
async def security_audit():
    return dashboard.get_security_audit()


@router.get("/drift-report")
async def drift_report():
    return dashboard.get_drift_report()


@router.get("/alerts")
async def alerts():
    return dashboard.get_alerts()


@router.get("/alerts/active")
async def active_alerts():
    return {"active": dashboard.alert_manager.get_active_alerts()}


@router.get("/alerts/history")
async def alerts_history():
    return {"history": dashboard.alert_manager.get_alert_history()}


@router.get("/perf/latency")
async def perf_latency():
    return dashboard.get_inference_health().get("latency", {})


@router.get("/perf/throughput")
async def perf_throughput():
    return {"throughput": dashboard.get_inference_health().get("throughput", 0)}


@router.get("/hub/health")
async def hub_health():
    try:
        from central_hub import hub_server
        faiss = getattr(hub_server, 'faiss_mgr_global', None)
        moe = getattr(hub_server, 'moe_mgr_global', None)
        
        device_count = 0
        try:
            from central_hub.adapter_registry import _devices
            device_count = len(_devices)
        except:
            pass
        
        expert_count = 0
        if moe:
            try:
                expert_count = moe.get_expert_count()
            except:
                pass
                
        return {
            "total_embeddings": faiss.total if faiss and hasattr(faiss, 'total') else 0,
            "total_devices": device_count,
            "clusters": 1,
            "experts": expert_count,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/hub/stats")
async def hub_stats():
    try:
        from central_hub import hub_server
        from central_hub.fed_avg import get_global_adapter_meta
        faiss = getattr(hub_server, 'faiss_mgr_global', None)
        moe = getattr(hub_server, 'moe_mgr_global', None)
        retrainer = getattr(hub_server, 'retrainer_global', None)
        task_tracker = getattr(hub_server, 'task_tracker_global', None)
        
        device_count = 0
        try:
            from central_hub.adapter_registry import _devices
            device_count = len(_devices)
        except:
            pass
        
        adapter_meta = get_global_adapter_meta()
        
        task_count = 0
        if task_tracker:
            try:
                task_count = len(task_tracker.tasks)
            except:
                pass
        
        return {
            "status": "operational",
            "total_embeddings": faiss.total if faiss and hasattr(faiss, 'total') else 0,
            "total_devices": device_count,
            "global_adapter_version": adapter_meta.get("version", 0),
            "adapter_checksum": adapter_meta.get("checksum", "N/A"),
            "num_experts": moe.get_expert_count() if moe else 0,
            "total_tasks": task_count,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/calibration")
async def calibration():
    return dashboard.get_calibration()


@router.get("/novelty")
async def novelty():
    return dashboard.get_novelty()


@router.get("/confusion-matrix")
async def confusion_matrix():
    return {"confusion_matrix": dashboard.get_confusion_matrix()}


@router.get("/per-class")
async def per_class():
    return {"per_class": dashboard.get_per_class()}


@router.get("/system")
async def system_metrics():
    return dashboard.get_inference_health().get("system", {})


@router.post("/drift/check")
async def drift_check():
    return dashboard.check_drift()


@router.post("/alerts/evaluate")
async def alerts_evaluate():
    fired = dashboard.evaluate_alerts()
    return {"fired": len(fired), "alerts": fired}