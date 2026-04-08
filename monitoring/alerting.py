import time
import threading
import json
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

from .metrics_collector import collector


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    description: str
    condition: Callable[[], bool]
    severity: AlertSeverity
    cooldown_seconds: int = 300
    message_template: str = ""
    enabled: bool = True


@dataclass
class Alert:
    """Alert instance."""
    alert_id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    fired_at: float
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Alert manager with rule evaluation and notification."""

    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.webhooks: List[str] = []
        self._lock = threading.Lock()
        self._last_evaluation: Dict[str, float] = {}

        self._init_default_rules()

    def _init_default_rules(self):
        """Initialize default alert rules."""
        self.add_rule(AlertRule(
            name="accuracy_drop",
            description="Model accuracy dropped below 50%",
            condition=lambda: collector.get("accuracy", None) is not None and collector.get("accuracy", 1.0) < 0.50,
            severity=AlertSeverity.ERROR,
            cooldown_seconds=300,
            message_template="[ALERT] Model accuracy is below 50%. Current: {value:.2%}",
        ))

        self.add_rule(AlertRule(
            name="high_novelty_false_positives",
            description="False novelty rate exceeds 30%",
            condition=lambda: collector.get("false_novelty_rate", 0) > 0.30,
            severity=AlertSeverity.WARNING,
            cooldown_seconds=300,
            message_template="[ALERT] False novelty rate above 30%. Current: {value:.2%}",
        ))

        self.add_rule(AlertRule(
            name="signature_failure_spike",
            description="Signature failure rate exceeds 10%",
            condition=lambda: collector.get("signature_success_rate", 1.0) < 0.90,
            severity=AlertSeverity.CRITICAL,
            cooldown_seconds=60,
            message_template="[ALERT] Signature failure rate above 10%. Current: {value:.2%}",
        ))

        self.add_rule(AlertRule(
            name="high_inference_latency",
            description="P95 latency exceeds 500ms",
            condition=lambda: collector.get("inference_latency_p95", 0) > 500,
            severity=AlertSeverity.WARNING,
            cooldown_seconds=120,
            message_template="[ALERT] P95 latency above 500ms. Current: {value:.2f}ms",
        ))

        self.add_rule(AlertRule(
            name="gpu_memory_warning",
            description="GPU memory usage exceeds 8GB",
            condition=lambda: collector.get("gpu_memory_used", 0) > 8,
            severity=AlertSeverity.WARNING,
            cooldown_seconds=600,
            message_template="[ALERT] GPU memory above 8GB. Current: {value:.2f}GB",
        ))

        self.add_rule(AlertRule(
            name="drift_detected",
            description="Any drift score exceeds threshold",
            condition=lambda: (
                collector.get("mmd_drift", 0) > 0.1 or
                collector.get("ks_drift", 0) > 0.1 or
                collector.get("chi_sq_drift", 0) > 0.05
            ),
            severity=AlertSeverity.WARNING,
            cooldown_seconds=600,
            message_template="[ALERT] Drift detected in system",
        ))

    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        with self._lock:
            self.rules[rule.name] = rule

    def remove_rule(self, name: str):
        """Remove an alert rule."""
        with self._lock:
            self.rules.pop(name, None)

    def add_webhook(self, url: str):
        """Add webhook notification URL."""
        self.webhooks.append(url)

    def evaluate_rules(self) -> List[Alert]:
        """Evaluate all rules and fire/resolve alerts."""
        fired_alerts = []

        with self._lock:
            for name, rule in self.rules.items():
                if not rule.enabled:
                    continue

                if name in self._last_evaluation:
                    time_since_last = time.time() - self._last_evaluation[name]
                    if time_since_last < rule.cooldown_seconds:
                        continue

                try:
                    condition_met = rule.condition()
                except Exception:
                    condition_met = False

                self._last_evaluation[name] = time.time()

                if condition_met:
                    if name not in self.active_alerts:
                        alert = Alert(
                            alert_id=str(int(time.time() * 1000)),
                            rule_name=name,
                            severity=rule.severity,
                            message=self._format_message(rule.message_template),
                            fired_at=time.time(),
                        )
                        self.active_alerts[name] = alert
                        self.alert_history.append(alert)
                        fired_alerts.append(alert)

                        self._send_webhook(alert)

                elif name in self.active_alerts:
                    alert = self.active_alerts[name]
                    alert.resolved_at = time.time()
                    del self.active_alerts[name]

        collector.set("active_alerts", len(self.active_alerts))

        return fired_alerts

    def _format_message(self, template: str) -> str:
        """Format alert message with current metric values."""
        message = template
        for metric_name in ["accuracy", "false_novelty_rate", "signature_success_rate",
                           "inference_latency_p95", "gpu_memory_used", "mmd_drift", "ks_drift", "chi_sq_drift"]:
            value = collector.get(metric_name, 0)
            message = message.replace("{" + metric_name + "}", str(value))
            message = message.replace("{value}", str(value))
        return message

    def _send_webhook(self, alert: Alert):
        """Send alert to webhook URLs."""
        import httpx

        payload = {
            "alert_id": alert.alert_id,
            "rule_name": alert.rule_name,
            "severity": alert.severity.value,
            "message": alert.message,
            "fired_at": alert.fired_at,
        }

        for webhook_url in self.webhooks:
            try:
                httpx.post(webhook_url, json=payload, timeout=5.0)
            except Exception as e:
                print(f"Failed to send webhook: {e}")

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        with self._lock:
            return [
                {
                    "alert_id": a.alert_id,
                    "rule_name": a.rule_name,
                    "severity": a.severity.value,
                    "message": a.message,
                    "fired_at": a.fired_at,
                }
                for a in self.active_alerts.values()
            ]

    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history."""
        with self._lock:
            return [
                {
                    "alert_id": a.alert_id,
                    "rule_name": a.rule_name,
                    "severity": a.severity.value,
                    "message": a.message,
                    "fired_at": a.fired_at,
                    "resolved_at": a.resolved_at,
                }
                for a in self.alert_history[-limit:]
            ]

    def get_status(self) -> Dict[str, Any]:
        """Get alert manager status."""
        with self._lock:
            return {
                "active_count": len(self.active_alerts),
                "total_rules": len(self.rules),
                "total_fired": len(self.alert_history),
                "webhooks_configured": len(self.webhooks),
            }


alert_manager = AlertManager()