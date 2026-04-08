import time
import threading
from typing import Dict, Any, List, Optional
from collections import defaultdict
from .metrics_collector import collector


class SecurityMonitor:
    """Monitor security events: signature verification, encryption, access patterns."""

    def __init__(self):
        self.signature_success = 0
        self.signature_failures = 0
        self.encryption_operations = 0
        self.decryption_operations = 0
        self.encryption_failures = 0
        self.decryption_failures = 0
        self.access_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def record_signature(self, success: bool):
        """Record signature verification result."""
        with self._lock:
            if success:
                self.signature_success += 1
            else:
                self.signature_failures += 1

            total = self.signature_success + self.signature_failures
            rate = self.signature_success / total if total > 0 else 0
            collector.set("signature_success_rate", rate)

    def record_encryption(self, success: bool, operation: str = "encrypt"):
        """Record encryption/decryption operation."""
        with self._lock:
            if operation == "encrypt":
                self.encryption_operations += 1
                if not success:
                    self.encryption_failures += 1
            else:
                self.decryption_operations += 1
                if not success:
                    self.decryption_failures += 1

            total_success = (
                self.encryption_operations - self.encryption_failures
                + self.decryption_operations - self.decryption_failures
            )
            total = self.encryption_operations + self.decryption_operations
            rate = total_success / total if total > 0 else 1.0
            collector.set("encryption_success_rate", rate)

    def record_access(self, device_id: str, endpoint: str, status: int):
        """Record API access event."""
        with self._lock:
            self.access_log.append({
                "timestamp": time.time(),
                "device_id": device_id,
                "endpoint": endpoint,
                "status": status,
            })

            if len(self.access_log) > 10000:
                self.access_log = self.access_log[-5000:]

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent security audit log."""
        with self._lock:
            return self.access_log[-limit:]

    def get_summary(self) -> Dict[str, Any]:
        """Get security summary."""
        with self._lock:
            total_signatures = self.signature_success + self.signature_failures
            signature_rate = self.signature_success / total_signatures if total_signatures > 0 else 1.0

            total_enc = self.encryption_operations + self.decryption_operations
            enc_success = (self.encryption_operations - self.encryption_failures + 
                          self.decryption_operations - self.decryption_failures)
            enc_rate = enc_success / total_enc if total_enc > 0 else 1.0

            return {
                "signature_verification": {
                    "total": total_signatures,
                    "success": self.signature_success,
                    "failures": self.signature_failures,
                    "success_rate": signature_rate,
                },
                "encryption_operations": {
                    "total": total_enc,
                    "encryption_ops": self.encryption_operations,
                    "decryption_ops": self.decryption_operations,
                    "failures": self.encryption_failures + self.decryption_failures,
                    "success_rate": enc_rate,
                },
                "recent_access_count": len(self.access_log),
            }

    def reset(self):
        """Reset security metrics."""
        with self._lock:
            self.signature_success = 0
            self.signature_failures = 0
            self.encryption_operations = 0
            self.decryption_operations = 0
            self.encryption_failures = 0
            self.decryption_failures = 0
            self.access_log.clear()


security_monitor = SecurityMonitor()