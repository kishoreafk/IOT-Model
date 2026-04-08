"""
fed_avg.py
──────────
Federated Averaging (FedAvg) aggregator for the Central Hub.

When an edge node submits adapter weights, the hub:
  1. Stores the incoming adapter in a per-device buffer
  2. When enough adapters have arrived (or a timeout fires), runs FedAvg
  3. Produces a single global_adapter state_dict
  4. Bumps the global adapter version + computes SHA-256 checksum
  5. Saves to disk; AdapterRegistry picks it up for distribution
"""

import hashlib
import io
import logging
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_pending_adapters: Dict[str, Dict] = {}
_global_version: int = 0
_global_checksum: str = ""
_global_adapter_path = Path("hub_data/global_adapter.bin")
_global_adapter_type: str = "lora_vit"


def submit_adapter(
    device_id: str,
    adapter_bytes: bytes,
    num_samples: int = 1,
):
    """
    Called when a POST /ingress_update arrives with adapter weights.
    """
    try:
        state_dict = torch.load(
            io.BytesIO(adapter_bytes), map_location="cpu"
        )
    except Exception as e:
        logger.error(f"[FedAvg] Cannot load adapter from {device_id}: {e}")
        return

    with _lock:
        _pending_adapters[device_id] = {
            "state_dict": state_dict,
            "weight": max(num_samples, 1),
            "timestamp": time.time(),
        }
        logger.info(
            f"[FedAvg] Adapter buffered from {device_id}. "
            f"Total pending: {len(_pending_adapters)}"
        )


def run_fedavg(min_participants: int = 1) -> Optional[int]:
    """
    Run FedAvg across all pending adapters if ≥ min_participants have submitted.
    Returns the new global adapter version on success, None if skipped.
    """
    global _global_version, _global_checksum

    with _lock:
        if len(_pending_adapters) < min_participants:
            logger.info(
                f"[FedAvg] Waiting for more adapters "
                f"({len(_pending_adapters)}/{min_participants})."
            )
            return None

        participants = dict(_pending_adapters)
        _pending_adapters.clear()

    logger.info(f"[FedAvg] Running FedAvg over {len(participants)} adapters…")

    total_weight = sum(p["weight"] for p in participants.values())
    averaged: Dict[str, torch.Tensor] = {}

    reference_keys = list(participants[next(iter(participants))]["state_dict"].keys())

    for key in reference_keys:
        weighted_sum = None
        for device_id, p in participants.items():
            sd = p["state_dict"]
            if key not in sd:
                logger.warning(f"[FedAvg] Key '{key}' missing from {device_id}, skipping.")
                continue
            tensor = sd[key].float()
            contribution = tensor * (p["weight"] / total_weight)
            weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution

        if weighted_sum is not None:
            averaged[key] = weighted_sum

    buf = io.BytesIO()
    torch.save(averaged, buf)
    adapter_bytes = buf.getvalue()
    checksum = hashlib.sha256(adapter_bytes).hexdigest()

    _global_adapter_path.parent.mkdir(parents=True, exist_ok=True)
    _global_adapter_path.write_bytes(adapter_bytes)

    with _lock:
        _global_version += 1
        _global_checksum = checksum
        new_version = _global_version

    logger.info(
        f"[FedAvg] ✓ Global adapter updated → v{new_version} "
        f"({len(averaged)} tensors, {len(adapter_bytes)/1024:.1f} KB, "
        f"participants={list(participants.keys())})"
    )
    return new_version


def get_global_adapter_meta() -> Dict:
    """Return current version + checksum + type for /adapters/latest/version."""
    with _lock:
        return {
            "version": _global_version,
            "checksum": _global_checksum,
            "adapter_type": _global_adapter_type,
        }


def get_global_adapter_bytes() -> Optional[bytes]:
    """Return raw adapter bytes for /adapters/latest/download."""
    if not _global_adapter_path.exists():
        return None
    return _global_adapter_path.read_bytes()