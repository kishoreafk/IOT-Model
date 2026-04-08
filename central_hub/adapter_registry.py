"""
adapter_registry.py
───────────────────
Keeps a live registry of all edge devices and their last-seen adapter version.
Serves the adapter download endpoints consumed by AdapterSyncClient.

Endpoints this module supports (mounted in hub_server.py):
  POST /devices/register              → register a new edge node
  GET  /devices                       → list all registered devices
  GET  /adapters/latest/version       → {version, checksum}
  GET  /adapters/latest/download      → binary adapter weights
  GET  /devices/{device_id}/status    → per-device sync status
"""

import logging
import threading
import time
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from central_hub.fed_avg import get_global_adapter_meta, get_global_adapter_bytes

logger = logging.getLogger(__name__)

router = APIRouter()

_registry_lock = threading.Lock()
_devices: Dict[str, Dict] = {}


class DeviceRegistration(BaseModel):
    device_id: str
    public_key_path: Optional[str] = None
    adapter_version: int = 0


class DeviceStatus(BaseModel):
    device_id: str
    adapter_version: int
    global_version: int
    is_current: bool
    last_seen: float
    registered_at: float


@router.post("/devices/register")
async def register_device(reg: DeviceRegistration):
    """Register a new edge device or update its record."""
    now = time.time()
    with _registry_lock:
        if reg.device_id in _devices:
            _devices[reg.device_id]["last_seen"] = now
            _devices[reg.device_id]["adapter_version"] = reg.adapter_version
        else:
            _devices[reg.device_id] = {
                "device_id": reg.device_id,
                "public_key_path": reg.public_key_path,
                "last_seen": now,
                "adapter_version": reg.adapter_version,
                "registered_at": now,
            }
    logger.info(f"[AdapterRegistry] Device registered: {reg.device_id}")
    meta = get_global_adapter_meta()
    return {
        "status": "registered",
        "device_id": reg.device_id,
        "global_adapter_version": meta["version"],
    }


@router.get("/devices")
async def list_devices():
    """Return all registered edge devices."""
    with _registry_lock:
        devices = list(_devices.values())
    meta = get_global_adapter_meta()
    return {
        "total_devices": len(devices),
        "global_adapter_version": meta["version"],
        "devices": devices,
    }


@router.get("/devices/{device_id}/status", response_model=DeviceStatus)
async def device_status(device_id: str):
    """Return the adapter sync status for a specific device."""
    with _registry_lock:
        device = _devices.get(device_id)
    if not device:
        raise HTTPException(status_code=404, detail=f"Device '{device_id}' not registered.")
    meta = get_global_adapter_meta()
    return DeviceStatus(
        device_id=device_id,
        adapter_version=device["adapter_version"],
        global_version=meta["version"],
        is_current=device["adapter_version"] >= meta["version"],
        last_seen=device["last_seen"],
        registered_at=device["registered_at"],
    )


@router.get("/adapters/latest/version")
async def get_adapter_version(device_id: Optional[str] = None):
    """
    Return the current global adapter version and SHA-256 checksum.
    Edge nodes poll this to decide whether to download a new adapter.
    """
    if device_id:
        with _registry_lock:
            if device_id in _devices:
                _devices[device_id]["last_seen"] = time.time()
    meta = get_global_adapter_meta()
    if meta["version"] == 0:
        raise HTTPException(
            status_code=404,
            detail="No global adapter has been published yet."
        )
    return meta


@router.get("/adapters/latest/download")
async def download_adapter(device_id: Optional[str] = None):
    """
    Stream the current global LoRA adapter binary to an edge node.
    Edge AdapterSyncClient calls this after seeing a newer version.
    """
    adapter_bytes = get_global_adapter_bytes()
    if adapter_bytes is None:
        raise HTTPException(
            status_code=404,
            detail="No global adapter binary available yet."
        )

    if device_id:
        with _registry_lock:
            if device_id in _devices:
                _devices[device_id]["last_seen"] = time.time()
                meta = get_global_adapter_meta()
                _devices[device_id]["adapter_version"] = meta["version"]

    return Response(
        content=adapter_bytes,
        media_type="application/octet-stream",
        headers={"X-Adapter-Version": str(get_global_adapter_meta()["version"])},
    )


def get_all_device_ids() -> list:
    with _registry_lock:
        return list(_devices.keys())


def get_stale_devices(current_version: int) -> list:
    """Return device IDs that have not yet updated to current_version."""
    with _registry_lock:
        return [
            d["device_id"]
            for d in _devices.values()
            if d["adapter_version"] < current_version
        ]