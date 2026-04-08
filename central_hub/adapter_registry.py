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

import httpx
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
    edge_url: Optional[str] = None


class DeviceStatus(BaseModel):
    device_id: str
    adapter_version: int
    global_version: int
    is_current: bool
    last_seen: float
    registered_at: float


class HubPushNotification(BaseModel):
    """Payload sent from hub to edge when new adapter is available."""
    new_version: int
    adapter_checksum: str
    hub_url: str


@router.post("/devices/register")
async def register_device(reg: DeviceRegistration):
    """Register a new edge device or update its record."""
    now = time.time()
    with _registry_lock:
        if reg.device_id in _devices:
            _devices[reg.device_id]["last_seen"] = now
            _devices[reg.device_id]["adapter_version"] = reg.adapter_version
            if reg.edge_url:
                _devices[reg.device_id]["edge_url"] = reg.edge_url
        else:
            _devices[reg.device_id] = {
                "device_id": reg.device_id,
                "public_key_path": reg.public_key_path,
                "last_seen": now,
                "adapter_version": reg.adapter_version,
                "registered_at": now,
                "edge_url": reg.edge_url,
            }
    logger.info(f"[AdapterRegistry] Device registered: {reg.device_id}")
    meta = get_global_adapter_meta()
    return {
        "status": "registered",
        "device_id": reg.device_id,
        "global_adapter_version": meta["version"],
    }


async def push_adapter_update(device_id: str, hub_url: str) -> bool:
    """Push notification to edge device to download new adapter."""
    with _registry_lock:
        device = _devices.get(device_id)
    
    if not device:
        logger.debug(f"[AdapterRegistry] Cannot push: device {device_id} not registered")
        return False
    
    edge_url = device.get("edge_url")
    if not edge_url:
        logger.debug(f"[AdapterRegistry] No edge_url for {device_id}, cannot push")
        return False
    
    meta = get_global_adapter_meta()
    new_version = meta.get("version", 0)
    new_checksum = meta.get("checksum", "")
    
    push_payload = {
        "new_version": new_version,
        "adapter_checksum": new_checksum,
        "hub_url": hub_url,
    }
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{edge_url}/hub/notify-adapter",
                json=push_payload,
                params={"device_id": device_id},
            )
            response.raise_for_status()
            logger.info(f"[AdapterRegistry] Push notification sent to {device_id}: v{new_version}")
            return True
    except Exception as e:
        logger.error(f"[AdapterRegistry] Push to {device_id} failed: {e}")
        return False


async def push_to_all_stale_edges(hub_url: str, current_version: int) -> int:
    """Push notifications to all stale edges. Returns count of successful pushes."""
    pushed_count = 0
    
    with _registry_lock:
        stale_devices = [
            d for d in _devices.values()
            if d.get("adapter_version", 0) < current_version
        ]
    
    for device in stale_devices:
        if await push_adapter_update(device["device_id"], hub_url):
            pushed_count += 1
    
    logger.debug(f"[AdapterRegistry] Pushed to {pushed_count}/{len(stale_devices)} stale edges")
    return pushed_count


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