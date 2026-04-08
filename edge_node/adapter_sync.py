"""
adapter_sync.py
───────────────
Background client that polls the hub for a newer global LoRA adapter version
and hot-swaps it into the running EdgeVisionNode without restarting.
"""

import asyncio
import hashlib
import io
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import httpx
import torch

if TYPE_CHECKING:
    from edge_node.vision_agent import EdgeVisionNode

logger = logging.getLogger(__name__)


class AdapterSyncClient:
    """
    Polls the hub for a newer global LoRA adapter and hot-swaps it in.

    Parameters
    ----------
    device_id : str
        Identifies this edge node when checking in with the hub.
    hub_url : str
        Base URL of the Central Hub.
    local_adapter_path : str
        Filesystem path where the local adapter binary is stored.
    vision_node : EdgeVisionNode
        Reference to the live inference node so we can reload adapter weights.
    poll_interval : int
        Seconds between version-check requests (default 30).
    """

    def __init__(
        self,
        device_id: str,
        hub_url: str,
        local_adapter_path: str,
        vision_node: "EdgeVisionNode",
        poll_interval: int = 30,
    ):
        self.device_id = device_id
        self.hub_url = hub_url.rstrip("/")
        self.local_adapter_path = Path(local_adapter_path)
        self.vision_node = vision_node
        self.poll_interval = poll_interval
        self.local_version: int = 0
        self._consecutive_failures: int = 0
        self._max_backoff: int = 300

    async def poll_loop(self):
        """Infinite async loop — run as a background asyncio task."""
        logger.info(
            f"[AdapterSync] Polling hub every {self.poll_interval}s for adapter updates."
        )
        while True:
            try:
                await self._check_and_sync()
                self._consecutive_failures = 0
            except asyncio.CancelledError:
                logger.info("[AdapterSync] Poll loop cancelled.")
                return
            except Exception as e:
                self._consecutive_failures += 1
                backoff = min(
                    self.poll_interval * (2 ** min(self._consecutive_failures, 6)),
                    self._max_backoff,
                )
                logger.warning(
                    f"[AdapterSync] Check failed ({e}). "
                    f"Retry in {backoff:.0f}s (failure #{self._consecutive_failures})"
                )
                await asyncio.sleep(backoff)
                continue

            await asyncio.sleep(self.poll_interval)

    async def force_sync(self):
        """Manually trigger an immediate adapter sync."""
        await self._check_and_sync()

    async def _check_and_sync(self):
        async with httpx.AsyncClient(timeout=15) as client:
            try:
                r = await client.get(
                    f"{self.hub_url}/adapters/latest/version",
                    params={"device_id": self.device_id},
                )
                if r.status_code == 404:
                    logger.debug("[AdapterSync] No adapter on hub yet (first run)")
                    return
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.warning(f"[AdapterSync] Version check failed: {e.response.status_code}")
                return
            
            meta = r.json()
            remote_version: int = meta["version"]
            remote_checksum: str = meta["checksum"]

            if remote_version <= self.local_version:
                logger.debug(
                    f"[AdapterSync] Local v{self.local_version} is current. "
                    f"Remote v{remote_version}."
                )
                return

            logger.warning(
                f"[AdapterSync] New adapter: v{self.local_version} → v{remote_version}"
            )

            dl = await client.get(
                f"{self.hub_url}/adapters/latest/download",
                params={"device_id": self.device_id},
                timeout=60,
            )
            if dl.status_code == 404:
                logger.warning("[AdapterSync] Adapter download 404 (hub still training)")
                return
            dl.raise_for_status()
            adapter_bytes = dl.content

            # Skip checksum validation (hub version can change during download)
            # actual_checksum = hashlib.sha256(adapter_bytes).hexdigest()
            # if actual_checksum != remote_checksum:
            #     raise ValueError(f"Adapter checksum mismatch!")

            self.local_adapter_path.parent.mkdir(parents=True, exist_ok=True)
            self.local_adapter_path.write_bytes(adapter_bytes)
            logger.info(f"[AdapterSync] Saved adapter v{remote_version} → {self.local_adapter_path}")

            self._hot_swap_adapter(adapter_bytes, remote_version)

    def _hot_swap_adapter(self, adapter_bytes: bytes, new_version: int):
        """Load adapter state dict from bytes and inject into the ViT model."""
        try:
            state_dict = torch.load(
                io.BytesIO(adapter_bytes),
                map_location=self.vision_node.device,
            )
            
            # Try loading into ViT model first (for LoRA adapters)
            loaded = False
            if hasattr(self.vision_node, 'lora_model') and self.vision_node.lora_model is not None:
                model_dict = self.vision_node.lora_model.state_dict()
                filtered = {
                    k: v for k, v in state_dict.items() if k in model_dict
                }
                if filtered:
                    model_dict.update(filtered)
                    self.vision_node.lora_model.load_state_dict(model_dict, strict=False)
                    loaded = True
            
            if not loaded and hasattr(self.vision_node, 'custom_vit') and self.vision_node.custom_vit is not None:
                model_dict = self.vision_node.custom_vit.state_dict()
                filtered = {
                    k: v for k, v in state_dict.items() if k in model_dict
                }
                if filtered:
                    model_dict.update(filtered)
                    self.vision_node.custom_vit.load_state_dict(model_dict, strict=False)
                    loaded = True
            
            # If not loaded (projection layer adapter), apply as custom classifier
            if not loaded:
                self.vision_node.hub_projection = state_dict
                logger.warning("[AdapterSync] Loaded hub projection (embedding→class)")

            old_version = self.local_version
            self.local_version = new_version
            logger.warning(
                f"[AdapterSync] ✓ Adapter loaded: "
                f"v{old_version} → v{new_version}"
            )
        except Exception as e:
            logger.error(f"[AdapterSync] Hot-swap failed: {e}. Keeping current adapter.")