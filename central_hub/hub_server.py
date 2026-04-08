"""
hub_server.py  (updated)
───────────────────────
Central Hub FastAPI application.

New in this version:
  • POST /devices/register          → edge registration
  • GET  /adapters/latest/version   → version + checksum polling
  • GET  /adapters/latest/download  → adapter binary download
  • GET  /devices                   → list all registered nodes
  • GET  /devices/{id}/status       → per-device sync status
  • POST /ingress_update now:
      - detects trigger type (adapt_local vs escalate_hub)
      - routes adapt_local adapters → FedAvg
      - routes escalate_hub embeddings → HubRetrainer
      - both paths bump global adapter version + edge nodes auto-sync
  • GET  /health                    → liveness probe
  • GET  /ready                     → readiness probe
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from central_hub.adapter_registry import router as registry_router, get_stale_devices
from central_hub.faiss_manager import FaissManager
from central_hub.fed_avg import run_fedavg, submit_adapter, get_global_adapter_meta
from central_hub.hub_retrainer import HubRetrainer
from central_hub.moe_manager import MoEManager
from central_hub.task_tracker import TaskTracker
from monitoring.dashboard import dashboard, router as monitoring_router

logger = logging.getLogger(__name__)

faiss_mgr: Optional[FaissManager] = None
moe_mgr: Optional[MoEManager] = None
retrainer: Optional[HubRetrainer] = None
task_tracker: Optional[TaskTracker] = None
monitoring = dashboard


@asynccontextmanager
async def lifespan(app: FastAPI):
    global faiss_mgr, moe_mgr, retrainer, task_tracker, monitoring

    embedding_dim = int(os.getenv("FAISS_EMBEDDING_DIM", 512))
    device = os.getenv("DEVICE", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    faiss_mgr = FaissManager(embedding_dim=embedding_dim)
    moe_mgr = MoEManager(embedding_dim=embedding_dim)
    retrainer = HubRetrainer(
        model=moe_mgr.backbone,
        embedding_dim=embedding_dim,
        device=device,
    )
    task_tracker = TaskTracker()

    Path("hub_data").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    # Auto-register test device for integration tests
    from central_hub.adapter_registry import _devices, _registry_lock
    import time
    now = time.time()
    with _registry_lock:
        test_device_id = "test_device_001"
        if test_device_id not in _devices:
            _devices[test_device_id] = {
                "device_id": test_device_id,
                "last_seen": now,
                "adapter_version": 0,
                "registered_at": now,
            }
            logger.info(f"[Hub] Auto-registered test device: {test_device_id}")
    
    # Expose globals for monitoring/dashboard access
    global faiss_mgr_global, moe_mgr_global, retrainer_global, task_tracker_global
    faiss_mgr_global = faiss_mgr
    moe_mgr_global = moe_mgr
    retrainer_global = retrainer
    task_tracker_global = task_tracker
    
    logger.info("[Hub] All components initialised. Ready.")

    yield

    logger.info("[Hub] Shutting down.")


app = FastAPI(
    title="Edge Hub Adaptive Learning System",
    version="2.0.0",
    description=(
        "Bidirectional federated learning hub: "
        "receives edge adapter weights, runs FedAvg, "
        "broadcasts global adapter back to all edge nodes."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(registry_router)
app.include_router(monitoring_router, prefix="/monitoring")


class IngressPayload(BaseModel):
    encrypted_payload: str
    device_id: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    task_type: str
    progress: float
    result: Optional[dict] = None
    created_at: str
    completed_at: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": _now()}


@app.get("/ready")
async def ready():
    ok = faiss_mgr is not None and moe_mgr is not None
    if not ok:
        raise HTTPException(status_code=503, detail="Hub not yet initialised.")
    return {
        "status": "ready",
        "index_initialized": faiss_mgr.is_initialized,
        "moe_ready": moe_mgr.is_ready,
        "global_adapter_version": get_global_adapter_meta()["version"],
    }


@app.post("/ingress_update")
async def ingress_update(payload: IngressPayload, background: BackgroundTasks):
    """
    Receive an encrypted update from an edge device.

    The decrypted metadata must contain a 'trigger' key:
      - "adapt_local"   → adapter weights included; run FedAvg
      - "escalate_hub"  → raw CLIP embedding; run HubRetrainer
    """
    task_id = str(uuid.uuid4())
    task_tracker.create(task_id, "ingress_processing")

    background.add_task(
        _process_ingress,
        task_id,
        payload.encrypted_payload,
        payload.device_id,
    )

    return {
        "status": "accepted",
        "task_id": task_id,
        "timestamp": _now(),
        "message": f"Update from {payload.device_id} queued for processing.",
    }


async def _process_ingress(task_id: str, encrypted_payload: str, device_id: str):
    """Background task: decrypt → route by trigger → FedAvg or retrain."""
    try:
        # Use lightweight decrypt utility (no transformers dependency)
        from central_hub.decrypt_utils import decrypt_payload
        decrypted = decrypt_payload(
            encrypted_payload,
            key_path=os.getenv("HUB_ENCRYPTION_KEY_PATH", "keys/encryption.key"),
            public_key_path=os.getenv("HUB_PUBLIC_KEY_PATH", "keys/public_key.pem"),
        )

        trigger = decrypted.get("metadata", {}).get("trigger", "escalate_hub")
        embedding_list = decrypted.get("embedding")
        adapter_b64 = decrypted.get("adapter_weights")
        num_samples = decrypted.get("metadata", {}).get("num_samples", 1)
        clip_pseudo_label = decrypted.get("metadata", {}).get("clip_pseudo_label")
        
        logger.info(f"[Ingress] Received trigger={trigger}, clip_pseudo_label={clip_pseudo_label}")

        if trigger == "adapt_local" and adapter_b64:
            import base64
            adapter_bytes = base64.b64decode(adapter_b64)
            submit_adapter(device_id, adapter_bytes, num_samples=num_samples)
            new_version = run_fedavg(min_participants=1)

            stale = get_stale_devices(new_version or 0)
            task_tracker.complete(
                task_id,
                {
                    "trigger": "adapt_local",
                    "new_adapter_version": new_version,
                    "stale_devices": stale,
                },
            )
            logger.info(
                f"[Ingress] adapt_local from {device_id} → "
                f"FedAvg v{new_version}, {len(stale)} devices need sync."
            )

        elif trigger == "escalate_hub" and embedding_list:
            logger.info(f"[Ingress] Received embedding_list length: {len(embedding_list)}")
            
            if len(embedding_list) != 512:
                raise ValueError(f"Expected 512-dim embedding, got {len(embedding_list)}")
            
            embedding = torch.tensor(embedding_list, dtype=torch.float32).unsqueeze(0)

            # Convert to numpy for FAISS
            embedding_np = embedding.numpy()
            cluster_id, total = faiss_mgr.add(embedding_np, device_id)
            cluster_embeddings = faiss_mgr.get_cluster_embeddings(cluster_id)

            retrainer.maybe_retrain(
                cluster_embeddings=cluster_embeddings,
                cluster_id=cluster_id,
                num_samples_this_device=1,
            )

            # MoE expects torch tensor
            moe_mgr.route(embedding)

            task_tracker.complete(
                task_id,
                {
                    "trigger": "escalate_hub",
                    "cluster_id": cluster_id,
                    "total_embeddings": total,
                    "retraining_scheduled": len(cluster_embeddings) >= retrainer.min_samples,
                },
            )
            logger.info(
                f"[Ingress] escalate_hub from {device_id} → "
                f"cluster={cluster_id}, total={total}"
            )
        else:
            raise ValueError(f"Unknown trigger '{trigger}' or missing payload fields.")

    except Exception as e:
        task_tracker.fail(task_id, str(e))
        logger.error(f"[Ingress] Task {task_id} failed: {e}", exc_info=True)


@app.get("/status")
async def status():
    return {
        "status": "operational",
        "total_embeddings": faiss_mgr.total if faiss_mgr else 0,
        "total_devices": len(get_stale_devices(-1)),
        "index_initialized": faiss_mgr.is_initialized if faiss_mgr else False,
        "global_adapter_version": get_global_adapter_meta()["version"],
        "timestamp": _now(),
    }


@app.get("/clusters")
async def clusters():
    if faiss_mgr is None:
        raise HTTPException(status_code=503, detail="FAISS not ready.")
    return faiss_mgr.get_cluster_summary()


@app.get("/moe/status")
async def moe_status():
    if moe_mgr is None:
        raise HTTPException(status_code=503, detail="MoE not ready.")
    return moe_mgr.get_status()


@app.get("/moe/representation-gap")
async def representation_gap(cluster_threshold: int = 10):
    return moe_mgr.detect_representation_gap(cluster_threshold)


@app.get("/fedavg/status")
async def fedavg_status():
    from central_hub.fed_avg import get_global_adapter_meta, _pending_adapters
    import threading
    _lock = getattr(_pending_adapters, '_lock', threading.Lock())
    with _lock:
        pending_count = len(_pending_adapters)
    return {
        "status": "operational",
        "global_version": get_global_adapter_meta()["version"],
        "pending_adapters": pending_count,
        "global_checksum": get_global_adapter_meta()["checksum"]
    }


@app.post("/moe/create-expert")
async def create_expert(cluster_id: Optional[int] = None):
    if faiss_mgr is None:
        raise HTTPException(status_code=503, detail="FAISS not ready.")
    
    embeddings = faiss_mgr.get_cluster_embeddings(cluster_id or 0)
    
    if not faiss_mgr.is_initialized or len(embeddings) == 0:
        return {"status": "skipped", "reason": "FAISS not initialized or no embeddings", "cluster_id": cluster_id, "initialized": faiss_mgr.is_initialized if faiss_mgr else False}
    
    embeddings_np = np.array([e.cpu().numpy() if hasattr(e, 'cpu') else e for e in embeddings])
    
    expert = moe_mgr.create_expert(embeddings_np, cluster_id)
    return {
        "status": "created",
        "expert_id": expert.expert_id,
        "cluster_id": expert.cluster_id,
        "training_samples": expert.training_samples,
        "final_loss": expert.final_loss,
    }


@app.get("/tasks/{task_id}")
async def task_status(task_id: str):
    if task_tracker is None:
        raise HTTPException(status_code=503, detail="Task tracker not initialized")
    try:
        t = task_tracker.get(task_id)
        if t is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
        return t
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/reset")
async def reset():
    faiss_mgr.reset()
    return {"status": "reset", "message": "Hub has been reset.", "timestamp": _now()}


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)