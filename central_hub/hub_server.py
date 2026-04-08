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

from central_hub.adapter_registry import get_stale_devices
from central_hub.adapter_registry import router as registry_router
from central_hub.faiss_manager import FaissManager
from central_hub.fed_avg import get_global_adapter_meta, run_fedavg, submit_adapter
from central_hub.hub_retrainer import HubRetrainer
from central_hub.moe_manager import MoEManager
from central_hub.task_tracker import TaskTracker
from monitoring.dashboard import dashboard
from monitoring.dashboard import router as monitoring_router
from peft import LoraConfig, get_peft_model
from transformers import ViTForImageClassification

logger = logging.getLogger(__name__)

# ── Ensure application logs appear in the console regardless of uvicorn config ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
)
logging.getLogger("central_hub").setLevel(logging.INFO)

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

    # Load ViT model for LoRA fine-tuning on hub
    logger.info("[Hub] Loading ViT model for LoRA fine-tuning...")
    try:
        vit_model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=50,
            ignore_mismatched_sizes=True,
        )
        
        weights_path = Path("model/best_vit_model.pth")
        if weights_path.exists():
            try:
                state_dict = torch.load(weights_path, map_location=device)
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("module."):
                        k = k[7:]
                    # Only load backbone weights, skip classifier (has 1000 classes vs 50)
                    if "classifier" not in k and "lm_head" not in k:
                        new_state_dict[k] = v
                
                if new_state_dict:
                    vit_model.load_state_dict(new_state_dict, strict=False)
                    logger.info(f"[Hub] Loaded custom ViT BACKBONE weights from {weights_path}")
                else:
                    logger.warning(f"[Hub] No matching backbone weights in {weights_path}")
            except Exception as e:
                logger.warning(f"[Hub] Could not load custom weights: {e}, using pretrained")
    except Exception as e:
        logger.error(f"[Hub] Failed to load ViT: {e}")
        vit_model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=50,
            ignore_mismatched_sizes=True,
        )

    # Wrap ViT with LoRA (matching edge config: r=8, alpha=16)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=0.1,
        bias="none",
        task_type="IMAGE_CLS",
    )
    vit_model = get_peft_model(vit_model, lora_config)
    vit_model = vit_model.to(device)
    logger.info(f"[Hub] ViT wrapped with LoRA (r=8, alpha=16)")

    retrainer = HubRetrainer(
        model=vit_model,
        embedding_dim=embedding_dim,
        device=device,
        min_samples=1,
    )
    task_tracker = TaskTracker()

    Path("hub_data").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    # Auto-register test device for integration tests
    import time

    from central_hub.adapter_registry import _devices, _registry_lock

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


def _process_ingress(task_id: str, encrypted_payload: str, device_id: str):
    """
    Background task (sync, runs in FastAPI's threadpool): decrypt → route by
    trigger → FedAvg or HubRetrainer.

    Changed from async def to def so that FastAPI runs it in a thread-pool
    executor rather than awaiting it on the event loop.  The function contains
    nothing but blocking I/O (file reads, FAISS locking, threading.Thread) so
    making it a coroutine only serialised all background processing on the loop
    without any concurrency benefit.
    """
    try:
        # Use lightweight decrypt utility (no transformers dependency)
        from central_hub.decrypt_utils import decrypt_payload

        try:
            decrypted = decrypt_payload(
                encrypted_payload,
                key_path=os.getenv("HUB_ENCRYPTION_KEY_PATH", "keys/encryption.key"),
                public_key_path=os.getenv("HUB_PUBLIC_KEY_PATH", "keys/public_key.pem"),
            )
        except FileNotFoundError as e:
            logger.error(
                f"[Ingress] Encryption key missing for device {device_id}: {e}"
            )
            task_tracker.fail(task_id, f"Encryption key not found: {str(e)}")
            return
        except Exception as e:
            logger.error(
                f"[Ingress] Decryption failed for device {device_id}: {e}",
                exc_info=True,
            )
            task_tracker.fail(task_id, f"Decryption failed: {str(e)}")
            return

        # ── Extract routing fields ───────────────────────────────────────
        metadata = decrypted.get("metadata") or {}
        trigger = metadata.get("trigger") or "escalate_hub"
        embedding_list = decrypted.get("embedding")
        adapter_b64 = decrypted.get("adapter_weights")
        num_samples = int(metadata.get("num_samples") or 1)
        clip_pseudo_label = metadata.get("clip_pseudo_label")

        # Always-visible diagnostics (print bypasses log-level filtering)
        print(
            f"[Ingress] device={device_id!r} trigger={trigger!r} "
            f"has_embedding={bool(embedding_list)} "
            f"embedding_len={len(embedding_list) if isinstance(embedding_list, list) else 'N/A'} "
            f"has_adapter={bool(adapter_b64)}",
            flush=True,
        )
        logger.info(
            "[Ingress] device=%s trigger=%r has_embedding=%s has_adapter=%s",
            device_id,
            trigger,
            bool(embedding_list),
            bool(adapter_b64),
        )

        # ── Route by trigger (checked independently of payload completeness) ──
        if trigger == "adapt_local":
            if not adapter_b64:
                error_msg = (
                    f"adapt_local trigger from {device_id!r} but adapter_weights "
                    f"is missing or empty. "
                    f"Payload keys present: {list(decrypted.keys())}"
                )
                logger.error("[Ingress] %s", error_msg)
                print(f"[Ingress] ERROR: {error_msg}", flush=True)
                task_tracker.fail(task_id, error_msg)
                return

            import base64

            try:
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
                logger.warning(
                    "[Ingress] adapt_local from %s → FedAvg v%s, %d devices need sync.",
                    device_id,
                    new_version,
                    len(stale),
                )
                print(
                    f"[Ingress] adapt_local from {device_id!r} → FedAvg v{new_version}, "
                    f"{len(stale)} devices need sync.",
                    flush=True,
                )
            except Exception as e:
                logger.error(
                    "[Ingress] adapt_local processing failed for %s: %s",
                    device_id,
                    e,
                    exc_info=True,
                )
                task_tracker.fail(task_id, f"adapt_local processing failed: {str(e)}")
                return

        elif trigger == "escalate_hub":
            # ── Guard: embedding field must be present and non-empty ────────
            if not embedding_list:
                error_msg = (
                    f"escalate_hub trigger from {device_id!r} but 'embedding' field "
                    f"is missing or empty "
                    f"(type={type(embedding_list).__name__}, "
                    f"payload_keys={list(decrypted.keys())}). "
                    "Check that SecureTransmitter.transmit() includes the embedding."
                )
                logger.error("[Ingress] %s", error_msg)
                print(f"[Ingress] ERROR: {error_msg}", flush=True)
                task_tracker.fail(task_id, error_msg)
                return

            try:
                embedding_array = np.array(embedding_list, dtype=np.float32)

                # Handle nested arrays (e.g., shape (1, 512))
                if embedding_array.ndim == 2:
                    if embedding_array.shape[0] == 1:
                        embedding_array = embedding_array[0]
                    else:
                        raise ValueError(
                            f"Expected 1-D or (1, 512) embedding, "
                            f"got shape {embedding_array.shape}"
                        )

                if embedding_array.ndim != 1 or len(embedding_array) != 512:
                    raise ValueError(
                        f"Expected 512-dim embedding, got shape {embedding_array.shape}"
                    )

                embedding = torch.tensor(
                    embedding_array, dtype=torch.float32
                ).unsqueeze(0)

                # Store in FAISS index
                embedding_np = embedding.numpy()
                cluster_id, total = faiss_mgr.add(
                    embedding_np, device_id, pseudo_label=clip_pseudo_label
                )
                cluster_embeddings = faiss_mgr.get_cluster_embeddings(cluster_id)
                cluster_pseudo_labels = faiss_mgr.get_cluster_pseudo_labels(cluster_id)

                print(
                    f"[Ingress] escalate_hub from {device_id!r}: "
                    f"cluster={cluster_id}, cluster_size={len(cluster_embeddings)}, "
                    f"total_in_index={total} → calling maybe_retrain ...",
                    flush=True,
                )
                logger.info(
                    "[Ingress] escalate_hub from %s: cluster=%s, "
                    "cluster_size=%d, total=%d → calling maybe_retrain",
                    device_id,
                    cluster_id,
                    len(cluster_embeddings),
                    total,
                )

                retraining_scheduled = retrainer.maybe_retrain(
                    cluster_embeddings=cluster_embeddings,
                    cluster_id=cluster_id,
                    num_samples_this_device=1,
                    pseudo_labels=cluster_pseudo_labels,
                )

                # MoE routing (non-critical — errors here must not mask retrain result)
                try:
                    moe_mgr.route(embedding)
                except Exception as moe_err:
                    logger.warning(
                        "[Ingress] MoE routing error (non-fatal): %s", moe_err
                    )

                task_tracker.complete(
                    task_id,
                    {
                        "trigger": "escalate_hub",
                        "cluster_id": cluster_id,
                        "total_embeddings": total,
                        "cluster_size": len(cluster_embeddings),
                        "retraining_scheduled": retraining_scheduled,
                    },
                )

                # Always-visible result line
                print(
                    f"[Ingress] escalate_hub COMPLETE: device={device_id!r} "
                    f"retraining_scheduled={retraining_scheduled} "
                    f"cluster={cluster_id} total_embeddings={total}",
                    flush=True,
                )
                logger.warning(
                    "[Ingress] escalate_hub from %s → cluster=%s, "
                    "cluster_size=%d, total_embeddings=%d, retraining_scheduled=%s",
                    device_id,
                    cluster_id,
                    len(cluster_embeddings),
                    total,
                    retraining_scheduled,
                )

            except ValueError as e:
                logger.error(
                    "[Ingress] Embedding validation failed for %s: %s", device_id, e
                )
                print(
                    f"[Ingress] Embedding validation FAILED for {device_id!r}: {e}",
                    flush=True,
                )
                task_tracker.fail(task_id, f"Embedding validation failed: {str(e)}")
                return
            except Exception as e:
                logger.error(
                    "[Ingress] escalate_hub processing failed for %s: %s",
                    device_id,
                    e,
                    exc_info=True,
                )
                print(
                    f"[Ingress] escalate_hub FAILED for {device_id!r}: {e}",
                    flush=True,
                )
                task_tracker.fail(task_id, f"escalate_hub processing failed: {str(e)}")
                return

        else:
            # Unknown trigger — give a precise error rather than a misleading
            # "missing payload fields" message.
            error_msg = (
                f"Unknown trigger={trigger!r} from device {device_id!r}. "
                f"Valid values: 'adapt_local', 'escalate_hub'. "
                f"Payload top-level keys: {list(decrypted.keys())}"
            )
            logger.error("[Ingress] %s", error_msg)
            print(f"[Ingress] ERROR: {error_msg}", flush=True)
            task_tracker.fail(task_id, error_msg)
            return

    except Exception as e:
        logger.error(
            f"[Ingress] Unexpected error in task {task_id} from {device_id}: {e}",
            exc_info=True,
        )
        task_tracker.fail(task_id, f"Unexpected error: {str(e)}")


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


@app.get("/hub/retraining/status")
async def hub_retraining_status():
    """
    Return whether the hub backbone retraining thread is currently running.
    Poll this endpoint to check if a previous escalate_hub triggered retraining.
    """
    import central_hub.hub_retrainer as _hr

    with _hr._retraining_lock:
        in_progress = _hr._retraining_in_progress
    return {
        "retraining_in_progress": in_progress,
        "total_embeddings": faiss_mgr.total if faiss_mgr else 0,
        "min_samples_threshold": retrainer.min_samples if retrainer else None,
        "timestamp": _now(),
    }


@app.post("/hub/retraining/force-reset")
async def hub_retraining_force_reset():
    """
    Force-reset the _retraining_in_progress flag to False.

    Use only when the retrain daemon thread has died without resetting the flag
    (e.g., after an OOM-kill, SIGKILL, or uvicorn --reload that kept the module
    in memory).  Under normal operation the finally-block in _retrain_background
    resets the flag automatically.
    """
    import central_hub.hub_retrainer as _hr

    with _hr._retraining_lock:
        previous = _hr._retraining_in_progress
        _hr._retraining_in_progress = False
    msg = f"Retraining flag reset to False (was {previous})"
    logger.warning("[Hub] %s", msg)
    print(f"[Hub] /hub/retraining/force-reset — {msg}", flush=True)
    return {"reset": True, "previous_value": previous, "timestamp": _now()}


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
    import threading

    from central_hub.fed_avg import _pending_adapters, get_global_adapter_meta

    _lock = getattr(_pending_adapters, "_lock", threading.Lock())
    with _lock:
        pending_count = len(_pending_adapters)
    return {
        "status": "operational",
        "global_version": get_global_adapter_meta()["version"],
        "pending_adapters": pending_count,
        "global_checksum": get_global_adapter_meta()["checksum"],
    }


@app.post("/moe/create-expert")
async def create_expert(cluster_id: Optional[int] = None):
    if faiss_mgr is None:
        raise HTTPException(status_code=503, detail="FAISS not ready.")

    embeddings = faiss_mgr.get_cluster_embeddings(cluster_id or 0)

    if not faiss_mgr.is_initialized or len(embeddings) == 0:
        return {
            "status": "skipped",
            "reason": "FAISS not initialized or no embeddings",
            "cluster_id": cluster_id,
            "initialized": faiss_mgr.is_initialized if faiss_mgr else False,
        }

    embeddings_np = np.array(
        [e.cpu().numpy() if hasattr(e, "cpu") else e for e in embeddings]
    )

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
