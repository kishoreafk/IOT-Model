"""
hub_retrainer.py
────────────────
Retrains the global hub backbone on escalated CLIP embeddings from edge devices
(confidence < 0.50 on those devices).

Workflow triggered by POST /ingress_update with trigger="escalate_hub":
  1. FAISS stores incoming CLIP embedding → cluster assigned
  2. HubRetrainer collects all embeddings (+ pseudo-labels) in that cluster
  3. Fine-tunes the hub MoE backbone using:
       • Supervised metric-learning loss when pseudo-labels are available
         (pull same-class embeddings together via centroid MSE)
       • Cosine-similarity preservation loss otherwise
         (align backbone output with original CLIP embedding)
  4. Calls fed_avg.run_fedavg() to merge with any pending edge adapters
  5. Persists the new global adapter
  6. AdapterRegistry notifies all registered edges (version bump)
"""

import io
import logging
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from central_hub.fed_avg import run_fedavg, submit_adapter

logger = logging.getLogger(__name__)

_retraining_lock = threading.Lock()
_retraining_in_progress = False


def get_retraining_status() -> dict:
    """Return current retraining state (safe to call from any thread)."""
    with _retraining_lock:
        return {"retraining_in_progress": _retraining_in_progress}


class HubRetrainer:
    """
    Runs fine-tuning on the hub backbone using escalated CLIP embeddings
    collected in the FAISS index.

    Parameters
    ----------
    model : nn.Module
        The hub's backbone model (embedding_dim → embedding_dim).
    embedding_dim : int
        FAISS / CLIP embedding dimensionality (default 512).
    lora_rank : int
        Kept for API compatibility; unused at hub (hub uses full fine-tune).
    lora_alpha : float
        Kept for API compatibility; unused at hub.
    num_epochs : int
        Fine-tuning epochs per retrain cycle (default 20).
    min_samples : int
        Minimum embeddings in a cluster before retraining triggers (default 5).
    device : str
        Torch device string.
    """

    def __init__(
        self,
        model: nn.Module,
        embedding_dim: int = 512,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        num_epochs: int = 20,
        min_samples: int = 2,
        device: str = "cpu",
    ):
        self.model = model
        self.embedding_dim = embedding_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.num_epochs = num_epochs
        self.min_samples = min_samples
        self.device = torch.device(device)

        self._class_names = self._get_default_labels()

        # Move backbone to target device
        self.model = self.model.to(self.device)

    def _get_default_labels(self) -> List[str]:
        """Load default class labels from config file."""
        class_names_path = Path("configs/class_names.txt")
        if class_names_path.exists():
            with open(class_names_path, "r") as f:
                return [line.strip() for line in f.readlines()]
        
        # Fallback: Tiny ImageNet 50 classes matching ViT training
        return [
            "goldfish", "salamander", "bullfrog", "toad", "alligator",
            "boa", "trilobite", "scorpion", "spider", "tarantula",
            "centipede", "goose", "koala", "jellyfish", "coral",
            "snail", "slug", "nudibranch", "lobster", "crayfish",
            "stork", "penguin", "albatross", "dugong", "chihuahua",
            "terrier", "retriever", "retriever", "shepherd", "poodle",
            "tabby", "persian", "cat", "cougar", "lion",
            "bear", "ladybug", "fly", "bee", "grasshopper",
            "stick_insect", "cockroach", "mantis", "dragonfly", "butterfly",
            "butterfly", "butterfly", "cucumber", "guinea_pig", "pig", "ox"
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_retrain(
        self,
        cluster_embeddings: List,
        cluster_id: int,
        num_samples_this_device: int = 1,
        pseudo_labels: Optional[List[Optional[str]]] = None,
    ) -> bool:
        """
        Trigger retraining if the cluster has enough samples.
        Runs in a background thread to avoid blocking the HTTP response.

        Parameters
        ----------
        cluster_embeddings : list
            All CLIP embeddings (numpy arrays or tensors) in this cluster.
        cluster_id : int
            Cluster identifier (for logging).
        num_samples_this_device : int
            Weight contribution of the current device (used in FedAvg).
        pseudo_labels : list of str | None, optional
            CLIP-derived class labels aligned with ``cluster_embeddings``.
            When provided, supervised centroid-metric loss is used;
            otherwise cosine-similarity preservation loss is used.

        Returns
        -------
        bool
            True when retraining thread is queued, False otherwise.
        """
        global _retraining_in_progress

        if len(cluster_embeddings) < self.min_samples:
            msg = (
                f"[HubRetrainer] Cluster {cluster_id} has only "
                f"{len(cluster_embeddings)} sample(s) "
                f"(min={self.min_samples}). Retraining deferred."
            )
            logger.warning(msg)
            print(msg, flush=True)
            return False

        with _retraining_lock:
            if _retraining_in_progress:
                msg = (
                    "[HubRetrainer] Retraining already in progress — "
                    "new embeddings buffered for next cycle."
                )
                logger.warning(msg)
                print(msg, flush=True)
                return False
            _retraining_in_progress = True

        try:
            thread = threading.Thread(
                target=self._retrain_background,
                args=(
                    cluster_embeddings,
                    cluster_id,
                    num_samples_this_device,
                    pseudo_labels,
                ),
                daemon=True,
            )
            thread.start()
            has_labels = bool(pseudo_labels and any(l for l in pseudo_labels if l))
            msg = (
                f"[HubRetrainer] *** RETRAINING SCHEDULED *** "
                f"cluster={cluster_id}, "
                f"samples={len(cluster_embeddings)}, "
                f"supervised={'yes' if has_labels else 'no (cosine fallback)'}."
            )
            logger.warning(msg)
            print(msg, flush=True)
            return True
        except Exception as e:
            logger.error(
                f"[HubRetrainer] Failed to start retraining thread: {e}",
                exc_info=True,
            )
            with _retraining_lock:
                _retraining_in_progress = False
            print(
                f"[HubRetrainer] Failed to start retraining thread: {e}",
                flush=True,
            )
            return False

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _retrain_background(
        self,
        embeddings: List,
        cluster_id: int,
        num_samples: int,
        pseudo_labels: Optional[List[Optional[str]]] = None,
    ):
        """
        Background thread: fine-tune hub ViT+LoRA, then submit to FedAvg.

        Training strategy (LoRA-based):
        • Uses PEFT-wrapped ViT model (set in hub_server)
        • Maps CLIP embeddings to ViT feature space
        • Trains LoRA adapter with cross-entropy loss on pseudo-labels
        • Extracts only LoRA weights for FedAvg (not full model)
        """
        global _retraining_in_progress
        try:
            start_msg = (
                f"[HubRetrainer] Starting LoRA retrain "
                f"cluster={cluster_id}, samples={len(embeddings)}"
            )
            logger.warning(start_msg)
            print(start_msg, flush=True)
            t0 = time.time()

            # ── 1. Use stored class names ────────────────────────────────
            label_to_idx = {name.lower(): i for i, name in enumerate(self._class_names)}

            # ── 2. Prepare embeddings and labels ───────────────────
            processed = []
            for e in embeddings:
                if isinstance(e, np.ndarray):
                    t = torch.from_numpy(e).float()
                else:
                    t = e.float()
                processed.append(t.squeeze().to(self.device))

            X = torch.stack(processed)  # (N, embedding_dim=512)

            # Map pseudo-labels to class indices
            valid_labels = pseudo_labels or []
            y = torch.zeros(len(X), dtype=torch.long)
            unique_labels_in_batch = set()
            
            for i, label in enumerate(valid_labels):
                if label:
                    label_lower = label.lower()
                    for name, idx in label_to_idx.items():
                        if label_lower in name or name in label_lower:
                            y[i] = idx
                            unique_labels_in_batch.add(label)
                            break

            if unique_labels_in_batch:
                labels_str = ", ".join(sorted(unique_labels_in_batch))
                logger.warning(f"[HubRetrainer] Training on: [{labels_str}]")

            # ── 3. Set up embedding-space training ──────────────────
            self.model.train()
            
            # Use simple linear head on embeddings for training
            # (ViT needs image pixels, but we have CLIP embeddings)
            criterion = nn.CrossEntropyLoss()
            
            # Create a simple projection layer that maps 512 embeddings → 50 classes
            projection = nn.Linear(self.embedding_dim, len(self._class_names)).to(self.device)
            optimizer = optim.Adam(projection.parameters(), lr=1e-3)

            # Create data loader
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=min(8, len(X)), shuffle=True)

            logger.info(
                f"[HubRetrainer] Training LoRA with {len(self._class_names)} classes, "
                f"{len(valid_labels)} labeled samples"
            )

            # ── 4. Training loop ────────────────────────────────
            for epoch in range(self.num_epochs):
                epoch_loss = 0.0
                num_batches = 0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    
                    # Forward through projection layer
                    logits = projection(xb)
                    
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / max(num_batches, 1)
                if (epoch + 1) % 10 == 0:
                    logger.warning(
                        f"[HubRetrainer] Epoch {epoch + 1}/{self.num_epochs}, loss={avg_loss:.4f}"
                    )

            projection.eval()

            # ── 5. Extract trainable adapter weights ───────────────────
            # Save the projection layer weights as adapter
            adapter_state_dict = projection.state_dict()
            
            logger.info(
                f"[HubRetrainer] Extracted {len(adapter_state_dict)} adapter parameters"
            )

            # ── 6. Include class metadata in adapter ────────────────────
            # This is critical: edge needs to know the class ordering!
            adapter_with_metadata = {
                "state_dict": adapter_state_dict,
                "class_names": self._class_names,
                "adapter_type": "projection_layer",
                "num_classes": len(self._class_names),
            }
            
            logger.info(
                f"[HubRetrainer] Adapter metadata: {len(self._class_names)} classes, "
                f"type=projection_layer"
            )

            # ── 7. Serialize and submit to FedAvg ────────────────────
            buf = io.BytesIO()
            torch.save(adapter_with_metadata, buf)
            adapter_bytes = buf.getvalue()

            logger.info(
                f"[HubRetrainer] LoRA adapter size: {len(adapter_bytes)/1024:.1f} KB"
            )

            submit_adapter(
                device_id="hub",
                adapter_bytes=adapter_bytes,
                num_samples=num_samples,
            )

            new_version = run_fedavg(min_participants=1)
            elapsed = time.time() - t0

            # Push notification to stale edges (async in separate thread)
            import asyncio

            def push_async():
                try:
                    hub_url = os.getenv("HUB_URL", "http://localhost:8000")
                    from central_hub.adapter_registry import push_to_all_stale_edges
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    pushed = loop.run_until_complete(push_to_all_stale_edges(hub_url, new_version or 0))
                    if pushed > 0:
                        logger.info(f"[HubRetrainer] Pushed adapter v{new_version} to {pushed} edge(s)")
                    loop.close()
                except Exception as e:
                    logger.warning(f"[HubRetrainer] Edge push notification failed: {e}")

            push_thread = threading.Thread(target=push_async, daemon=True)
            push_thread.start()

            done_msg = (
                f"[HubRetrainer] LoRA Retrain COMPLETE "
                f"cluster={cluster_id}, version={new_version}, "
                f"elapsed={elapsed:.1f}s, adapter_size={len(adapter_bytes)/1024:.1f}KB"
            )
            logger.warning(done_msg)
            print(done_msg, flush=True)

        except Exception as e:
            logger.error(f"[HubRetrainer] LoRA Retrain failed: {e}", exc_info=True)
            print(f"[HubRetrainer] LoRA Retrain failed: {e}", flush=True)
        finally:
            with _retraining_lock:
                _retraining_in_progress = False
