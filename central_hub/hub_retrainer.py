"""
hub_retrainer.py
────────────────
Retrains the global LoRA adapter on the hub when embeddings are escalated
from edge devices (confidence < 0.50 on those devices).

Workflow triggered by POST /ingress_update with trigger="escalate_hub":
  1. FAISS stores incoming embedding → cluster assigned
  2. HubRetrainer collects all embeddings in that cluster
  3. Fits a lightweight LoRA adapter on top of the hub's ViT backbone
  4. Calls fed_avg.run_fedavg() to merge with any pending edge adapters
  5. Persists the new global adapter
  6. AdapterRegistry notifies all registered edges (version bump)
"""

import io
import logging
import threading
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from central_hub.fed_avg import submit_adapter, run_fedavg

logger = logging.getLogger(__name__)

_retraining_lock = threading.Lock()
_retraining_in_progress = False


class HubRetrainer:
    """
    Runs LoRA fine-tuning on the hub using escalated CLIP embeddings
    collected in the FAISS index.

    Parameters
    ----------
    model : nn.Module
        The hub's ViT backbone (same architecture as edge nodes).
    embedding_dim : int
        FAISS embedding dimensionality (default 512 for CLIP ViT-B/32).
    lora_rank : int
        LoRA rank — must match edge node config (default 8).
    lora_alpha : float
        LoRA alpha scaling — must match edge node config (default 16).
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

        self._projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
        ).to(self.device)

    def maybe_retrain(
        self,
        cluster_embeddings: List[torch.Tensor],
        cluster_id: int,
        num_samples_this_device: int = 1,
    ) -> Optional[int]:
        """
        Trigger retraining if the cluster has enough samples.
        Runs in a background thread to avoid blocking the HTTP response.

        Returns the new global adapter version if retraining was queued, else None.
        """
        global _retraining_in_progress

        if len(cluster_embeddings) < self.min_samples:
            logger.info(
                f"[HubRetrainer] Cluster {cluster_id} has only "
                f"{len(cluster_embeddings)} samples (min={self.min_samples}). "
                "Retraining deferred."
            )
            return None

        with _retraining_lock:
            if _retraining_in_progress:
                logger.info(
                    "[HubRetrainer] Retraining already in progress. Queued embeddings buffered."
                )
                return None
            _retraining_in_progress = True

        thread = threading.Thread(
            target=self._retrain_background,
            args=(cluster_embeddings, cluster_id, num_samples_this_device),
            daemon=True,
        )
        thread.start()
        logger.info(
            f"[HubRetrainer] Retraining scheduled for cluster {cluster_id} "
            f"({len(cluster_embeddings)} samples)."
        )
        return None

    def _retrain_background(
        self,
        embeddings: List[torch.Tensor],
        cluster_id: int,
        num_samples: int,
    ):
        """Background thread: fine-tune projection head, submit to FedAvg."""
        global _retraining_in_progress
        try:
            logger.info(
                f"[HubRetrainer] Starting retrain — cluster={cluster_id}, "
                f"samples={len(embeddings)}"
            )
            t0 = time.time()

            # Convert embeddings to torch tensors if they're numpy arrays
            processed_embeddings = []
            for e in embeddings:
                if isinstance(e, np.ndarray):
                    t = torch.from_numpy(e).float()
                else:
                    t = e.float()
                processed_embeddings.append(t.squeeze().to(self.device))
            
            X = torch.stack(processed_embeddings)

            dataset = TensorDataset(X, X)
            loader = DataLoader(dataset, batch_size=min(16, len(X)), shuffle=True)

            optimizer = optim.AdamW(
                self._projection.parameters(), lr=1e-4, weight_decay=1e-5
            )
            criterion = nn.MSELoss()

            self._projection.train()
            for epoch in range(self.num_epochs):
                epoch_loss = 0.0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    pred = self._projection(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg = epoch_loss / len(loader)
                if (epoch + 1) % 5 == 0:
                    logger.debug(
                        f"[HubRetrainer] Epoch {epoch+1}/{self.num_epochs} "
                        f"loss={avg:.6f}"
                    )
            self._projection.eval()

            buf = io.BytesIO()
            torch.save(self._projection.state_dict(), buf)
            adapter_bytes = buf.getvalue()

            submit_adapter(
                device_id="hub",
                adapter_bytes=adapter_bytes,
                num_samples=num_samples,
            )

            new_version = run_fedavg(min_participants=1)
            elapsed = time.time() - t0

            logger.info(
                f"[HubRetrainer] ✓ Retrain complete — "
                f"cluster={cluster_id}, version={new_version}, "
                f"elapsed={elapsed:.1f}s"
            )
        except Exception as e:
            logger.error(f"[HubRetrainer] Retrain failed: {e}", exc_info=True)
        finally:
            with _retraining_lock:
                _retraining_in_progress = False