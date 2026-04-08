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
import threading
import time
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

        # Move backbone to target device
        self.model = self.model.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_retrain(
        self,
        cluster_embeddings: List,
        cluster_id: int,
        num_samples_this_device: int = 1,
        pseudo_labels: Optional[List[Optional[str]]] = None,
    ) -> Optional[int]:
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
        None — retraining is always asynchronous.
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
                    "[HubRetrainer] Retraining already in progress — "
                    "new embeddings buffered."
                )
                return None
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
            logger.info(
                f"[HubRetrainer] Retraining scheduled — cluster={cluster_id}, "
                f"samples={len(cluster_embeddings)}, "
                f"supervised={'yes' if has_labels else 'no (cosine fallback)'}."
            )
            return None
        except Exception as e:
            logger.error(
                f"[HubRetrainer] Failed to start retraining thread: {e}",
                exc_info=True,
            )
            with _retraining_lock:
                _retraining_in_progress = False
            return None

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
        Background thread: fine-tune the hub backbone, then submit to FedAvg.

        Loss strategy
        -------------
        • **Supervised** (pseudo_labels available with ≥ 2 distinct classes):
          Compute a per-class centroid from the raw CLIP embeddings.
          Train the backbone to map each embedding toward its class centroid
          (MSE in embedding space = metric/prototype learning).

        • **Unsupervised** (single class or no labels):
          Train the backbone to preserve cosine similarity with the original
          CLIP embedding (self-supervised feature alignment).
        """
        global _retraining_in_progress
        try:
            logger.info(
                f"[HubRetrainer] ▶ Starting retrain — "
                f"cluster={cluster_id}, samples={len(embeddings)}"
            )
            t0 = time.time()

            # ── 1. Normalise embeddings to tensors ─────────────────────
            processed = []
            for e in embeddings:
                if isinstance(e, np.ndarray):
                    t = torch.from_numpy(e).float()
                else:
                    t = e.float()
                processed.append(t.squeeze().to(self.device))

            X = torch.stack(processed)  # (N, embedding_dim)

            # ── 2. Set up optimiser ────────────────────────────────────
            self.model.train()
            optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

            # ── 3. Choose loss strategy ────────────────────────────────
            valid_labels = [l for l in (pseudo_labels or []) if l]
            unique_labels = list(
                dict.fromkeys(valid_labels)
            )  # preserve order, deduplicate

            if len(unique_labels) >= 1 and len(valid_labels) == len(embeddings):
                # ── Supervised: centroid metric learning ───────────────
                logger.info(
                    f"[HubRetrainer] Using supervised centroid loss "
                    f"(classes={unique_labels})."
                )
                # Compute per-class centroids from raw CLIP embeddings (X)
                centroids: dict = {}
                for label in unique_labels:
                    mask = torch.tensor(
                        [l == label for l in (pseudo_labels or [])],
                        dtype=torch.bool,
                        device=self.device,
                    )
                    centroids[label] = X[mask].mean(dim=0).detach()

                # Build target tensor: each embedding → its class centroid
                targets = torch.stack(
                    [centroids[l] for l in (pseudo_labels or [])]
                )  # (N, embedding_dim)

                dataset = TensorDataset(X, targets)
                loader = DataLoader(dataset, batch_size=min(16, len(X)), shuffle=True)
                criterion = nn.MSELoss()

                for epoch in range(self.num_epochs):
                    epoch_loss = 0.0
                    for xb, yb in loader:
                        optimizer.zero_grad()
                        out = self.model(xb)
                        loss = criterion(out, yb)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    if (epoch + 1) % 5 == 0:
                        logger.debug(
                            f"[HubRetrainer] Supervised epoch "
                            f"{epoch + 1}/{self.num_epochs} "
                            f"loss={epoch_loss / len(loader):.6f}"
                        )

            else:
                # ── Unsupervised: cosine-similarity preservation ────────
                logger.info("[HubRetrainer] Using unsupervised cosine-alignment loss.")
                dataset = TensorDataset(X)
                loader = DataLoader(dataset, batch_size=min(16, len(X)), shuffle=True)

                for epoch in range(self.num_epochs):
                    epoch_loss = 0.0
                    for (xb,) in loader:
                        optimizer.zero_grad()
                        out = self.model(xb)
                        # Maximise cosine similarity between backbone output
                        # and original CLIP embedding
                        cos_sim = nn.functional.cosine_similarity(out, xb)
                        loss = (1.0 - cos_sim).mean()
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    if (epoch + 1) % 5 == 0:
                        logger.debug(
                            f"[HubRetrainer] Cosine epoch "
                            f"{epoch + 1}/{self.num_epochs} "
                            f"loss={epoch_loss / len(loader):.6f}"
                        )

            self.model.eval()

            # ── 4. Serialise the fine-tuned backbone and submit to FedAvg
            buf = io.BytesIO()
            torch.save(self.model.state_dict(), buf)
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
