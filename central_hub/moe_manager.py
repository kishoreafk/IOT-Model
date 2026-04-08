import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import threading
from dataclasses import dataclass
import timm


@dataclass
class Expert:
    """Represents an expert in the MoE system."""
    expert_id: int
    cluster_id: int
    status: str = "active"
    training_samples: int = 0
    final_loss: Optional[float] = None


class GatingNetwork(nn.Module):
    """Soft gating network for expert routing."""

    def __init__(self, embedding_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.gate(x)


class ExpertModel(nn.Module):
    """Expert model for classification."""

    def __init__(self, input_dim: int, num_classes: int = 50):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class Backbone(nn.Module):
    """Hub backbone model for shared use with HubRetrainer."""

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x):
        return self.model(x)


class MoEManager:
    """Mixture-of-Experts manager for expert routing and training."""

    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 50,
        cluster_threshold: int = 10,
        model_save_path: str = "models/experts",
    ):
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.cluster_threshold = cluster_threshold
        self.model_save_path = model_save_path

        self.experts: Dict[int, Expert] = {}
        self.gating_network: Optional[GatingNetwork] = None
        self.expert_models: Dict[int, ExpertModel] = {}
        self.next_expert_id = 0

        self._lock = threading.Lock()
        self._init_gating_network()
        self.backbone = Backbone(embedding_dim)
        self.is_ready = True

    def _init_gating_network(self):
        """Initialize gating network."""
        self.gating_network = GatingNetwork(self.embedding_dim, 1)
        self.gating_network.eval()

    def get_expert_count(self) -> int:
        """Get number of active experts."""
        with self._lock:
            return len([e for e in self.experts.values() if e.status == "active"])

    def get_experts_info(self) -> List[Dict[str, Any]]:
        """Get information about all experts."""
        with self._lock:
            return [
                {
                    "expert_id": e.expert_id,
                    "cluster_id": e.cluster_id,
                    "status": e.status,
                    "training_samples": e.training_samples,
                }
                for e in self.experts.values()
            ]

    def route(self, embedding: torch.Tensor) -> Tuple[int, np.ndarray]:
        """Route embedding to appropriate expert."""
        with self._lock:
            if not self.experts:
                return 0, np.array([1.0])

            with torch.no_grad():
                weights = self.gating_network(embedding).numpy()[0]

            expert_id = int(np.argmax(weights))
            return expert_id, weights

    def route_embedding(self, embedding: np.ndarray) -> Tuple[int, np.ndarray]:
        """Route embedding to appropriate expert."""
        with self._lock:
            if not self.experts:
                return 0, np.array([1.0])

            embedding_tensor = torch.FloatTensor(embedding.reshape(1, -1))

            with torch.no_grad():
                weights = self.gating_network(embedding_tensor).numpy()[0]

            expert_id = int(np.argmax(weights))
            return expert_id, weights

    def check_representation_gap(self, cluster_id: int, cluster_size: int) -> bool:
        """Check if a cluster needs a new expert."""
        return cluster_size >= self.cluster_threshold

    def detect_representation_gap(self, cluster_threshold: int = 10) -> Dict[str, Any]:
        """Detect if there's a representation gap requiring new expert."""
        with self._lock:
            total_experts = len(self.experts)
            return {
                "has_gap": total_experts == 0 or total_experts < cluster_threshold,
                "current_experts": total_experts,
                "threshold": cluster_threshold,
            }

    def create_expert(
        self,
        training_embeddings: np.ndarray,
        cluster_id: Optional[int] = None,
    ) -> Expert:
        """Create and train a new expert for a cluster."""
        with self._lock:
            expert_id = self.next_expert_id
            self.next_expert_id += 1

            cluster = cluster_id if cluster_id is not None else expert_id

            expert = Expert(
                expert_id=expert_id,
                cluster_id=cluster,
                status="training",
                training_samples=len(training_embeddings),
            )

            expert_model = ExpertModel(self.embedding_dim, self.num_classes)
            optimizer = torch.optim.Adam(expert_model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            training_labels = np.random.randint(0, self.num_classes, size=len(training_embeddings))

            embeddings_tensor = torch.FloatTensor(training_embeddings)
            labels_tensor = torch.LongTensor(training_labels)

            dataset = torch.utils.data.TensorDataset(embeddings_tensor, labels_tensor)
            loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

            expert_model.train()
            for epoch in range(50):
                total_loss = 0
                for batch_emb, batch_labels in loader:
                    optimizer.zero_grad()
                    outputs = expert_model(batch_emb)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            expert_model.eval()
            final_loss = total_loss / len(loader)

            expert.status = "active"
            expert.final_loss = final_loss

            self.experts[expert_id] = expert
            self.expert_models[expert_id] = expert_model

            os.makedirs(self.model_save_path, exist_ok=True)
            torch.save(expert_model.state_dict(), f"{self.model_save_path}/expert_{expert_id}.pt")

            return expert

    def predict(self, embedding: np.ndarray) -> Tuple[int, float]:
        """Make prediction using MoE."""
        with self._lock:
            if not self.expert_models:
                return 0, 0.0

            expert_id, _ = self.route_embedding(embedding)

            if expert_id in self.expert_models:
                model = self.expert_models[expert_id]
                embedding_tensor = torch.FloatTensor(embedding.reshape(1, -1))

                with torch.no_grad():
                    logits = model(embedding_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    conf, pred = probs.max(dim=-1)

                return int(pred.item()), float(conf.item())

            return 0, 0.0

    def get_status(self) -> Dict[str, Any]:
        """Get MoE system status."""
        with self._lock:
            return {
                "status": "operational",
                "num_experts": len(self.experts),
                "embedding_dim": self.embedding_dim,
                "experts": [
                    {
                        "expert_id": e.expert_id,
                        "cluster_id": e.cluster_id,
                        "status": e.status,
                    }
                    for e in self.experts.values()
                ],
            }

    def get_cluster_embeddings(self, cluster_id: Optional[int]) -> List[torch.Tensor]:
        """Get embeddings for a cluster."""
        return []

    def reset(self):
        """Reset MoE system."""
        with self._lock:
            self.experts = {}
            self.expert_models = {}
            self.next_expert_id = 0
            self._init_gating_network()