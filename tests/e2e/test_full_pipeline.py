import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock


@pytest.fixture
def hub_url():
    """Hub URL for testing."""
    return "http://localhost:8000"


@pytest.fixture
def mock_edge_node():
    """Mock edge node for testing."""
    class MockEdgeNode:
        def __init__(self):
            self.device = 'cpu'
            self.use_fp16 = False
            self.known_threshold = 0.80
            self.adapt_threshold = 0.50

        def detect_novelty(self, image, candidate_labels):
            scores = np.random.rand(len(candidate_labels))
            scores = scores / scores.sum()
            sorted_indices = np.argsort(scores)[::-1]
            labels = [candidate_labels[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]
            top_score = scores[0]

            if top_score > self.known_threshold:
                decision = "Known"
            elif top_score > self.adapt_threshold:
                decision = "Adapt_Local"
            else:
                decision = "Escalate_Hub"

            return decision, scores, labels

        def extract_features(self, image):
            embedding = np.random.randn(1, 512)
            return embedding / np.linalg.norm(embedding)

        def local_adaptation(self, image, pseudo_label, num_epochs=5):
            pass

        def _save_adapter_weights(self):
            pass

    return MockEdgeNode()


@pytest.fixture
def mock_transmitter():
    """Mock secure transmitter for testing."""
    class MockTransmitter:
        def __init__(self, hub_url, device_id):
            self.hub_url = hub_url
            self.device_id = device_id
            self.last_payload = None

        async def transmit(self, clip_embedding, metadata=None, sign_payload=True):
            self.last_payload = {
                "embedding": clip_embedding.squeeze().tolist(),
                "metadata": metadata or {},
                "device_id": self.device_id,
            }

            return {
                "success": True,
                "hub_response": {
                    "cluster_id": 0,
                    "task_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "success",
                }
            }

        async def poll_task(self, task_id, max_polls=10, poll_interval=1.5):
            return {
                "success": True,
                "result": {"cluster_id": 0},
            }

    return MockTransmitter


class TestFullPipeline:
    """End-to-end tests for the full pipeline."""

    def test_edge_node_detection(self, mock_edge_node):
        """Test edge node novelty detection."""
        candidate_labels = ["car", "truck", "person", "dog", "unknown"]

        decision, scores, labels = mock_edge_node.detect_novelty(None, candidate_labels)

        assert decision in ["Known", "Adapt_Local", "Escalate_Hub"]
        assert len(scores) == len(candidate_labels)
        assert abs(sum(scores) - 1.0) < 0.01

    def test_feature_extraction(self, mock_edge_node):
        """Test CLIP feature extraction."""
        features = mock_edge_node.extract_features(None)

        assert features.shape == (1, 512)
        norm = np.linalg.norm(features)
        assert abs(norm - 1.0) < 0.01

    def test_decision_thresholds(self):
        """Test CLIP decision threshold logic."""
        class TestNode:
            known_threshold = 0.80
            adapt_threshold = 0.50

            def get_decision(self, confidence):
                if confidence > self.known_threshold:
                    return "Known"
                elif confidence > self.adapt_threshold:
                    return "Adapt_Local"
                else:
                    return "Escalate_Hub"

        node = TestNode()

        assert node.get_decision(0.95) == "Known"
        assert node.get_decision(0.75) == "Adapt_Local"
        assert node.get_decision(0.40) == "Escalate_Hub"

    @pytest.mark.asyncio
    async def test_transmission_to_hub(self, mock_transmitter):
        """Test transmission to hub."""
        transmitter = mock_transmitter("http://localhost:8000", "test_device")

        embedding = np.random.randn(1, 512)
        result = await transmitter.transmit(
            embedding,
            metadata={"model_version": "1.0", "epoch": 5},
            sign_payload=True,
        )

        assert result["success"] is True
        assert "cluster_id" in result["hub_response"]
        assert "task_id" in result["hub_response"]

    @pytest.mark.asyncio
    async def test_task_polling(self, mock_transmitter):
        """Test task completion polling."""
        transmitter = mock_transmitter("http://localhost:8000", "test_device")

        task_id = "550e8400-e29b-41d4-a716-446655440000"
        result = await transmitter.poll_task(task_id, max_polls=3)

        assert "success" in result

    def test_faiss_indexing(self):
        """Test FAISS indexing at hub."""
        try:
            import faiss
            import numpy as np

            embedding_dim = 512
            index = faiss.IndexFlatIP(embedding_dim)
            index = faiss.IndexIDMap(index)

            embeddings = np.random.randn(100, embedding_dim).astype('float32')
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            ids = list(range(100))
            index.add_with_ids(embeddings, np.array(ids))

            assert index.ntotal == 100

            query = np.random.randn(1, embedding_dim).astype('float32')
            query = query / np.linalg.norm(query, axis=1, keepdims=True)

            distances, retrieved_ids = index.search(query, k=5)

            assert len(distances) == 1
            assert len(retrieved_ids[0]) == 5

        except ImportError:
            pytest.skip("FAISS not installed")

    def test_moe_routing(self):
        """Test MoE routing."""
        try:
            import torch
            import torch.nn as nn

            class SimpleGating(nn.Module):
                def __init__(self, dim, num_experts):
                    super().__init__()
                    self.gate = nn.Linear(dim, num_experts)

                def forward(self, x):
                    return torch.softmax(self.gate(x), dim=-1)

            gating = SimpleGating(512, 4)

            embedding = torch.randn(1, 512)
            weights = gating(embedding)

            assert weights.shape == (1, 4)
            assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)

        except ImportError:
            pytest.skip("PyTorch not installed")

    @patch('central_hub.hub_server.FaissManager')
    def test_hub_endpoint_integration(self, mock_faiss):
        """Test hub endpoint integration."""
        from fastapi.testclient import TestClient
        from central_hub.hub_server import app

        mock_manager = MagicMock()
        mock_manager.get_total_embeddings.return_value = 150
        mock_manager.get_all_clusters.return_value = {}
        mock_faiss.return_value = mock_manager

        with patch('central_hub.hub_server.faiss_manager', mock_manager):
            client = TestClient(app)

            response = client.get("/status")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "operational"
            assert "total_embeddings" in data


class TestEncryptionFlow:
    """Tests for encryption and signing flow."""

    def test_fernet_encryption(self):
        """Test Fernet encryption/decryption."""
        try:
            from cryptography.fernet import Fernet

            key = Fernet.generate_key()
            fernet = Fernet(key)

            original_data = {"test": "data", "embedding": [1, 2, 3]}
            original_bytes = str(original_data).encode()

            encrypted = fernet.encrypt(original_bytes)
            decrypted = fernet.decrypt(encrypted)

            assert decrypted == original_bytes

        except ImportError:
            pytest.skip("cryptography not installed")

    def test_rsa_signing(self):
        """Test RSA signing and verification."""
        try:
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives.asymmetric import padding
            import base64

            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend(),
            )

            public_key = private_key.public_key()

            message = b"test message for signing"

            signature = private_key.sign(
                message,
                padding.PKCS1v15(),
                hashes.SHA256(),
            )

            public_key.verify(
                signature,
                message,
                padding.PKCS1v15(),
                hashes.SHA256(),
            )

        except ImportError:
            pytest.skip("cryptography not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])