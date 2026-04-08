import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock


class TestEdgeVisionNode:
    """Unit tests for EdgeVisionNode."""

    @patch('edge_node.vision_agent.CLIPModel')
    @patch('edge_node.vision_agent.CLIPProcessor')
    @patch('edge_node.vision_agent.timm.create_model')
    def test_init(self, mock_vit, mock_clip_processor, mock_clip_model):
        """Test node initialization."""
        from edge_node.vision_agent import EdgeVisionNode

        node = EdgeVisionNode(device='cpu', use_fp16=False)

        assert node.device.type == 'cpu'
        assert node.use_fp16 is False

    @patch('edge_node.vision_agent.CLIPModel')
    @patch('edge_node.vision_agent.CLIPProcessor')
    @patch('edge_node.vision_agent.timm.create_model')
    def test_detect_novelty_known(self, mock_vit, mock_clip_processor, mock_clip_model):
        """Test novelty detection with high confidence (Known)."""
        from edge_node.vision_agent import EdgeVisionNode

        mock_clip = MagicMock()
        mock_clip.get_text_features.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])

        node = EdgeVisionNode(device='cpu', use_fp16=False)
        node.clip_model = mock_clip

        mock_processor = MagicMock()
        mock_processor.return_value = {
            'input_ids': torch.tensor([[1]]),
            'pixel_values': torch.tensor([[[1]]]),
        }
        node.clip_processor = mock_processor

        from PIL import Image
        img = Image.new('RGB', (224, 224))

        decision, scores, labels = node.detect_novelty(img, ['car', 'truck', 'person'])

        assert decision in ['Known', 'Adapt_Local', 'Escalate_Hub']

    @patch('edge_node.vision_agent.CLIPModel')
    @patch('edge_node.vision_agent.CLIPProcessor')
    def test_extract_features(self, mock_clip_processor, mock_clip_model):
        """Test feature extraction."""
        from edge_node.vision_agent import EdgeVisionNode

        mock_clip = MagicMock()
        mock_clip.get_image_features.return_value = torch.randn(1, 512)

        node = EdgeVisionNode(device='cpu', use_fp16=False)
        node.clip_model = mock_clip

        from PIL import Image
        img = Image.new('RGB', (224, 224))

        features = node.extract_features(img)

        assert features.shape == (1, 512)

    def test_get_default_labels(self):
        """Test default label loading."""
        from edge_node.vision_agent import EdgeVisionNode
        from pathlib import Path
        import yaml

        config_path = Path("configs/model_config.yaml")
        if config_path.exists():
            node = EdgeVisionNode(device='cpu')
            labels = node._get_default_labels()
            assert isinstance(labels, list)
            assert len(labels) > 0


class TestSecureTransmitter:
    """Unit tests for SecureTransmitter."""

    @patch('builtins.open')
    @patch('edge_node.secure_transmitter.Fernet')
    def test_init(self, mock_fernet, mock_open):
        """Test transmitter initialization."""
        from edge_node.secure_transmitter import SecureTransmitter

        mock_fernet_instance = MagicMock()
        mock_fernet.return_value = mock_fernet_instance

        with patch('os.path.exists', return_value=True):
            transmitter = SecureTransmitter(
                adapter_weights_path='test_path/adapter.bin',
                hub_url='http://localhost:8000',
                device_id='test_device',
                key_path='test_key.key',
                private_key_path='test_private.pem',
                public_key_path='test_public.pem',
            )

            assert transmitter.device_id == 'test_device'
            assert transmitter.hub_url == 'http://localhost:8000'

    @patch('builtins.open')
    @patch('edge_node.secure_transmitter.serialization.load_pem_private_key')
    @patch('edge_node.secure_transmitter.serialization.load_pem_public_key')
    @patch('edge_node.secure_transmitter.Fernet')
    def test_sign_payload(self, mock_fernet, mock_load_public, mock_load_private, mock_open):
        """Test RSA payload signing."""
        from edge_node.secure_transmitter import SecureTransmitter

        mock_key = MagicMock()
        mock_private = MagicMock()
        mock_private.sign.return_value = b'signature_bytes'
        mock_load_private.return_value = mock_private

        mock_load_public.return_value = MagicMock()

        mock_fernet_instance = MagicMock()
        mock_fernet.return_value = mock_fernet_instance

        with patch('os.path.exists', return_value=True):
            transmitter = SecureTransmitter(
                adapter_weights_path='test_path/adapter.bin',
                hub_url='http://localhost:8000',
                device_id='test_device',
                key_path='test_key.key',
                private_key_path='test_private.pem',
                public_key_path='test_public.pem',
            )
            transmitter.private_key = mock_private

            signature = transmitter._sign_payload(b'test_payload')

            assert signature is not None

    @patch('builtins.open')
    @patch('edge_node.secure_transmitter.Fernet')
    def test_encrypt_payload(self, mock_fernet, mock_open):
        """Test Fernet payload encryption."""
        from edge_node.secure_transmitter import SecureTransmitter

        mock_fernet_instance = MagicMock()
        mock_fernet_instance.encrypt.return_value = b'encrypted_bytes'
        mock_fernet.return_value = mock_fernet_instance

        with patch('os.path.exists', return_value=True):
            transmitter = SecureTransmitter(
                adapter_weights_path='test_path/adapter.bin',
                hub_url='http://localhost:8000',
                device_id='test_device',
                key_path='test_key.key',
                private_key_path='test_private.pem',
                public_key_path='test_public.pem',
            )

            payload = {'test': 'data'}
            encrypted = transmitter._encrypt_payload(payload)

            assert encrypted is not None


class TestFaissManager:
    """Unit tests for FaissManager."""

    def test_init(self):
        """Test FAISS manager initialization."""
        from central_hub.faiss_manager import FaissManager

        manager = FaissManager(embedding_dim=512)

        assert manager.embedding_dim == 512
        assert manager.index is not None

    def test_add_embeddings(self):
        """Test adding embeddings to index."""
        from central_hub.faiss_manager import FaissManager

        manager = FaissManager(embedding_dim=512)

        embeddings = np.random.randn(10, 512).astype('float32')
        ids = manager.add_embeddings(embeddings)

        assert len(ids) == 10
        assert manager.get_total_embeddings() == 10

    def test_search(self):
        """Test similarity search."""
        from central_hub.faiss_manager import FaissManager

        manager = FaissManager(embedding_dim=512)

        embeddings = np.random.randn(10, 512).astype('float32')
        manager.add_embeddings(embeddings)

        query = np.random.randn(512).astype('float32')
        distances, ids = manager.search(query, k=5)

        assert distances.shape[0] == 5
        assert ids.shape[0] == 5

    def test_reset(self):
        """Test index reset."""
        from central_hub.faiss_manager import FaissManager

        manager = FaissManager(embedding_dim=512)

        embeddings = np.random.randn(10, 512).astype('float32')
        manager.add_embeddings(embeddings)

        assert manager.get_total_embeddings() == 10

        manager.reset()

        assert manager.get_total_embeddings() == 0


class TestMoEManager:
    """Unit tests for MoEManager."""

    def test_init(self):
        """Test MoE manager initialization."""
        from central_hub.moe_manager import MoEManager

        manager = MoEManager(embedding_dim=512)

        assert manager.embedding_dim == 512
        assert manager.get_expert_count() == 0

    def test_route_embedding(self):
        """Test expert routing."""
        from central_hub.moe_manager import MoEManager

        manager = MoEManager(embedding_dim=512)

        embedding = np.random.randn(512)
        expert_id, weights = manager.route_embedding(embedding)

        assert isinstance(expert_id, int)
        assert len(weights) >= 1

    def test_check_representation_gap(self):
        """Test representation gap detection."""
        from central_hub.moe_manager import MoEManager

        manager = MoEManager(embedding_dim=512, cluster_threshold=10)

        assert manager.check_representation_gap(0, 5) is False
        assert manager.check_representation_gap(0, 15) is True

    def test_get_status(self):
        """Test status retrieval."""
        from central_hub.moe_manager import MoEManager

        manager = MoEManager(embedding_dim=512)

        status = manager.get_status()

        assert status['status'] == 'operational'
        assert status['num_experts'] == 0


class TestDriftDetector:
    """Unit tests for DriftDetector."""

    def test_init(self):
        """Test drift detector initialization."""
        from monitoring.drift_detector import DriftDetector

        detector = DriftDetector(
            mmd_threshold=0.1,
            ks_threshold=0.1,
            chi_sq_threshold=0.05,
        )

        assert detector.mmd_threshold == 0.1

    def test_set_reference(self):
        """Test setting reference distribution."""
        from monitoring.drift_detector import DriftDetector

        detector = DriftDetector()

        embeddings = np.random.randn(100, 512)
        detector.set_reference(embeddings)

        assert detector.reference_embeddings is not None
        assert detector.reference_embeddings.shape == (100, 512)

    def test_compute_mmd(self):
        """Test MMD computation."""
        from monitoring.drift_detector import DriftDetector

        detector = DriftDetector()

        ref_embeddings = np.random.randn(100, 512)
        detector.set_reference(ref_embeddings)

        current_embeddings = np.random.randn(50, 512)
        mmd = detector.compute_mmd(current_embeddings)

        assert isinstance(mmd, float)
        assert mmd >= 0

    def test_compute_ks_drift(self):
        """Test KS drift computation."""
        from monitoring.drift_detector import DriftDetector

        detector = DriftDetector()

        ref_confidences = np.random.rand(100)
        detector.set_reference(None, ref_confidences, None)

        current_confidences = np.random.rand(50)
        ks = detector.compute_ks_drift(current_confidences)

        assert isinstance(ks, float)
        assert 0 <= ks <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])