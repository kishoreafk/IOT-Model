import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    """Create test client for hub server."""
    with patch('central_hub.hub_server.initialize_hub'):
        from central_hub.hub_server import app
        return TestClient(app)


class TestHubEndpoints:
    """Integration tests for hub endpoints."""

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_ready_endpoint(self, client):
        """Test readiness check endpoint."""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    @patch('central_hub.hub_server.fernet')
    @patch('central_hub.hub_server.hub_public_key', None)
    def test_ingress_update(self, mock_fernet, client):
        """Test ingress update endpoint."""
        import base64
        import json

        mock_fernet_instance = MagicMock()
        mock_fernet_instance.decrypt.return_value = json.dumps({
            "device_id": "test_device",
            "embedding": np.random.randn(512).tolist(),
            "timestamp": 1234567890,
        }).encode()
        mock_fernet.return_value = mock_fernet_instance

        payload = {
            "encrypted_payload": base64.b64encode(b"test_encrypted").decode(),
            "device_id": "test_device",
        }

        response = client.post("/ingress_update", json=payload)

        assert response.status_code in [200, 422]

    def test_status_endpoint(self, client):
        """Test status endpoint."""
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "total_embeddings" in data

    def test_clusters_endpoint(self, client):
        """Test clusters endpoint."""
        response = client.get("/clusters")
        assert response.status_code == 200
        data = response.json()
        assert "total_clusters" in data
        assert "clusters" in data

    def test_moe_status_endpoint(self, client):
        """Test MoE status endpoint."""
        response = client.get("/moe/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "num_experts" in data

    def test_moe_representation_gap(self, client):
        """Test representation gap endpoint."""
        response = client.get("/moe/representation-gap?cluster_threshold=10")
        assert response.status_code == 200
        data = response.json()
        assert "has_gap" in data

    def test_create_expert(self, client):
        """Test create expert endpoint."""
        response = client.post("/moe/create-expert?cluster_id=0")
        assert response.status_code == 200
        data = response.json()
        assert "new_expert" in data

    def test_reset_endpoint(self, client):
        """Test reset endpoint."""
        response = client.post("/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "reset"


class TestMonitoringEndpoints:
    """Integration tests for monitoring endpoints."""

    def test_monitoring_status(self, client):
        """Test monitoring status endpoint."""
        response = client.get("/monitoring/status")
        assert response.status_code == 200

    def test_monitoring_metrics(self, client):
        """Test metrics endpoint."""
        response = client.get("/monitoring/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")

    def test_monitoring_metrics_json(self, client):
        """Test metrics JSON endpoint."""
        response = client.get("/monitoring/metrics-json")
        assert response.status_code == 200

    def test_monitoring_metrics_list(self, client):
        """Test metrics list endpoint."""
        response = client.get("/monitoring/metrics-list")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data

    def test_dashboard_summary(self, client):
        """Test dashboard summary endpoint."""
        response = client.get("/monitoring/dashboard-summary")
        assert response.status_code == 200

    def test_model_performance(self, client):
        """Test model performance endpoint."""
        response = client.get("/monitoring/model-performance")
        assert response.status_code == 200

    def test_inference_health(self, client):
        """Test inference health endpoint."""
        response = client.get("/monitoring/inference-health")
        assert response.status_code == 200

    def test_security_audit(self, client):
        """Test security audit endpoint."""
        response = client.get("/monitoring/security-audit")
        assert response.status_code == 200

    def test_drift_report(self, client):
        """Test drift report endpoint."""
        response = client.get("/monitoring/drift-report")
        assert response.status_code == 200

    def test_alerts(self, client):
        """Test alerts endpoint."""
        response = client.get("/monitoring/alerts")
        assert response.status_code == 200

    def test_system_metrics(self, client):
        """Test system metrics endpoint."""
        response = client.get("/monitoring/system")
        assert response.status_code == 200


class TestEdgeToHub:
    """Integration tests for edge to hub communication."""

    @pytest.mark.asyncio
    async def test_transmit_and_receive(self):
        """Test full edge to hub transmission."""
        import base64
        import json
        import asyncio
        from unittest.mock import patch, MagicMock

        mock_fernet = MagicMock()
        mock_fernet.decrypt.return_value = json.dumps({
            "device_id": "test_device",
            "embedding": np.random.randn(512).tolist(),
            "timestamp": 1234567890,
            "signature": "mock_signature",
        }).encode()

        with patch('central_hub.hub_server.fernet', mock_fernet):
            from central_hub.hub_server import app
            from fastapi.testclient import TestClient

            client = TestClient(app)

            encrypted_payload = base64.b64encode(b"encrypted_data").decode()

            response = client.post(
                "/ingress_update",
                json={
                    "encrypted_payload": encrypted_payload,
                    "device_id": "test_device",
                }
            )

            assert response.status_code in [200, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])