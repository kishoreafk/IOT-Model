import base64
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import httpx
import torch
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend


class SecureTransmitter:
    """Secure transmitter with Fernet encryption and RSA signing."""

    def __init__(
        self,
        adapter_weights_path: str,
        hub_url: str = "http://localhost:8000",
        device_id: str = "edge_device_001",
        key_path: str = "keys/encryption.key",
        private_key_path: str = "keys/private_key.pem",
        public_key_path: str = "keys/public_key.pem",
        retry_attempts: int = 3,
        retry_backoff: Optional[list] = None,
    ):
        self.adapter_weights_path = adapter_weights_path
        self.hub_url = hub_url.rstrip("/")
        self.device_id = device_id
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff or [1, 3, 9]

        self.fernet = self._load_fernet_key(key_path)
        self.private_key = self._load_private_key(private_key_path)
        self.public_key = self._load_public_key(public_key_path)

    def _load_fernet_key(self, key_path: str) -> Fernet:
        """Load Fernet encryption key."""
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"Fernet key not found: {key_path}")

        with open(key_path, "rb") as f:
            key = f.read()

        return Fernet(key)

    def _load_private_key(self, key_path: str):
        """Load RSA private key."""
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"Private key not found: {key_path}")

        with open(key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=default_backend(),
            )

        return private_key

    def _load_public_key(self, key_path: str):
        """Load RSA public key."""
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"Public key not found: {key_path}")

        with open(key_path, "rb") as f:
            public_key = serialization.load_pem_public_key(
                f.read(),
                backend=default_backend(),
            )

        return public_key

    @staticmethod
    def generate_rsa_keypair(
        private_key_path: str,
        public_key_path: str,
        key_size: int = 2048,
    ):
        """Generate RSA-2048 key pair."""
        from cryptography.hazmat.primitives.asymmetric import rsa

        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend(),
        )

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        os.makedirs(os.path.dirname(private_key_path), exist_ok=True)

        with open(private_key_path, "wb") as f:
            f.write(private_pem)

        with open(public_key_path, "wb") as f:
            f.write(public_pem)

        print(f"RSA key pair generated: {private_key_path}, {public_key_path}")

    def _sign_payload(self, payload: bytes) -> str:
        """Sign payload with RSA private key."""
        signature = self.private_key.sign(
            payload,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )

        return base64.b64encode(signature).decode("utf-8")

    def _encrypt_payload(self, payload: Dict[str, Any]) -> str:
        """Encrypt payload with Fernet."""
        payload_bytes = json.dumps(payload).encode("utf-8")
        encrypted = self.fernet.encrypt(payload_bytes)

        return base64.b64encode(encrypted).decode("utf-8")

    async def transmit(
        self,
        clip_embedding: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
        sign_payload: bool = True,
    ) -> Dict[str, Any]:
        """
        Transmit encrypted payload to hub.

        Args:
            clip_embedding: CLIP embedding tensor (1, 512)
            metadata: Additional metadata to include
            sign_payload: Whether to sign the payload

        Returns:
            Dictionary with transmission result
        """
        embedding_list = clip_embedding.squeeze().cpu().tolist()

        adapter_weights = None
        if os.path.exists(self.adapter_weights_path):
            try:
                state_dict = torch.load(self.adapter_weights_path, map_location="cpu")
                adapter_weights = {
                    k: v.cpu().tolist() if torch.is_tensor(v) else v
                    for k, v in state_dict.items()
                }
            except Exception:
                pass

        payload = {
            "device_id": self.device_id,
            "timestamp": time.time(),
            "embedding": embedding_list,
            "adapter_weights": adapter_weights,
            "metadata": metadata or {},
        }

        if sign_payload:
            payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
            payload["signature"] = self._sign_payload(payload_bytes)

        encrypted_payload = self._encrypt_payload(payload)

        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(self.retry_attempts):
                try:
                    response = await client.post(
                        f"{self.hub_url}/ingress_update",
                        json={
                            "encrypted_payload": encrypted_payload,
                            "device_id": self.device_id,
                        },
                    )

                    if response.status_code == 200:
                        result = response.json()
                        return {
                            "success": True,
                            "hub_response": result,
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Hub returned {response.status_code}",
                            "hub_response": response.json() if response.content else {},
                        }

                except httpx.ConnectError as e:
                    if attempt < self.retry_attempts - 1:
                        wait_time = self.retry_backoff[attempt] if attempt < len(self.retry_backoff) else self.retry_backoff[-1]
                        print(f"Connection failed, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        return {
                            "success": False,
                            "error": f"Connection failed: {e}",
                        }

                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                    }

        return {
            "success": False,
            "error": "Max retries exceeded",
        }

    async def poll_task(self, task_id: str, max_polls: int = 10, poll_interval: float = 1.5) -> Dict[str, Any]:
        """Poll task completion from hub."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            for _ in range(max_polls):
                try:
                    response = await client.get(f"{self.hub_url}/tasks/{task_id}")

                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status") == "completed":
                            return {"success": True, "result": data.get("result", {})}
                        elif data.get("status") == "failed":
                            return {"success": False, "error": "Task failed"}

                    time.sleep(poll_interval)

                except Exception as e:
                    return {"success": False, "error": str(e)}

        return {"success": False, "error": "Poll timeout"}

    @staticmethod
    def decrypt_and_verify(
        encrypted_payload: str,
        key_path: str = "keys/encryption.key",
        public_key_path: str = "keys/public_key.pem",
    ) -> Dict[str, Any]:
        """
        Decrypt payload and verify RSA signature on the hub side.

        Args:
            encrypted_payload: Base64-encoded Fernet ciphertext
            key_path: Path to Fernet encryption key
            public_key_path: Path to RSA public key

        Returns:
            Decrypted payload dictionary
        """
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"Fernet key not found: {key_path}")

        with open(key_path, "rb") as f:
            fernet = Fernet(f.read())

        encrypted_bytes = base64.b64decode(encrypted_payload)
        decrypted = fernet.decrypt(encrypted_bytes)
        payload = json.loads(decrypted)

        signature = payload.get("signature")
        if signature and os.path.exists(public_key_path):
            try:
                with open(public_key_path, "rb") as f:
                    public_key = serialization.load_pem_public_key(
                        f.read(),
                        backend=default_backend(),
                    )

                payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
                signature_bytes = base64.b64decode(signature)
                public_key.verify(
                    signature_bytes,
                    payload_bytes,
                    padding.PKCS1v15(),
                    hashes.SHA256(),
                )
            except Exception as e:
                print(f"Signature verification failed: {e}")

        return payload


async def main():
    """Demo usage of SecureTransmitter."""
    transmitter = SecureTransmitter(
        adapter_weights_path="edge_node/lora_adapter/adapter_model.bin",
        hub_url="http://localhost:8000",
        device_id="edge_device_001",
        key_path="keys/encryption.key",
        private_key_path="keys/private_key.pem",
        public_key_path="keys/public_key.pem",
    )

    clip_embedding = torch.randn(1, 512)

    result = await transmitter.transmit(
        clip_embedding,
        metadata={"model_version": "1.0", "epoch": 5},
        sign_payload=True,
    )

    print(f"Transmission success: {result['success']}")
    if result.get('hub_response'):
        print(f"Cluster assigned: {result['hub_response'].get('cluster_id')}")
        print(f"Task ID: {result['hub_response'].get('task_id')}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())