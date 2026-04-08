"""
decrypt_utils.py
────────────────
Lightweight decryption utilities for hub server.
Does NOT import transformers or any edge-specific modules.
"""

import base64
import json
import os
from typing import Dict, Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend


def decrypt_payload(
    encrypted_payload: str,
    key_path: str = "keys/encryption.key",
    public_key_path: str = "keys/public_key.pem",
) -> Dict[str, Any]:
    """
    Decrypt payload and verify RSA signature on the hub side.
    This is a standalone function that doesn't require transformers.
    
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
