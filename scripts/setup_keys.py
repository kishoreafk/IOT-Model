#!/usr/bin/env python3
"""Bootstrap script: generates RSA-2048 key pair + Fernet key."""
import os
import sys
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend


KEYS_DIR = Path("keys")
KEYS_DIR.mkdir(exist_ok=True)

fernet_path = KEYS_DIR / "encryption.key"
if not fernet_path.exists():
    key = Fernet.generate_key()
    fernet_path.write_bytes(key)
    print(f"[OK] Fernet key written -> {fernet_path}")
else:
    print(f"[SKIP] Fernet key already exists, skipping.")

private_path = KEYS_DIR / "private_key.pem"
public_path = KEYS_DIR / "public_key.pem"

if not private_path.exists():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    print(f"[OK] RSA-2048 key pair written -> {private_path}, {public_path}")
else:
    print(f"[SKIP] RSA keys already exist, skipping.")

print("\n[OK] Key bootstrap complete. Add keys/ to .gitignore immediately.")