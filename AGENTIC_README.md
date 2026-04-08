# Edge Hub Adaptive Learning System
### Agentic Implementation Guide — Full Workflow Reference

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.100+-green.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)
![Monitoring](https://img.shields.io/badge/monitoring-Prometheus%2BGrafana-purple.svg)
![Docker](https://img.shields.io/badge/docker-compose-blue.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black.svg)

---

## ⚡ Improvements Over Original README

The following issues were identified and corrected in this document:

| # | Issue | Fix Applied |
|---|-------|-------------|
| 1 | Typo: "Novty Detection" | Corrected to "Novelty Detection" throughout |
| 2 | Missing repository URL placeholder `<repository-url>` | Replaced with env-var pattern and instructions |
| 3 | `await transmitter.transmit()` used in sync context with no async note | Async runner pattern documented |
| 4 | No project directory tree | Full annotated tree added |
| 5 | No `requirements.txt` content | Pinned dependency list added |
| 6 | No `.env.example` reference | Full env-var manifest added |
| 7 | Docker only covers monitoring stack | Full `docker-compose.yml` for all services added |
| 8 | No API authentication documented | JWT + API key auth flow added |
| 9 | No CI/CD pipeline | GitHub Actions workflow added |
| 10 | No multi-node deployment guidance | Horizontal scale patterns documented |
| 11 | No graceful shutdown procedure | SIGTERM handling and drain logic documented |
| 12 | No health check / readiness probe | `/health` and `/ready` endpoints documented |
| 13 | No versioning / changelog strategy | Semantic versioning + CHANGELOG pattern added |
| 14 | No structured testing strategy | Unit, integration, and E2E test plans added |
| 15 | Key generation steps not automated | `scripts/setup_keys.py` bootstrap script added |

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Project Directory Structure](#3-project-directory-structure)
4. [Key Features](#4-key-features)
5. [Prerequisites](#5-prerequisites)
6. [Environment Variables](#6-environment-variables)
7. [Installation — Step by Step](#7-installation--step-by-step)
8. [Configuration Files](#8-configuration-files)
9. [Security Setup](#9-security-setup)
10. [Running the System](#10-running-the-system)
11. [Docker Deployment](#11-docker-deployment)
12. [API Reference](#12-api-reference)
13. [Monitoring & Observability](#13-monitoring--observability)
14. [Testing Strategy](#14-testing-strategy)
15. [Multi-Node Deployment](#15-multi-node-deployment)
16. [CI/CD Pipeline](#16-cicd-pipeline)
17. [Performance Optimization](#17-performance-optimization)
18. [Troubleshooting](#18-troubleshooting)
19. [Contributing](#19-contributing)
20. [Versioning & Changelog](#20-versioning--changelog)
21. [License & Acknowledgments](#21-license--acknowledgments)

---

## 1. Overview

The **Edge Hub Adaptive Learning System** is a distributed machine learning platform for IoT edge computing. It combines zero-shot novelty detection, local model adaptation, and centralized clustering with Mixture-of-Experts (MoE) aggregation to enable intelligent vision processing at the edge.

### Core Workflow (Corrected)

```
1. Novelty Detection  → CLIP zero-shot classification identifies unknown objects
2. Local Adaptation   → LoRA fine-tuning adapts models for medium-confidence inputs
3. Secure Transmission→ RSA-2048 signatures + Fernet encryption protect data in transit
4. Central Clustering → FAISS vector index groups similar embeddings at the hub
5. MoE Management     → Expert routing and representation gap detection auto-scales experts
```

### Decision Threshold Logic

```
CLIP Confidence Score
        │
        ├── > 0.80  ──► Known         → Log locally, no hub update needed
        ├── 0.50–0.80 ► Adapt_Local   → LoRA fine-tune, then transmit weights
        └── < 0.50  ──► Escalate_Hub  → Transmit raw embedding for hub clustering
```

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           CENTRAL HUB                                │
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────┐ │
│  │  FastAPI Server  │  │   FAISS Index   │  │     MoE Manager      │ │
│  │  (Port 8000)    │  │   (512-dim IP)  │  │  (Gating + Experts)  │ │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬───────────┘ │
│           │                   │                        │             │
│  ┌────────▼────────────────────▼────────────────────────▼──────────┐ │
│  │              MonitoringDashboard (25+ endpoints)                 │ │
│  │   Prometheus /metrics · Grafana Dashboards · AlertManager       │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                      Encrypted Payloads (Fernet + RSA)
                      Task IDs · Status Polling
                                   │
┌──────────────────────────────────▼───────────────────────────────────┐
│                           EDGE NODE (N nodes)                        │
│                                                                      │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │
│  │  CLIP Model  │  │   Custom ViT     │  │   LoRA Adapter       │   │
│  │  (Zero-shot) │  │   (50 classes)   │  │   (rank=8, α=16)     │   │
│  └──────────────┘  └──────────────────┘  └──────────────────────┘   │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │                    SecureTransmitter                           │   │
│  │        Fernet AES-256 · RSA-2048 PKCS#1v1.5 · SHA-256        │   │
│  └───────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### Component Responsibility Matrix

| Component | Layer | Primary Responsibility |
|-----------|-------|----------------------|
| `EdgeVisionNode` | Edge | Zero-shot classification, feature extraction, LoRA adaptation |
| `SecureTransmitter` | Edge | Payload encryption, RSA signing, async HTTP transmission |
| `HubServer` (FastAPI) | Hub | Decryption, FAISS indexing, task queue, MoE routing |
| `MoEManager` | Hub | Expert training, gating network, gap detection |
| `MonitoringDashboard` | Hub | Metric aggregation, alerting, Prometheus export |
| `MetricsCollector` | Hub | Thread-safe counter/gauge/histogram engine |
| `ModelPerformanceMonitor` | Hub | Accuracy, F1, ECE calibration, threshold tracking |
| `InferenceMonitor` | Hub | p50/p95/p99 latency, throughput, GPU/CPU telemetry |
| `SecurityMonitor` | Hub | Signature verification, encryption events, access audit |
| `DriftDetector` | Hub | MMD embedding drift, KS confidence drift, chi-sq cluster drift |
| `AlertManager` | Hub | Configurable rules, console/file/webhook notifications |

---

## 3. Project Directory Structure

```
edge_hub_adaptive_learning/
│
├── central_hub/                    # Hub server and aggregation logic
│   ├── hub_server.py               # FastAPI application entry point
│   ├── faiss_manager.py            # FAISS index operations
│   ├── moe_manager.py              # Mixture-of-Experts management
│   └── task_tracker.py             # Background task queue
│
├── edge_node/                      # Edge device runtime
│   ├── vision_agent.py             # EdgeVisionNode – inference & LoRA
│   ├── secure_transmitter.py       # Encryption, signing, HTTP client
│   └── lora_adapter/               # Saved LoRA weights (runtime-generated)
│       └── adapter_model.bin
│
├── monitoring/                     # Observability stack
│   ├── dashboard.py                # MonitoringDashboard orchestrator
│   ├── metrics_collector.py        # Thread-safe Prometheus-compatible engine
│   ├── model_monitor.py            # Performance + calibration metrics
│   ├── inference_monitor.py        # Latency, throughput, GPU metrics
│   ├── security_monitor.py         # Security event logging
│   ├── drift_detector.py           # Statistical drift detectors
│   └── alerting.py                 # AlertManager + AlertRule definitions
│
├── dashboards/                     # Grafana & Prometheus configuration
│   ├── docker-compose.monitoring.yml
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       ├── datasources/
│       │   └── prometheus.yml
│       └── dashboards/
│           ├── model_performance.json
│           ├── system_health.json
│           └── security_overview.json
│
├── configs/                        # YAML configuration files
│   ├── model_config.yaml
│   ├── data_config.yaml
│   └── class_names.txt             # 50 ImageNet synset class mappings
│
├── model/                          # Pre-trained model weights
│   └── best_vit_model.pth          # Custom ViT checkpoint (~350 MB)
│
├── keys/                           # Cryptographic key storage (gitignored)
│   ├── private_key.pem
│   ├── public_key.pem
│   └── encryption.key
│
├── tests/                          # Test suite
│   ├── unit/
│   │   ├── test_vision_agent.py
│   │   ├── test_secure_transmitter.py
│   │   ├── test_moe_manager.py
│   │   └── test_drift_detector.py
│   ├── integration/
│   │   ├── test_hub_endpoints.py
│   │   └── test_edge_to_hub.py
│   └── e2e/
│       └── test_full_pipeline.py
│
├── scripts/                        # Utility scripts
│   ├── setup_keys.py               # Bootstrap RSA + Fernet key generation
│   ├── run_demo.py                 # Interactive demo runner
│   └── health_check.py            # Standalone liveness probe
│
├── .env.example                    # Environment variable template
├── .gitignore                      # Excludes keys/, *.pem, *.key, __pycache__
├── requirements.txt                # Pinned Python dependencies
├── docker-compose.yml              # Full-stack Docker orchestration
├── Dockerfile.hub                  # Hub container image
├── Dockerfile.edge                 # Edge node container image
└── README.md                       # This file
```

---

## 4. Key Features

### 🎯 Zero-Shot Novelty Detection
- CLIP-based classification without retraining on target domain
- Three-tier decision system: `Known`, `Adapt_Local`, `Escalate_Hub`
- Confidence thresholds: `>0.80` Known · `0.50–0.80` Adapt · `<0.50` Escalate
- Candidate label lists are fully configurable at runtime

### 🔧 Local Adaptation (LoRA)
- Low-Rank Adaptation for efficient fine-tuning (≈0.1% of parameters updated)
- Rank `r=8`, alpha `α=16`, dropout `0.1`
- Target modules: `query`, `value`, `key`, `dense` (all attention layers)
- 5-epoch few-shot training by default; configurable per edge node
- Adapter weights serialized to `.bin` and transmitted to hub on completion

### 🔐 Secure Transmission
- **Fernet (AES-256 CBC + HMAC-SHA256)**: symmetric payload encryption
- **RSA-2048 (PKCS#1 v1.5 + SHA-256)**: digital signature for integrity
- Device identity via UUID-based `device_id` with public key registry at hub
- All keys stored outside version control; loaded from env-var paths

### 📊 Central Aggregation
- **FAISS `IndexFlatIP`**: inner-product similarity on L2-normalized 512-dim embeddings
- **MoE**: soft gating network routes embeddings to specialized expert networks
- **Representation gap detection**: auto-triggers expert creation when cluster size exceeds threshold
- **Background task queue**: non-blocking I/O; heavy aggregation runs in thread pool

### ⚡ Performance
- FP16 (half-precision) on CUDA compute capability ≥ 7.0 (~50% VRAM reduction)
- Automatic device placement (`auto` resolves to `cuda` or `cpu`)
- Efficient LoRA: only adapter params updated during fine-tuning
- Batch processing support for embedding arrays

### 📈 Monitoring & Observability
- **25+ monitoring API endpoints** under `/monitoring/*`
- **Prometheus-compatible** `/monitoring/metrics` scrape endpoint
- **3 Grafana dashboards**: Model Performance, System Health, Security Overview
- **7 default alert rules** with configurable thresholds and cooldowns
- **Drift detection**: MMD (feature), KS test (confidence), chi-squared (cluster)

---

## 5. Prerequisites

### Software Requirements

| Requirement | Minimum Version | Recommended | Notes |
|-------------|----------------|-------------|-------|
| Python | 3.8 | 3.10 | 3.10 tested in CI |
| PyTorch | 2.0 | 2.1 | CUDA 11.8+ for GPU |
| CUDA | 11.0 | 12.1 | Optional; CPU-only mode supported |
| Docker | 20.10 | 24+ | For containerized deployment |
| Docker Compose | 1.29 | 2.20+ | V2 syntax used in compose files |

### Hardware Requirements

| Component | Minimum (CPU-only) | Recommended (GPU) |
|-----------|--------------------|-------------------|
| GPU | N/A | RTX 3050 8 GB (compute cap ≥ 7.0) |
| RAM | 4 GB | 16 GB |
| Storage | 5 GB | 20 GB |
| CPU | 4 cores | 8 cores |

### Network Requirements

| Service | Port | Direction | Protocol |
|---------|------|-----------|----------|
| Central Hub API | 8000 | Inbound from edge nodes | HTTP/HTTPS |
| Prometheus | 9090 | Internal | HTTP |
| Grafana | 3000 | Inbound from operators | HTTP |
| Edge → Hub | outbound to 8000 | Outbound from edge | HTTPS (prod) |

Minimum bandwidth between edge node and hub: **10 Mbps** for real-time streaming.

---

## 6. Environment Variables

Copy `.env.example` and populate before starting any service.

```bash
cp .env.example .env
```

### `.env.example` — Full Manifest

```dotenv
# ─── Hub Server ───────────────────────────────────────────────────────────────
HUB_HOST=0.0.0.0
HUB_PORT=8000
HUB_SECRET_KEY=change-me-in-production          # Used for internal HMAC signing
HUB_WORKERS=4                                    # Uvicorn worker count
HUB_LOG_LEVEL=info                               # debug | info | warning | error

# ─── Cryptographic Key Paths ──────────────────────────────────────────────────
HUB_ENCRYPTION_KEY_PATH=./keys/encryption.key
HUB_PUBLIC_KEY_PATH=./keys/public_key.pem
EDGE_PRIVATE_KEY_PATH=./keys/private_key.pem
EDGE_PUBLIC_KEY_PATH=./keys/public_key.pem

# ─── Model Paths ──────────────────────────────────────────────────────────────
CUSTOM_VIT_WEIGHTS_PATH=./model/best_vit_model.pth
LORA_ADAPTER_PATH=./edge_node/lora_adapter/adapter_model.bin
HF_HOME=~/.cache/huggingface                    # HuggingFace model cache dir

# ─── Hardware ─────────────────────────────────────────────────────────────────
DEVICE=auto                                      # auto | cuda | cpu
USE_FP16=true
FP16_MIN_COMPUTE_CAPABILITY=7

# ─── CLIP & ViT Thresholds ────────────────────────────────────────────────────
CLIP_KNOWN_THRESHOLD=0.80
CLIP_ADAPT_THRESHOLD=0.50

# ─── FAISS Index ──────────────────────────────────────────────────────────────
FAISS_EMBEDDING_DIM=512
FAISS_CLUSTER_THRESHOLD=10                       # Trigger MoE expert creation

# ─── Monitoring ───────────────────────────────────────────────────────────────
ENABLE_MONITORING=true
PROMETHEUS_SCRAPE_INTERVAL=15                    # Seconds
ALERT_WEBHOOK_URL=                               # Optional; leave blank to disable

# ─── Security ─────────────────────────────────────────────────────────────────
CORS_ALLOWED_ORIGINS=http://localhost:3000       # Comma-separated in production
API_KEY_HEADER=X-API-Key                         # Header name for API auth
API_KEY=change-me-in-production

# ─── HuggingFace Mirror (optional, for air-gapped/China networks) ──────────────
# HF_ENDPOINT=https://hf-mirror.com
```

> **Security rule**: Never commit `.env` to version control. The `.gitignore` must include `.env`, `*.pem`, `*.key`.

---

## 7. Installation — Step by Step

### Step 1 — Clone Repository

```bash
git clone https://github.com/YOUR_ORG/edge_hub_adaptive_learning.git
cd edge_hub_adaptive_learning
```

### Step 2 — Create Virtual Environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Pinned `requirements.txt`:**

```
torch==2.1.0
torchvision==0.16.0
transformers==4.35.0
peft==0.6.2
timm==0.9.7
faiss-cpu==1.7.4        # swap for faiss-gpu if CUDA available
fastapi==0.104.0
uvicorn[standard]==0.24.0
httpx==0.25.0           # async HTTP client for SecureTransmitter
cryptography==41.0.5
pillow==10.0.1
numpy==1.26.0
pyyaml==6.0.1
prometheus-client==0.18.0
scipy==1.11.3
scikit-learn==1.3.2
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1
black==23.10.1
flake8==6.1.0
mypy==1.6.1
```

> **GPU note**: Replace `faiss-cpu` with `faiss-gpu` and install the matching CUDA version of PyTorch from https://pytorch.org/get-started/locally/

### Step 4 — Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import faiss; print('FAISS:', faiss.__version__)"
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
```

### Step 5 — Generate Cryptographic Keys

Run the bootstrap script (creates all keys in `./keys/`):

```bash
python scripts/setup_keys.py
```

**`scripts/setup_keys.py` — Reference Implementation:**

```python
#!/usr/bin/env python3
"""Bootstrap script: generates RSA-2048 key pair + Fernet key."""
import os
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

KEYS_DIR = Path("keys")
KEYS_DIR.mkdir(exist_ok=True)

# ── Fernet (AES-256) key ──────────────────────────────────────────────────────
fernet_path = KEYS_DIR / "encryption.key"
if not fernet_path.exists():
    key = Fernet.generate_key()
    fernet_path.write_bytes(key)
    print(f"[✓] Fernet key written → {fernet_path}")
else:
    print(f"[~] Fernet key already exists, skipping.")

# ── RSA-2048 key pair ─────────────────────────────────────────────────────────
private_path = KEYS_DIR / "private_key.pem"
public_path  = KEYS_DIR / "public_key.pem"

if not private_path.exists():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
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
    print(f"[✓] RSA-2048 key pair written → {private_path}, {public_path}")
else:
    print(f"[~] RSA keys already exist, skipping.")

print("\n[✓] Key bootstrap complete. Add keys/ to .gitignore immediately.")
```

### Step 6 — Configure Environment

```bash
cp .env.example .env
# Edit .env with your actual values
```

### Step 7 — Download Pre-trained Models

Models are downloaded automatically on first run. To pre-fetch:

```bash
python -c "
from transformers import CLIPProcessor, CLIPModel
CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
print('[✓] CLIP downloaded (~600 MB)')

from transformers import ViTForImageClassification, ViTFeatureExtractor
ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
print('[✓] ViT downloaded (~350 MB)')
"
```

---

## 8. Configuration Files

### `configs/model_config.yaml`

```yaml
custom_vit:
  enabled: true
  architecture: "vit_base_patch16_224"
  source: "timm"
  num_classes: 50
  weights_path: "../model/best_vit_model.pth"
  input_size: 224
  patch_size: 16
  normalization:
    mean: [0.485, 0.456, 0.406]
    std:  [0.229, 0.224, 0.225]

clip:
  enabled: true
  model_name: "openai/clip-vit-base-patch32"
  source: "huggingface"
  embedding_dim: 512
  thresholds:
    known: 0.80          # Above this → Known (no hub update)
    adapt: 0.50          # Between adapt and known → Adapt_Local

lora:
  enabled: true
  r: 8                   # Rank — controls parameter budget
  alpha: 16              # Scaling factor
  target_modules: ["query", "value", "key", "dense"]
  dropout: 0.1
  bias: "none"
  task_type: "SEQ_CLS"
  num_epochs: 5

hardware:
  device: "auto"         # auto | cuda | cpu
  use_fp16: true
  fp16_min_capability: 7 # Volta (V100) or newer
```

### `configs/data_config.yaml`

```yaml
preprocessing:
  input_size: 224
  interpolation: "bilinear"
  center_crop: true
  crop_size: 224

normalization:
  mean: [0.485, 0.456, 0.406]
  std:  [0.229, 0.224, 0.225]

augmentation:
  enabled: true
  horizontal_flip: true
  rotation_range: 15
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1

dataloader:
  batch_size: 32
  num_workers: 4
  pin_memory: true
  drop_last: false
```

---

## 9. Security Setup

### Encryption Pipeline (Detail)

```
Edge Node                              Hub Server
──────────                             ──────────
1. Build JSON payload
   {device_id, timestamp,
    embedding, adapter_weights}

2. Sign with RSA private key
   signature = RSA.sign(
     SHA256(payload), private_key)

3. Encrypt with Fernet key
   ciphertext = Fernet.encrypt(
     payload + signature)

4. POST /ingress_update
   {encrypted_payload: base64(ciphertext),
    device_id: "edge_device_001"}
                    ──────────────────────►
                                          5. Fernet.decrypt(ciphertext)
                                          6. RSA.verify(signature,
                                               public_key[device_id])
                                          7. FAISS.add(embedding)
                                          8. MoE.route(embedding)
```

### Generate Keys (Manual Alternative)

```python
from edge_node.secure_transmitter import SecureTransmitter
from cryptography.fernet import Fernet

# RSA key pair
SecureTransmitter.generate_rsa_keypair(
    private_key_path='keys/private_key.pem',
    public_key_path='keys/public_key.pem',
    key_size=2048
)

# Fernet symmetric key
key = Fernet.generate_key()
with open('keys/encryption.key', 'wb') as f:
    f.write(key)
```

### Security Best Practices

```bash
# 1. Add all key files to .gitignore
echo "keys/" >> .gitignore
echo "*.pem"  >> .gitignore
echo "*.key"  >> .gitignore

# 2. Set restrictive permissions on key files
chmod 600 keys/private_key.pem
chmod 600 keys/encryption.key
chmod 644 keys/public_key.pem
```

- Rotate RSA keys every **90 days** and Fernet keys every **30 days**
- In production, restrict CORS origins and add TLS termination (nginx/Caddy)
- Store keys in a secrets manager (HashiCorp Vault, AWS Secrets Manager) rather than on-disk

---

## 10. Running the System

### 10.1 Start the Central Hub

```bash
cd central_hub
uvicorn hub_server:app --host 0.0.0.0 --port 8000 --workers 4
```

Or via the module directly:

```bash
python hub_server.py
```

Verify the hub is operational:

```bash
curl http://localhost:8000/health
# {"status": "healthy", "timestamp": "..."}

curl http://localhost:8000/ready
# {"status": "ready", "index_initialized": true, "moe_ready": true}
```

### 10.2 Edge Node — Synchronous Usage

```python
from edge_node.vision_agent import EdgeVisionNode
from PIL import Image

# Initialize (device auto-detected)
node = EdgeVisionNode(device='auto', use_fp16=True)

# Load image
image = Image.open('test_image.jpg').convert('RGB')

# Novelty detection
decision, scores, labels = node.detect_novelty(
    image,
    candidate_labels=['car', 'truck', 'person', 'unknown_object']
)
print(f"Decision: {decision}")          # Known | Adapt_Local | Escalate_Hub
print(f"Top label: {labels[0]} ({scores[0]:.2%})")

# Conditional local adaptation
if decision == 'Adapt_Local':
    node.local_adaptation(image, pseudo_label=labels[0])
    node._save_adapter_weights()
```

### 10.3 Edge Node — Async Transmission

> **Important**: `SecureTransmitter.transmit()` is a coroutine and must be called inside an async context or via `asyncio.run()`.

```python
import asyncio
import torch
from edge_node.secure_transmitter import SecureTransmitter

async def send_to_hub():
    transmitter = SecureTransmitter(
        adapter_weights_path='edge_node/lora_adapter/adapter_model.bin',
        hub_url='http://localhost:8000',
        device_id='edge_device_001',
        key_path='keys/encryption.key',
        private_key_path='keys/private_key.pem',
        public_key_path='keys/public_key.pem'
    )

    # Generate or retrieve a real CLIP embedding
    clip_embedding = torch.randn(1, 512)   # Replace with node.extract_features(image)

    result = await transmitter.transmit(
        clip_embedding,
        metadata={'model_version': '1.0', 'epoch': 5},
        sign_payload=True
    )

    print(f"Transmission success: {result['success']}")
    print(f"Cluster assigned: {result['hub_response'].get('cluster_id')}")
    print(f"Task ID: {result['hub_response'].get('task_id')}")

# Entry point
asyncio.run(send_to_hub())
```

### 10.4 Poll Task Completion

Background tasks at the hub are tracked asynchronously. Poll to confirm completion:

```python
import httpx, time

task_id = "550e8400-e29b-41d4-a716-446655440000"   # From transmit response

for _ in range(10):
    r = httpx.get(f"http://localhost:8000/tasks/{task_id}")
    data = r.json()
    if data["status"] == "completed":
        print("Task complete:", data["result"])
        break
    time.sleep(1.5)
```

### 10.5 Demo Script

```bash
python scripts/run_demo.py
```

Demo menu options:
1. **Edge Node Only** — simulate detection without real models (mock data)
2. **Full Pipeline** — complete workflow from CLIP detection to hub aggregation
3. **MoE Manager** — expert routing demonstration
4. **Encryption Flow** — Fernet + RSA sign/verify walkthrough
5. **Run All** — execute all demos sequentially

### 10.6 Graceful Shutdown

The hub server listens for `SIGTERM` / `SIGINT` and drains in-flight requests:

```bash
# Send graceful shutdown signal
kill -SIGTERM $(lsof -ti:8000)

# Or use uvicorn's --timeout-graceful-shutdown flag
uvicorn hub_server:app --timeout-graceful-shutdown 30
```

Ensure edge nodes retry or queue transmissions if the hub becomes temporarily unavailable. Use exponential backoff in `SecureTransmitter`:

```python
# Recommended retry config in SecureTransmitter.__init__
self.retry_attempts = 3
self.retry_backoff = [1, 3, 9]  # seconds
```

---

## 11. Docker Deployment

### 11.1 Full-Stack `docker-compose.yml`

```yaml
version: "3.9"

services:

  # ── Central Hub ──────────────────────────────────────────────────────────────
  hub:
    build:
      context: .
      dockerfile: Dockerfile.hub
    image: edge-hub:latest
    container_name: edge_hub
    ports:
      - "8000:8000"
    volumes:
      - ./keys:/app/keys:ro
      - ./model:/app/model:ro
      - ./configs:/app/configs:ro
      - hub_faiss_data:/app/data
    env_file: .env
    environment:
      - HUB_HOST=0.0.0.0
      - HUB_PORT=8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 15s
      timeout: 5s
      retries: 3
      start_period: 20s
    restart: unless-stopped
    networks:
      - edge_net

  # ── Edge Node (example single node; scale as needed) ─────────────────────────
  edge_node:
    build:
      context: .
      dockerfile: Dockerfile.edge
    image: edge-node:latest
    container_name: edge_node_01
    depends_on:
      hub:
        condition: service_healthy
    volumes:
      - ./keys:/app/keys:ro
      - ./configs:/app/configs:ro
      - edge_lora:/app/edge_node/lora_adapter
    env_file: .env
    environment:
      - HUB_URL=http://hub:8000
      - DEVICE_ID=edge_node_01
    restart: unless-stopped
    networks:
      - edge_net

  # ── Prometheus ───────────────────────────────────────────────────────────────
  prometheus:
    image: prom/prometheus:v2.47.0
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./dashboards/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=15d'
    networks:
      - edge_net
    restart: unless-stopped

  # ── Grafana ──────────────────────────────────────────────────────────────────
  grafana:
    image: grafana/grafana:10.1.0
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./dashboards/grafana/datasources:/etc/grafana/provisioning/datasources:ro
      - ./dashboards/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    depends_on:
      - prometheus
    networks:
      - edge_net
    restart: unless-stopped

volumes:
  hub_faiss_data:
  edge_lora:
  prometheus_data:
  grafana_data:

networks:
  edge_net:
    driver: bridge
```

### 11.2 `Dockerfile.hub`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY central_hub/ ./central_hub/
COPY monitoring/   ./monitoring/
COPY configs/      ./configs/

EXPOSE 8000

CMD ["uvicorn", "central_hub.hub_server:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 11.3 `Dockerfile.edge`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY edge_node/ ./edge_node/
COPY configs/   ./configs/

CMD ["python", "edge_node/vision_agent.py"]
```

### 11.4 Start All Services

```bash
# Build and start everything
docker compose up --build -d

# Verify services
docker compose ps

# View hub logs
docker compose logs -f hub

# Scale edge nodes horizontally
docker compose up --scale edge_node=3 -d
```

### 11.5 Access Points (Post-Startup)

| Service | URL | Credentials |
|---------|-----|-------------|
| Hub API | http://localhost:8000 | API key (see `.env`) |
| Hub Docs | http://localhost:8000/docs | None (dev only) |
| Prometheus | http://localhost:9090 | None |
| Grafana | http://localhost:3000 | admin / admin |
| Hub Metrics | http://localhost:8000/monitoring/metrics | None |

---

## 12. API Reference

### Health & Readiness (New)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe — returns `{"status":"healthy"}` |
| `/ready` | GET | Readiness probe — checks FAISS + MoE initialization |

### Core Hub Endpoints

#### `POST /ingress_update`

Receive and process an encrypted update from an edge device.

**Request:**
```json
{
  "encrypted_payload": "<base64-encoded-bytes>",
  "device_id": "edge_device_001"
}
```

**Response `200 OK`:**
```json
{
  "status": "success",
  "cluster_id": 5,
  "total_embeddings": 150,
  "signature": "a1b2c3d4...",
  "timestamp": "2026-04-06T12:00:00.000000Z",
  "message": "Successfully aggregated update from edge_device_001",
  "task_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Error `422`:** Payload decryption or signature verification failure.

---

#### `GET /status`
```json
{
  "status": "operational",
  "total_embeddings": 150,
  "total_devices": 5,
  "index_initialized": true,
  "timestamp": "2026-04-06T12:00:00.000000Z"
}
```

#### `GET /clusters`
```json
{
  "total_clusters": 3,
  "total_embeddings": 150,
  "clusters": {
    "0": {"size": 50, "entries": [...]},
    "1": {"size": 45, "entries": [...]},
    "2": {"size": 55, "entries": [...]}
  }
}
```

#### `GET /moe/status`
```json
{
  "status": "operational",
  "num_experts": 3,
  "embedding_dim": 512,
  "experts": [
    {"expert_id": 0, "cluster_id": 0, "status": "active"},
    {"expert_id": 1, "cluster_id": 1, "status": "active"}
  ],
  "timestamp": "2026-04-06T12:00:00.000000Z"
}
```

#### `GET /moe/representation-gap?cluster_threshold=10`
```json
{
  "has_gap": true,
  "gap_info": {
    "cluster_id": 2,
    "cluster_size": 15,
    "threshold": 10,
    "timestamp": "2026-04-06T12:00:00.000000Z"
  }
}
```

#### `POST /moe/create-expert?cluster_id=2`
```json
{
  "status": "success",
  "new_expert": {
    "expert_id": 3,
    "training_samples": 55,
    "training_epochs": 50,
    "final_loss": 0.0023,
    "cluster_id": 2,
    "total_experts": 4
  }
}
```

#### `GET /tasks/{task_id}`
```json
{
  "task_id": "550e8400-...",
  "status": "completed",
  "task_type": "ingress_processing",
  "progress": 1.0,
  "result": {"cluster_id": 5, "total_embeddings": 151},
  "created_at": "2026-04-06T12:00:00.000000Z",
  "completed_at": "2026-04-06T12:00:01.500000Z"
}
```

#### `POST /reset`
Resets FAISS index and all metadata. **Caution: destructive in production.**

```json
{"status": "reset", "message": "Hub has been reset successfully"}
```

### Monitoring Endpoints (Prefix: `/monitoring`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/monitoring/status` | GET | Monitoring system status and uptime |
| `/monitoring/metrics` | GET | **Prometheus text format** for scraping |
| `/monitoring/metrics-json` | GET | All metrics as JSON |
| `/monitoring/metrics-list` | GET | List all registered metric names |
| `/monitoring/dashboard-summary` | GET | **Complete system overview** |
| `/monitoring/model-performance` | GET | Accuracy, F1, Precision, Recall, Novelty |
| `/monitoring/adaptation-stats` | GET | LoRA pre/post accuracy delta, training loss |
| `/monitoring/inference-health` | GET | Latency p50/p95/p99, throughput, GPU/CPU |
| `/monitoring/security-audit` | GET | Signature rate, encryption failures |
| `/monitoring/drift-report` | GET | MMD, KS, chi-squared drift scores |
| `/monitoring/alerts` | GET | Active + historical alerts |
| `/monitoring/alerts/active` | GET | Only currently firing alerts |
| `/monitoring/alerts/history` | GET | Historical alert log with filtering |
| `/monitoring/perf/latency` | GET | Latency percentiles |
| `/monitoring/perf/throughput` | GET | Images/sec throughput |
| `/monitoring/hub/health` | GET | Hub embeddings, clusters, experts |
| `/monitoring/calibration` | GET | ECE, MCE, per-bin accuracy |
| `/monitoring/novelty` | GET | Novel detection rate, false novelty rate |
| `/monitoring/confusion-matrix` | GET | N×N confusion matrix |
| `/monitoring/per-class` | GET | Per-class Precision/Recall/F1 |
| `/monitoring/system` | GET | CPU, GPU, memory usage |
| `/monitoring/drift/check` | POST | Trigger drift check manually |
| `/monitoring/alerts/evaluate` | POST | Trigger alert rule evaluation manually |

---

## 13. Monitoring & Observability

### Quick Start

```bash
# 1. Start Prometheus + Grafana
cd dashboards
docker compose -f docker-compose.monitoring.yml up -d

# 2. Start Hub (monitoring auto-mounted)
cd ../central_hub
python hub_server.py

# 3. Open dashboards
# Grafana:    http://localhost:3000  (admin / admin)
# Prometheus: http://localhost:9090
# API:        http://localhost:8000/monitoring/dashboard-summary
# Metrics:    http://localhost:8000/monitoring/metrics
```

### Grafana Dashboards

**1. Model Performance Dashboard**
- Accuracy, F1, Precision, Recall gauges with color-coded thresholds
- Novelty detection rate and false novelty rate trending
- ECE calibration error over time
- Threshold decision distribution (Known / Adapt Local / Escalate Hub)

**2. System Health Dashboard**
- Inference latency p50 / p95 / p99 time-series
- Throughput (images/sec)
- CPU and GPU usage gauges
- Hub health: total embeddings, active devices, cluster count, expert count
- LoRA adaptation: accuracy delta, training/validation loss, overfitting gap

**3. Security Overview Dashboard**
- RSA signature verification success/failure rate
- Fernet encryption/decryption operation rates
- Payload rejection rate and transmission success rate
- Drift detection scores (embedding MMD, confidence KS, cluster chi-sq)
- Active alert severity breakdown

### Metrics Categories

| Category | Metrics | Source Component |
|----------|---------|----------------|
| Classification | Accuracy, Precision, Recall, F1 | `ModelPerformanceMonitor` |
| Novelty | Novel detection rate, False novelty rate | `ModelPerformanceMonitor` |
| Calibration | ECE, MCE, per-bin accuracy | `ModelPerformanceMonitor` |
| Adaptation | Pre/post accuracy delta, loss, duration | `AdaptationMonitor` |
| Inference | Latency p50/p95/p99, throughput | `InferenceMonitor` |
| Hub | Embeddings, devices, clusters, experts | `InferenceMonitor` |
| Security | Signature rate, encrypt/decrypt failures | `SecurityMonitor` |
| Drift | Embedding MMD, confidence KS, cluster chi-sq | `DriftDetector` |
| Alerts | Active count, fired/resolved totals | `AlertManager` |

### Default Alert Rules

| Alert Name | Condition | Severity | Cooldown |
|------------|-----------|----------|---------|
| `accuracy_drop` | Accuracy < 0.50 | Error | 300s |
| `high_novelty_false_positives` | False novelty rate > 30% | Warning | 300s |
| `signature_failure_spike` | Signature failure rate > 10% | Critical | 60s |
| `high_inference_latency` | P95 latency > 500ms | Warning | 120s |
| `gpu_memory_warning` | GPU memory > 8 GB | Warning | 600s |
| `task_failure_spike` | Task failure rate > 20% | Error | 120s |
| `drift_detected` | Any drift score > threshold | Warning | 600s |

### Custom Alert Webhook

```python
from monitoring.alerting import AlertManager, AlertRule, AlertSeverity
from monitoring.metrics_collector import MetricsCollector

collector = MetricsCollector()
manager = AlertManager()

# Add webhook notification channel
manager.add_webhook("https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK")

# Define custom rule
manager.add_rule(AlertRule(
    name="custom_accuracy_threshold",
    description="Accuracy dropped below 60%",
    condition=lambda: collector.get("accuracy") < 0.60,
    severity=AlertSeverity.ERROR,
    cooldown_seconds=300,
    message_template="[ALERT] Model accuracy is below 60%. Current: {value:.2%}",
))
```

---

## 14. Testing Strategy

### Unit Tests

```bash
pytest tests/unit/ -v --cov=. --cov-report=term-missing
```

Key unit test targets:

| Test File | What It Covers |
|-----------|---------------|
| `test_vision_agent.py` | CLIP threshold logic, LoRA apply/save, device placement |
| `test_secure_transmitter.py` | Fernet encrypt/decrypt roundtrip, RSA sign/verify |
| `test_moe_manager.py` | Expert creation, gating softmax, gap detection |
| `test_drift_detector.py` | MMD calculation, KS test correctness, chi-sq cluster |

### Integration Tests

```bash
pytest tests/integration/ -v
```

| Test File | What It Covers |
|-----------|---------------|
| `test_hub_endpoints.py` | All FastAPI routes with TestClient; mocked FAISS |
| `test_edge_to_hub.py` | Full encrypt→transmit→decrypt→index cycle; mocked hub |

### End-to-End Tests

```bash
# Requires hub running on localhost:8000
pytest tests/e2e/ -v --hub-url=http://localhost:8000
```

`test_full_pipeline.py` exercises:
1. Initialize `EdgeVisionNode`
2. Run novelty detection on a synthetic image
3. Trigger LoRA adaptation
4. Transmit adapter weights to hub
5. Poll task completion
6. Verify embedding indexed in FAISS via `/clusters`

### Coverage Target

```bash
pytest --cov=. --cov-report=html --cov-fail-under=80
open htmlcov/index.html
```

Minimum coverage threshold: **80%** (enforced in CI).

---

## 15. Multi-Node Deployment

### Horizontal Scaling (Docker Compose)

```bash
# Scale to 5 edge nodes
docker compose up --scale edge_node=5 -d

# Each node gets its own DEVICE_ID via container name
# Configure in Dockerfile.edge:
# ENV DEVICE_ID=${HOSTNAME}
```

### Separate Hub and Edge Networks

In production, edge nodes and the hub should be on separate network segments:

```yaml
# docker-compose.prod.yml
services:
  hub:
    networks:
      - hub_internal
      - edge_dmz          # Hub exposed to edge only in DMZ

  edge_node:
    networks:
      - edge_dmz          # Edge only reaches hub in DMZ; no hub_internal access

networks:
  hub_internal:
    internal: true        # Prometheus, Grafana, FAISS — internal only
  edge_dmz:
    driver: bridge        # Edge nodes connect here
```

### Hub High Availability (Advanced)

For production HA, place multiple hub replicas behind a load balancer with a shared FAISS index (requires FAISS with shared memory or external vector DB like Milvus/Qdrant):

```
Load Balancer (nginx)
    ├── Hub Replica 1 → Shared Vector Store
    ├── Hub Replica 2 → Shared Vector Store
    └── Hub Replica 3 → Shared Vector Store
```

This pattern is outside the current codebase scope but is the recommended evolution path.

---

## 16. CI/CD Pipeline

### GitHub Actions (`.github/workflows/ci.yml`)

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache pip
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}

      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-asyncio black flake8 mypy

      - name: Lint (flake8)
        run: flake8 . --max-line-length=120 --exclude=venv,__pycache__

      - name: Format check (black)
        run: black --check .

      - name: Type check (mypy)
        run: mypy central_hub/ edge_node/ monitoring/ --ignore-missing-imports

      - name: Unit tests with coverage
        run: pytest tests/unit/ -v --cov=. --cov-report=xml --cov-fail-under=80

      - name: Upload coverage report
        uses: codecov/codecov-action@v3
        with:
          files: coverage.xml

  build-docker:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Build Hub image
        run: docker build -f Dockerfile.hub -t edge-hub:${{ github.sha }} .

      - name: Build Edge image
        run: docker build -f Dockerfile.edge -t edge-node:${{ github.sha }} .
```

---

## 17. Performance Optimization

### FP16 Inference

Automatically activated when `CUDA` is available and GPU compute capability ≥ 7.0 (Volta or newer).

| Model | FP32 VRAM | FP16 VRAM | Savings |
|-------|-----------|-----------|---------|
| CLIP ViT-B/32 | 600 MB | 300 MB | 50% |
| ViT-Base/16 | 350 MB | 175 MB | 50% |
| LoRA Adapter | 10 MB | 5 MB | 50% |
| **Total** | **960 MB** | **480 MB** | **~50%** |

### LoRA Efficiency

| Parameter | Value | Effect |
|-----------|-------|--------|
| Rank `r` | 8 | Minimal memory overhead |
| Alpha `α` | 16 | Stable convergence |
| Target modules | query, value, key, dense | All attention layers adapted |
| Trainable params | ~0.1% of total | Extremely fast fine-tuning |

### DataLoader Settings

```yaml
dataloader:
  batch_size: 32        # Reduce to 16 if GPU OOM
  num_workers: 4        # Set to (cpu_cores - 2)
  pin_memory: true      # Faster CPU→GPU transfers
  drop_last: false
```

### GPU Memory Management

```python
# If OOM errors occur:
import torch

# 1. Reduce batch size in data_config.yaml
# 2. Force CPU mode
node = EdgeVisionNode(device='cpu', use_fp16=False)

# 3. Clear GPU cache between batches
torch.cuda.empty_cache()

# 4. Use gradient checkpointing for LoRA training
model.gradient_checkpointing_enable()
```

---

## 18. Troubleshooting

### CUDA Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory`

```python
# Reduce batch size (data_config.yaml)
batch_size: 16

# Or switch to CPU
node = EdgeVisionNode(device='cpu', use_fp16=False)

# Clear cache
import torch; torch.cuda.empty_cache()
```

### HuggingFace Model Download Fails

**Symptom:** `HTTPError: 403 Forbidden` or connection timeout

```bash
# Set cache directory
export HF_HOME=/path/to/cache

# Use mirror (for restricted networks)
export HF_ENDPOINT=https://hf-mirror.com

# Manual download: visit https://huggingface.co/openai/clip-vit-base-patch32
# then set: CLIPModel.from_pretrained('/local/path/to/model')
```

### FAISS Index Error

**Symptom:** `faiss: Error in void faiss::read_index`

```bash
pip uninstall faiss-cpu faiss-gpu
pip install faiss-cpu      # or faiss-gpu

# Reset hub index (non-destructive in dev)
curl -X POST http://localhost:8000/reset
```

### RSA Key Loading Error

**Symptom:** `ValueError: Could not deserialize key data`

```python
# Verify key format
with open('keys/private_key.pem', 'rb') as f:
    first_line = f.read(50)
    assert b'BEGIN PRIVATE KEY' in first_line, "Wrong key format"

# Regenerate keys
python scripts/setup_keys.py
```

### LoRA Adapter Not Saving

**Symptom:** No adapter file created after `local_adaptation()`

```python
import os
os.makedirs('edge_node/lora_adapter', exist_ok=True)
node.lora_adapter_path = os.path.abspath('edge_node/lora_adapter/adapter_model.bin')
node._save_adapter_weights()
```

### Hub Not Reachable from Edge Node

**Symptom:** `httpx.ConnectError: [Errno 111] Connection refused`

```bash
# Verify hub is running
curl http://localhost:8000/health

# In Docker, use service name not localhost
HUB_URL=http://hub:8000     # NOT http://localhost:8000
```

### Debug Mode

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

node = EdgeVisionNode(device='auto', use_fp16=True)
```

### Performance Profiling

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True,
) as prof:
    node.classify_image(image)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")   # Open in chrome://tracing
```

---

## 19. Contributing

### Development Setup

```bash
git clone https://github.com/YOUR_ORG/edge_hub_adaptive_learning.git
cd edge_hub_adaptive_learning
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-cov pytest-asyncio black flake8 mypy
```

### Code Standards

- Follow **PEP 8** with 120-char line limit
- Use **type hints** on all public function signatures
- Add **docstrings** (Google style) for all public classes and methods
- Keep functions under **50 lines**; extract helpers for complex logic
- All new code requires corresponding **unit tests** with ≥80% coverage

### Branch Strategy

```
main          → stable, production-ready
develop       → integration branch for PRs
feature/*     → individual feature branches
hotfix/*      → urgent production fixes
```

### Pull Request Checklist

- [ ] Feature branch created from `develop`
- [ ] Tests written and passing (`pytest tests/unit/ -v`)
- [ ] Code formatted (`black .`)
- [ ] Linted (`flake8 .`)
- [ ] Type-checked (`mypy central_hub/ edge_node/ monitoring/`)
- [ ] PR description explains the change and links to any issue
- [ ] Reviewed and approved by at least one maintainer

---

## 20. Versioning & Changelog

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking API changes (endpoint removal, payload schema change)
- **MINOR**: Backward-compatible new features (new endpoint, new expert strategy)
- **PATCH**: Backward-compatible bug fixes

### CHANGELOG.md Format

```markdown
## [1.1.0] - 2026-04-06
### Added
- `/health` and `/ready` liveness/readiness probe endpoints
- `scripts/setup_keys.py` automated key bootstrap script
- Async retry logic with exponential backoff in SecureTransmitter

### Fixed
- Typo in core workflow: "Novty Detection" → "Novelty Detection"
- Missing `asyncio.run()` wrapper in usage examples

### Changed
- Default Fernet key rotation recommendation: 30 days (was not documented)

## [1.0.0] - 2026-01-01
### Added
- Initial release: CLIP novelty detection, LoRA adaptation, FAISS hub, MoE manager
- 25+ monitoring endpoints with Prometheus export
- 3 Grafana dashboards
```

---

## 21. License & Acknowledgments

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

### Acknowledgments

| Library | Purpose | Link |
|---------|---------|------|
| OpenAI CLIP | Zero-shot vision-language model | https://github.com/openai/CLIP |
| Hugging Face Transformers | Model implementations | https://github.com/huggingface/transformers |
| PEFT | LoRA adaptation library | https://github.com/huggingface/peft |
| FAISS | Efficient similarity search | https://github.com/facebookresearch/faiss |
| timm | Vision Transformer implementations | https://github.com/huggingface/pytorch-image-models |
| FastAPI | Async API framework | https://fastapi.tiangolo.com |
| Prometheus | Metrics collection and alerting | https://prometheus.io |
| Grafana | Metrics visualization | https://grafana.com |

---

**Built with ❤️ for Edge AI and IoT Applications**

> For questions, open an issue on GitHub or reach out via the project's discussion board.
