# 🚀 Edge Hub - Complete Setup, Endpoints & Testing Guide (Detailed)

## Project Structure
```
d:/Iot Model/
├── central_hub/           # Hub server (main)
│   ├── hub_server.py      # FastAPI app :8000 (ALL endpoints)
│   ├── moe_manager.py     # MoE experts
│   ├── faiss_manager.py   # Embeddings index
│   ├── fed_avg.py         # FedAvg aggregator
│   └── adapter_registry.py # Devices/adapters
├── monitoring/            # Mounted /monitoring/*
│   └── dashboard.py       # Dashboards/metrics
├── edge_node/             # Edge clients (no server)
├── tests/                 # pytest + E2E
├── requirements.txt       # Deps
├── test_system_v2.py      # E2E test script
└── keys/, hub_data/       # Runtime data ✓
```

## 1. Prerequisites & Setup

**Windows/PowerShell** (current dir `d:/Iot Model`):
```powershell
# 1. Fix/clean env
deactivate
Remove-Item -Recurse -Force venv -ErrorAction SilentlyContinue

# 2. Install all deps (torch/fastapi/uvicorn etc.)
python -m pip install --user -r requirements.txt

# Verify
python -c "import uvicorn, fastapi, torch, faiss; print('Ready!')"
```

## 2. Start Systems

**Terminal 1: Hub Server** (keep running):
```powershell
python -m uvicorn central_hub.hub_server:app --host 0.0.0.0 --port 8000 --reload
```
- Logs: `Uvicorn running... startup complete`, `test_device_001 registered`.
- Access: [Swagger](http://localhost:8000/docs) | [Redoc](http://localhost:8000/redoc)

**No other servers** - All endpoints (monitoring/MoE/FedAvg) in hub.

**Stop**: Ctrl+C.

## 3. All Endpoints (Categorized + Curl Tests)

### 🔍 Health & Status
| Endpoint | Method | Curl | Expected |
|----------|--------|------|----------|
| `/health` | GET | `curl http://localhost:8000/health` | `{"status":"healthy"}` |
| `/ready` | GET | `curl http://localhost:8000/ready` | `{"status":"ready",...}` |
| `/status` | GET | `curl http://localhost:8000/status` | `{"status":"operational","total_embeddings":0,...}` |
| `/reset` | POST | `curl -X POST http://localhost:8000/reset` | `{"status":"reset"}` |

### 📱 Devices/Registry (Fixed 404)
| Endpoint | Method | Curl | Expected |
|----------|--------|------|----------|
| `/devices` | GET | `curl http://localhost:8000/devices` | `{"total_devices":1,...}` |
| `/devices/{id}/status` | GET | `curl http://localhost:8000/devices/test_device_001/status` | `{"device_id":"test_device_001","is_current":true,...}` |
| `/devices/register` | POST | `curl -X POST -d '{\"device_id\":\"new\",\"adapter_version\":0}' http://localhost:8000/devices/register` | `{"status":"registered"}` |

### 🧠 MoE/Mixture of Experts (Fixed 500)
| Endpoint | Method | Curl | Expected |
|----------|--------|------|----------|
| `/moe/status` | GET | `curl http://localhost:8000/moe/status` | `{"num_experts":0,"embedding_dim":512}` |
| `/moe/representation-gap` | GET | `curl 'http://localhost:8000/moe/representation-gap?cluster_threshold=5'` | `{"has_gap":true}` |
| `/moe/create-expert` | POST | `curl -X POST 'http://localhost:8000/moe/create-expert?cluster_id=0'` | `{"status":"skipped",...}` **(200 ✓)** |

### 📊 Monitoring/Dashboards (Fixed 404)
| Endpoint | Method | Curl | Expected |
|----------|--------|------|----------|
| `/monitoring/dashboard` | GET | `curl http://localhost:8000/monitoring/dashboard` | `{"status":"operational","inference":...,"system":...}` |
| `/monitoring/hub/stats` | GET | `curl http://localhost:8000/monitoring/hub/stats` | `{"status":"operational","total_devices":2,...}` |
| `/monitoring/hub/health` | GET | `curl http://localhost:8000/monitoring/hub/health` | `{"total_embeddings":0,...}` |
| `/monitoring/status` | GET | `curl http://localhost:8000/monitoring/status` | `{"status":"operational"}` |
| `/monitoring/alerts` | GET | `curl http://localhost:8000/monitoring/alerts` | `{"active":[],...}` |

### 🌐 FedAvg (Fixed Not Found)
| Endpoint | Method | Curl | Expected |
|----------|--------|------|----------|
| `/fedavg/status` | GET | `curl http://localhost:8000/fedavg/status` | `{"status":"operational","global_version":0}` |

### 🔄 Core Flows
| Flow | Curl | Expected |
|------|------|----------|
| Ingress | (via test) | 200 task_id (bg process) |
| Tasks | `curl http://localhost:8000/tasks/{task_id}` | Task status |
| Adapters | `curl http://localhost:8000/adapters/latest/version` | `{"version":0,"checksum":""}` |

## 4. Automated Tests (Terminal 2)

```powershell
# Comprehensive E2E (endpoints + edge-hub)
python test_system_v2.py
# Expected: ALL [OK], no 404/500

# Unit/integration
pytest tests/integration/ -v
# Expected: 100% passed

# System tests
python test_system.py
```

## 5. Demo Full Flow (Create Experts!)

```powershell
# 1. Run edge tests (adds embeddings)
python test_system.py

# 2. Check data
curl http://localhost:8000/status  # embeddings>0

# 3. Create expert
curl -X POST 'http://localhost:8000/moe/create-expert?cluster_id=0'  # created!

# 4. Verify
curl http://localhost:8000/moe/status  # num_experts>0
curl http://localhost:8000/monitoring/hub/stats
```

## 6. Troubleshooting

| Issue | Fix |
|-------|-----|
| Port busy | `netstat -ano | findstr :8000` → `taskkill /PID <id> /F` |
| No uvicorn | `python -m pip install --user uvicorn[standard] fastapi` |
| ImportError | `python -m pip install --user -r requirements.txt --force-reinstall` |
| No test_device | Server auto-registers on startup |
| Experts=0 | Run test_system.py → POST /moe/create-expert |

**Reset**: `curl -X POST http://localhost:8000/reset`

## Original Issues Fixed
- `/devices/test_device_001/status` 404 → 200 auto-reg
- `/moe/create-expert` 500 → 200 skipped/created
- Monitoring 404 → 200 mounted router
- FedAvg → /fedavg/status 200

**Success!** Copy-paste ready. 🎉
