# Edge Hub Adaptive Learning System - Setup & Testing Guide

## Overview
- **Hub Server** (port 8000): Main FastAPI server (central_hub/hub_server.py). Includes MoE, FedAvg, monitoring (/monitoring/*), devices.
- **No Separate Servers**: Edge/monitoring integrated. Edge_node/ for clients.
- **Requirements**: Python 3.10+, deps in requirements.txt.

## 1. Setup (Windows/PowerShell)

```powershell
# Current dir: d:/Iot Model
# Remove broken venv if any
deactivate; Remove-Item -Recurse -Force venv -ErrorAction SilentlyContinue

# Install deps (system Python --user safe)
python -m pip install --user -r requirements.txt
```

## 2. Start Hub Server (Terminal 1 - Keep Running)

```powershell
python -m uvicorn central_hub.hub_server:app --host 0.0.0.0 --port 8000 --reload
```

**Expected Logs**:
```
INFO: Uvicorn running on http://0.0.0.0:8000
INFO: Application startup complete.
[Hub] Auto-registered test device: test_device_001
```

**Access**:
- Swagger: http://localhost:8000/docs
- Re-docs: http://localhost:8000/redoc

## 3. Endpoint Testing

### Health/Status
```powershell
curl http://localhost:8000/health                # healthy
curl http://localhost:8000/status               # operational
curl http://localhost:8000/ready                # ready
curl http://localhost:8000/devices              # list devices
```

### Devices (Fixed 404 → 200)
```powershell
curl http://localhost:8000/devices/test_device_001/status  # 200 DeviceStatus (auto-reg)
curl -X POST http://localhost:8000/devices/register -d "{\"device_id\":\"test2\",\"adapter_version\":0}"  # register new
```

### MoE (Fixed 500 → 200)
```powershell
curl http://localhost:8000/moe/status                                # num_experts:0
curl "http://localhost:8000/moe/representation-gap?cluster_threshold=5"  # has_gap:true
curl -X POST "http://localhost:8000/moe/create-expert?cluster_id=0"  # skipped (no embeddings)
```

### Monitoring (Fixed 404 → 200)
```powershell
curl http://localhost:8000/monitoring/dashboard     # operational metrics
curl http://localhost:8000/monitoring/hub/stats     # hub stats (devices/embeddings)
curl http://localhost:8000/monitoring/hub/health    # health
```

### FedAvg (Fixed Not Found → 200)
```powershell
curl http://localhost:8000/fedavg/status  # operational v0
```

## 4. Full Tests

```powershell
# E2E tests (Terminal 2)
python test_system_v2.py  # ALL [OK] endpoints + edge

# Unit/integration
pytest tests/integration/test_hub_endpoints.py -v  # all pass
```

**Expected**: All [OK], 200s. Ingress errors OK (test data).

## 5. Demo Data/Experts
```powershell
python test_system.py  # edge → hub (embeddings/adapt)

# Check
curl http://localhost:8000/status  # embeddings>0, devices>0
curl -X POST /moe/create-expert    # creates expert!
curl /moe/status                   # num_experts>0
```

## 6. Troubleshooting
- **Port 8000 busy**: Kill `netstat -ano | findstr :8000` + taskkill /PID
- **Deps**: `python -m pip install --user -r requirements.txt --force-reinstall`
- **Logs**: Server console.
- **Reset**: curl -X POST http://localhost:8000/reset

## Architecture
```
Edge Nodes → POST /ingress_update (encrypt) → Hub (FedAvg/MoE)
Hub → /adapters/latest/* broadcast global adapter
Monitoring: /monitoring/* dashboards
```

**Complete!** Servers/endpoints ready. See TODO.md for changes.
