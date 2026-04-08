"""
Test script for remote hub server at 10.243.38.174
"""

import asyncio
import httpx
import sys
import os
sys.path.insert(0, os.getcwd())

# Remote hub server IP
REMOTE_HUB_URL = "http://10.243.38.174:8000"

async def test_remote_hub():
    """Test all endpoints against remote hub."""
    print(f"Testing remote hub at: {REMOTE_HUB_URL}")
    print("=" * 60)
    
    results = []
    
    async with httpx.AsyncClient(timeout=30) as client:
        # Test health
        try:
            r = await client.get(f"{REMOTE_HUB_URL}/health")
            print(f"[OK] /health: {r.json()}")
            results.append(("health", True))
        except Exception as e:
            print(f"[FAIL] /health: {e}")
            results.append(("health", False))
        
        # Test ready
        try:
            r = await client.get(f"{REMOTE_HUB_URL}/ready")
            print(f"[OK] /ready: {r.json()}")
            results.append(("ready", True))
        except Exception as e:
            print(f"[FAIL] /ready: {e}")
            results.append(("ready", False))
        
        # Test status
        try:
            r = await client.get(f"{REMOTE_HUB_URL}/status")
            print(f"[OK] /status: {r.json()}")
            results.append(("status", True))
        except Exception as e:
            print(f"[FAIL] /status: {e}")
            results.append(("status", False))
        
        # Test clusters
        try:
            r = await client.get(f"{REMOTE_HUB_URL}/clusters")
            print(f"[OK] /clusters: {r.json()}")
            results.append(("clusters", True))
        except Exception as e:
            print(f"[FAIL] /clusters: {e}")
            results.append(("clusters", False))
        
        # Test devices
        try:
            r = await client.get(f"{REMOTE_HUB_URL}/devices")
            print(f"[OK] /devices: {r.json()}")
            results.append(("devices", True))
        except Exception as e:
            print(f"[FAIL] /devices: {e}")
            results.append(("devices", False))
        
        # Test MoE status
        try:
            r = await client.get(f"{REMOTE_HUB_URL}/moe/status")
            print(f"[OK] /moe/status: {r.json()}")
            results.append(("moe_status", True))
        except Exception as e:
            print(f"[FAIL] /moe/status: {e}")
            results.append(("moe_status", False))
        
        # Test monitoring
        try:
            r = await client.get(f"{REMOTE_HUB_URL}/monitoring/hub/health")
            print(f"[OK] /monitoring/hub/health: {r.json()}")
            results.append(("monitoring_health", True))
        except Exception as e:
            print(f"[FAIL] /monitoring/hub/health: {e}")
            results.append(("monitoring_health", False))
        
        # Register a test device
        try:
            device_id = f"test_device_{os.urandom(4).hex()}"
            r = await client.post(
                f"{REMOTE_HUB_URL}/devices/register",
                json={"device_id": device_id, "adapter_version": 0}
            )
            print(f"[OK] Device registered: {r.json()}")
            results.append(("register", True))
        except Exception as e:
            print(f"[FAIL] Register: {e}")
            results.append(("register", False))
    
    # Test task tracking by sending an update
    print("\n" + "=" * 60)
    print("TESTING TASK TRACKING (Ingress)")
    print("=" * 60)
    
    try:
        from edge_node.vision_agent import EdgeVisionNode
        from edge_node.secure_transmitter import SecureTransmitter
        from PIL import Image
        import numpy as np
        
        vision_node = EdgeVisionNode(device="cpu", use_fp16=False)
        img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        clip_emb = vision_node.extract_features(img)
        
        transmitter = SecureTransmitter(
            adapter_weights_path="edge_node/lora_adapter/test.bin",
            hub_url=REMOTE_HUB_URL,
            device_id="remote_test_device",
            key_path="keys/encryption.key",
            private_key_path="keys/private_key.pem",
            public_key_path="keys/public_key.pem",
        )
        
        result = await transmitter.transmit(
            clip_emb,
            metadata={"trigger": "escalate_hub", "adapter_version": 0},
            sign_payload=True,
        )
        
        print(f"Transmission: {result.get('success')}")
        
        if result.get('hub_response'):
            task_id = result['hub_response'].get('task_id')
            print(f"Task ID: {task_id}")
            
            await asyncio.sleep(2)
            
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(f"{REMOTE_HUB_URL}/tasks/{task_id}")
                print(f"Task status: {r.status_code}")
                if r.status_code == 200:
                    print(f"Task result: {r.json()}")
                    results.append(("task_tracking", True))
                else:
                    print(f"Task error: {r.text}")
                    results.append(("task_tracking", False))
    except Exception as e:
        print(f"[FAIL] Task tracking: {e}")
        results.append(("task_tracking", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for name, ok in results:
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}")

asyncio.run(test_remote_hub())
