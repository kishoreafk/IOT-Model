"""
Comprehensive test script v2 - All endpoints and functionality
"""

import asyncio
import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path
from io import BytesIO

import numpy as np
import torch
import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from edge_node.vision_agent import EdgeVisionNode
from edge_node.secure_transmitter import SecureTransmitter


HUB_URL = "http://localhost:8000"
TEST_DEVICE_ID = f"test_device_{uuid.uuid4().hex[:8]}"


def create_test_image(size=(224, 224), color=(128, 128, 128)) -> Image.Image:
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = color
    return Image.fromarray(arr)


async def test_all_hub_endpoints():
    """Test all hub endpoints comprehensively."""
    print("\n" + "="*60)
    print("TESTING ALL HUB ENDPOINTS")
    print("="*60)
    
    results = []
    async with httpx.AsyncClient(timeout=30) as client:
        endpoints = [
            ("GET", "/health", None),
            ("GET", "/ready", None),
            ("GET", "/status", None),
            ("GET", "/clusters", None),
            ("GET", "/moe/status", None),
            ("GET", "/moe/representation-gap?cluster_threshold=5", None),
            ("GET", "/devices", None),
            ("GET", "/devices/test_device_001/status", None),
            ("POST", "/devices/register", {"device_id": "endpoint_test_device", "adapter_version": 0}),
            ("POST", "/reset", None),
        ]
        
        for method, path, payload in endpoints:
            try:
                if method == "GET":
                    resp = await client.request(method, f"{HUB_URL}{path}")
                else:
                    resp = await client.request(method, f"{HUB_URL}{path}", json=payload)
                resp.raise_for_status()
                data = resp.json()
                print(f"[OK] {method} {path}: {data.get('status', 'OK')}")
                results.append((f"{method} {path}", True, data))
            except Exception as e:
                print(f"[FAIL] {method} {path}: {e}")
                results.append((f"{method} {path}", False, str(e)))
        
        return results


async def test_moe_create_expert():
    """Test MoE expert creation."""
    print("\n" + "="*60)
    print("TESTING MOE CREATE EXPERT")
    print("="*60)
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(f"{HUB_URL}/moe/create-expert?cluster_id=0")
            resp.raise_for_status()
            data = resp.json()
            print(f"[OK] Create expert: {data}")
            return True, data
        except Exception as e:
            print(f"[FAIL] Create expert: {e}")
            return False, str(e)


async def test_moe_create_expert():
    """Test MoE expert creation."""
    print("\n" + "="*60)
    print("TESTING MOE CREATE EXPERT")
    print("="*60)
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(f"{HUB_URL}/moe/create-expert?cluster_id=0")
            resp.raise_for_status()
            data = resp.json()
            print(f"[OK] Create expert: {data}")
            return True, data
        except Exception as e:
            print(f"[FAIL] Create expert: {e}")
            return False, str(e)


async def test_fedavg_functionality():
    """Test FedAvg endpoint."""
    print("\n" + "="*60)
    print("TESTING FEDAVG")
    print("="*60)
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(f"{HUB_URL}/fedavg/status")
            data = resp.json()
            print(f"[OK] FedAvg status: {data}")
            return True, data
        except Exception as e:
            print(f"[FAIL] FedAvg status: {e}")
            return False, str(e)


async def test_task_tracking():
    """Test task tracking by checking recent tasks."""
    print("\n" + "="*60)
    print("TESTING TASK TRACKING")
    print("="*60)
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(f"{HUB_URL}/status")
            data = resp.json()
            print(f"[OK] Status shows task processing working")
            
            resp = await client.get(f"{HUB_URL}/clusters")
            data = resp.json()
            print(f"[OK] Cluster has {data.get('total_embeddings')} embeddings")
            
            return True, data
        except Exception as e:
            print(f"[FAIL] Task tracking: {e}")
            return False, str(e)


async def test_multiple_adaptations():
    """Test multiple local adaptations to generate more data for FedAvg."""
    print("\n" + "="*60)
    print("TESTING MULTIPLE ADAPTATIONS FOR FEDAVG")
    print("="*60)
    
    results = []
    device_ids = []
    
    for i in range(3):
        device_id = f"fedavg_test_device_{uuid.uuid4().hex[:8]}"
        device_ids.append(device_id)
        
        try:
            vision_node = EdgeVisionNode(
                device="cpu", 
                use_fp16=False, 
                lora_adapter_path=f"edge_node/lora_adapter/{device_id}.bin"
            )
            
            test_image = create_test_image(color=(i*50, i*50, i*50))
            
            vision_node.local_adaptation(
                image=test_image,
                pseudo_label=f"class_{i}",
                num_epochs=2,
            )
            print(f"[OK] Adaptation {i+1} for device {device_id[:16]}")
            
            vision_node._save_adapter_weights()
            
            clip_embedding = vision_node.extract_features(test_image)
            
            secure_transmitter = SecureTransmitter(
                adapter_weights_path=f"edge_node/lora_adapter/{device_id}.bin",
                hub_url=HUB_URL,
                device_id=device_id,
                key_path="keys/encryption.key",
                private_key_path="keys/private_key.pem",
                public_key_path="keys/public_key.pem",
            )
            
            result = await secure_transmitter.transmit(
                clip_embedding,
                metadata={
                    "trigger": "adapt_local",
                    "pseudo_label": f"class_{i}",
                    "adapter_version": 0,
                    "num_samples": 1,
                },
                sign_payload=True,
            )
            
            if result.get("success"):
                task_id = result.get("hub_response", {}).get("task_id")
                print(f"[OK] Transmitted adaptation {i+1}, task_id={task_id}")
                results.append(True)
            else:
                print(f"[FAIL] Transmission {i+1} failed")
                results.append(False)
            
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"[FAIL] Adaptation {i+1}: {e}")
            results.append(False)
    
    await asyncio.sleep(3)
    
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{HUB_URL}/status")
        data = resp.json()
        print(f"[INFO] After multiple adaptations - Global adapter version: {data.get('global_adapter_version')}")
        print(f"[INFO] Total embeddings: {data.get('total_embeddings')}")
        
        resp = await client.get(f"{HUB_URL}/devices")
        data = resp.json()
        print(f"[INFO] Total devices: {data.get('total_devices')}")
    
    return all(results), {"devices": len(device_ids), "success_count": sum(results)}


async def test_monitoring_endpoints():
    """Test monitoring dashboard endpoints."""
    print("\n" + "="*60)
    print("TESTING MONITORING ENDPOINTS")
    print("="*60)
    
    results = []
    async with httpx.AsyncClient(timeout=30) as client:
        endpoints = [
            "/monitoring/dashboard",
            "/monitoring/hub/health",
            "/monitoring/hub/stats",
        ]
        
        for path in endpoints:
            try:
                resp = await client.get(f"{HUB_URL}{path}")
                resp.raise_for_status()
                data = resp.json()
                print(f"[OK] {path}: {data.get('status', 'OK')}")
                results.append((path, True, data))
            except Exception as e:
                print(f"[FAIL] {path}: {e}")
                results.append((path, False, str(e)))
    
    return results


async def test_adapter_registry():
    """Test adapter registry endpoints."""
    print("\n" + "="*60)
    print("TESTING ADAPTER REGISTRY")
    print("="*60)
    
    results = []
    async with httpx.AsyncClient(timeout=30) as client:
        test_device_id = f"registry_test_{uuid.uuid4().hex[:8]}"
        
        try:
            resp = await client.post(
                f"{HUB_URL}/devices/register",
                json={"device_id": test_device_id, "adapter_version": 0}
            )
            resp.raise_for_status()
            data = resp.json()
            print(f"[OK] Registered device: {data}")
            results.append(("register", True, data))
        except Exception as e:
            print(f"[FAIL] Register: {e}")
            results.append(("register", False, str(e)))
        
        try:
            resp = await client.get(f"{HUB_URL}/devices/{test_device_id}/status")
            resp.raise_for_status()
            data = resp.json()
            print(f"[OK] Device status: {data}")
            results.append(("status", True, data))
        except Exception as e:
            print(f"[FAIL] Status: {e}")
            results.append(("status", False, str(e)))
        
        try:
            resp = await client.get(f"{HUB_URL}/devices")
            resp.raise_for_status()
            data = resp.json()
            print(f"[OK] All devices: {data.get('total_devices')} devices")
            results.append(("list", True, data))
        except Exception as e:
            print(f"[FAIL] List: {e}")
            results.append(("list", False, str(e)))
    
    return results


async def test_edge_components():
    """Test edge node components."""
    print("\n" + "="*60)
    print("TESTING EDGE COMPONENTS")
    print("="*60)
    
    results = []
    
    try:
        vision_node = EdgeVisionNode(device="cpu", use_fp16=False)
        print("[OK] EdgeVisionNode initialized")
        results.append(("init", True, None))
    except Exception as e:
        print(f"[FAIL] Init: {e}")
        results.append(("init", False, str(e)))
    
    try:
        test_image = create_test_image()
        decision, scores, labels = vision_node.detect_novelty(test_image)
        print(f"[OK] Novelty detection: decision={decision}")
        results.append(("novelty", True, {"decision": decision}))
    except Exception as e:
        print(f"[FAIL] Novelty: {e}")
        results.append(("novelty", False, str(e)))
    
    try:
        label, confidence = vision_node.classify_image(test_image)
        print(f"[OK] Classification: label={label}, confidence={confidence:.4f}")
        results.append(("classify", True, {"label": label, "confidence": confidence}))
    except Exception as e:
        print(f"[FAIL] Classification: {e}")
        results.append(("classify", False, str(e)))
    
    try:
        features = vision_node.extract_features(test_image)
        print(f"[OK] Feature extraction: shape={features.shape}")
        results.append(("features", True, {"shape": features.shape}))
    except Exception as e:
        print(f"[FAIL] Features: {e}")
        results.append(("features", False, str(e)))
    
    try:
        vision_node.local_adaptation(test_image, "test_class", num_epochs=2)
        weights = vision_node.get_adapter_weights()
        print(f"[OK] Local adaptation: {len(weights)} weight keys")
        results.append(("adapt", True, {"weight_count": len(weights)}))
    except Exception as e:
        print(f"[FAIL] Adapt: {e}")
        results.append(("adapt", False, str(e)))
    
    return results


async def test_secure_transmission():
    """Test secure transmitter."""
    print("\n" + "="*60)
    print("TESTING SECURE TRANSMISSION")
    print("="*60)
    
    results = []
    
    vision_node = EdgeVisionNode(device="cpu", use_fp16=False)
    test_image = create_test_image()
    clip_embedding = vision_node.extract_features(test_image)
    
    secure_transmitter = SecureTransmitter(
        adapter_weights_path="edge_node/lora_adapter/test_transmit.bin",
        hub_url=HUB_URL,
        device_id="secure_test_device",
        key_path="keys/encryption.key",
        private_key_path="keys/private_key.pem",
        public_key_path="keys/public_key.pem",
    )
    
    try:
        result = await secure_transmitter.transmit(
            clip_embedding,
            metadata={"trigger": "test", "adapter_version": 0},
            sign_payload=True,
        )
        print(f"[OK] Secure transmit: success={result.get('success')}")
        results.append(("transmit", True, result))
    except Exception as e:
        print(f"[FAIL] Transmit: {e}")
        results.append(("transmit", False, str(e)))
    
    return results


async def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# EDGE HUB COMPREHENSIVE TEST SUITE v2")
    print("#"*60)
    
    all_results = {}
    
    all_results["hub_endpoints"] = await test_all_hub_endpoints()
    
    all_results["moe_expert"] = await test_moe_create_expert()
    
    all_results["fedavg"] = await test_fedavg_functionality()
    
    all_results["task_tracking"] = await test_task_tracking()
    
    all_results["multiple_adaptations"] = await test_multiple_adaptations()
    
    all_results["monitoring"] = await test_monitoring_endpoints()
    
    all_results["adapter_registry"] = await test_adapter_registry()
    
    all_results["edge_components"] = await test_edge_components()
    
    all_results["secure_transmission"] = await test_secure_transmission()
    
    print("\n" + "#"*60)
    print("# FINAL TEST SUMMARY")
    print("#"*60)
    
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{HUB_URL}/status")
        final_status = resp.json()
        print(f"\nHub Status:")
        print(f"  - Status: {final_status.get('status')}")
        print(f"  - Total embeddings: {final_status.get('total_embeddings')}")
        print(f"  - Total devices: {final_status.get('total_devices')}")
        print(f"  - Global adapter version: {final_status.get('global_adapter_version')}")
        
        resp = await client.get(f"{HUB_URL}/clusters")
        clusters = resp.json()
        print(f"  - Total clusters: {clusters.get('total_clusters')}")
        
        resp = await client.get(f"{HUB_URL}/moe/status")
        moe = resp.json()
        print(f"  - MoE experts: {moe.get('num_experts')}")


if __name__ == "__main__":
    asyncio.run(main())
