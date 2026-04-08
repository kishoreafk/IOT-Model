"""
Comprehensive test script for Edge Hub Adaptive Learning System
Tests:
1. local_adaptation method
2. Edge-to-hub weight transfer flow
3. All hub endpoints
4. Integration between edge and hub
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
    """Create a test image."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = color
    return Image.fromarray(arr)


async def test_hub_endpoints():
    """Test all hub endpoints."""
    print("\n" + "="*60)
    print("TESTING HUB ENDPOINTS")
    print("="*60)
    
    async with httpx.AsyncClient(timeout=30) as client:
        endpoints = [
            ("GET", "/health"),
            ("GET", "/ready"),
            ("GET", "/status"),
            ("GET", "/clusters"),
            ("GET", "/moe/status"),
            ("GET", "/devices"),
            ("GET", "/adapters/latest/version"),
        ]
        
        results = []
        for method, path in endpoints:
            try:
                if method == "GET":
                    resp = await client.request(method, f"{HUB_URL}{path}")
                    resp.raise_for_status()
                    data = resp.json()
                    print(f"[OK] {path}: {data.get('status', 'OK')}")
                    results.append((path, True, data))
            except Exception as e:
                print(f"[ERR] {path}: {e}")
                results.append((path, False, str(e)))
        
        return results


async def test_device_registration():
    """Test device registration with hub."""
    print("\n" + "="*60)
    print("TESTING DEVICE REGISTRATION")
    print("="*60)
    
    async with httpx.AsyncClient(timeout=30) as client:
        payload = {
            "device_id": TEST_DEVICE_ID,
            "public_key_path": "keys/public_key.pem",
            "adapter_version": 0,
        }
        try:
            resp = await client.post(f"{HUB_URL}/devices/register", json=payload)
            resp.raise_for_status()
            data = resp.json()
            print(f"[OK] Device registered: {data}")
            return True, data
        except Exception as e:
            print(f"[FAIL] Registration failed: {e}")
            return False, str(e)


async def test_local_adaptation():
    """Test local_adaptation method."""
    print("\n" + "="*60)
    print("TESTING LOCAL ADAPTATION")
    print("="*60)
    
    try:
        vision_node = EdgeVisionNode(device="cpu", use_fp16=False, lora_adapter_path="edge_node/lora_adapter/test_adapter.bin")
        
        test_image = create_test_image()
        print(f"[OK] Test image created: {test_image.size}")
        
        vision_node.local_adaptation(
            image=test_image,
            pseudo_label="test_class",
            num_epochs=2,
        )
        print("[OK] local_adaptation completed")
        
        label, confidence = vision_node.classify_image(test_image)
        print(f"[OK] Classification result: label={label}, confidence={confidence:.4f}")
        
        weights = vision_node.get_adapter_weights()
        if weights:
            print(f"[OK] Adapter weights retrieved: {len(weights)} keys")
        else:
            print("⚠ No adapter weights (first adaptation)")
        
        return True, {"label": label, "confidence": confidence}
        
    except Exception as e:
        print(f"[FAIL] Local adaptation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_detect_novelty():
    """Test detect_novelty method."""
    print("\n" + "="*60)
    print("TESTING NOVELTY DETECTION")
    print("="*60)
    
    try:
        vision_node = EdgeVisionNode(device="cpu", use_fp16=False)
        print("[OK] EdgeVisionNode initialized")
        
        test_image = create_test_image()
        
        candidate_labels = ["car", "truck", "person", "dog", "cat"]
        decision, scores, labels = vision_node.detect_novelty(
            test_image, 
            candidate_labels=candidate_labels
        )
        
        print(f"[OK] Decision: {decision}")
        print(f"[OK] Top label: {labels[0] if labels else 'N/A'}")
        print(f"[OK] Top score: {scores[0] if scores else 0:.4f}")
        print(f"[OK] All scores: {dict(zip(labels[:3], [f'{s:.3f}' for s in scores[:3]]))}")
        
        return True, {"decision": decision, "labels": labels, "scores": scores}
        
    except Exception as e:
        print(f"[FAIL] Novelty detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


async def test_edge_to_hub_weight_transfer():
    """Test complete edge-to-hub weight transfer flow."""
    print("\n" + "="*60)
    print("TESTING EDGE-TO-HUB WEIGHT TRANSFER")
    print("="*60)
    
    try:
        vision_node = EdgeVisionNode(device="cpu", use_fp16=False, lora_adapter_path="edge_node/lora_adapter/test_adapter.bin")
        
        test_image = create_test_image()
        
        vision_node.local_adaptation(
            image=test_image,
            pseudo_label="test_class",
            num_epochs=2,
        )
        print("[OK] Local adaptation done")
        
        vision_node._save_adapter_weights()
        print("[OK] Adapter weights saved")
        
        clip_embedding = vision_node.extract_features(test_image)
        embedding_list = clip_embedding.cpu().numpy().tolist()[0]
        print(f"[OK] CLIP embedding extracted: {len(embedding_list)} dimensions")
        
        weights = vision_node.get_adapter_weights()
        if not weights:
            print("[FAIL] No adapter weights available")
            return False, "No adapter weights"
        
        adapter_bytes = json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v.tolist() for k, v in weights.items()}).encode()
        adapter_b64 = base64.b64encode(adapter_bytes).decode()
        
        secure_transmitter = SecureTransmitter(
            adapter_weights_path="edge_node/lora_adapter/test_adapter.bin",
            hub_url=HUB_URL,
            device_id=TEST_DEVICE_ID,
            key_path="keys/encryption.key",
            private_key_path="keys/private_key.pem",
            public_key_path="keys/public_key.pem",
        )
        
        result = await secure_transmitter.transmit(
            clip_embedding,
            metadata={
                "trigger": "adapt_local",
                "pseudo_label": "test_class",
                "adapter_version": 0,
            },
            sign_payload=True,
        )
        
        print(f"[OK] Transmission result: {result.get('success')}")
        
        if result.get("success"):
            hub_response = result.get("hub_response", {})
            task_id = hub_response.get("task_id")
            print(f"[OK] Task ID: {task_id}")
            
            await asyncio.sleep(2)
            
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(f"{HUB_URL}/tasks/{task_id}")
                if resp.status_code == 200:
                    task_data = resp.json()
                    print(f"[OK] Task status: {task_data.get('status')}")
                    print(f"[OK] Task result: {task_data.get('result')}")
                    
                    status_resp = await client.get(f"{HUB_URL}/status")
                    status = status_resp.json()
                    print(f"[OK] Global adapter version: {status.get('global_adapter_version')}")
        
        return True, result
        
    except Exception as e:
        print(f"[FAIL] Weight transfer failed: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


async def test_escalate_to_hub():
    """Test escalation to hub with CLIP embedding."""
    print("\n" + "="*60)
    print("TESTING ESCALATE TO HUB")
    print("="*60)
    
    try:
        vision_node = EdgeVisionNode(device="cpu", use_fp16=False)
        
        test_image = create_test_image()
        
        clip_embedding = vision_node.extract_features(test_image)
        print(f"[OK] CLIP embedding extracted")
        
        secure_transmitter = SecureTransmitter(
            adapter_weights_path="edge_node/lora_adapter/test_adapter.bin",
            hub_url=HUB_URL,
            device_id=TEST_DEVICE_ID,
            key_path="keys/encryption.key",
            private_key_path="keys/private_key.pem",
            public_key_path="keys/public_key.pem",
        )
        
        result = await secure_transmitter.transmit(
            clip_embedding,
            metadata={
                "trigger": "escalate_hub",
                "adapter_version": 0,
            },
            sign_payload=True,
        )
        
        print(f"[OK] Transmission result: {result.get('success')}")
        
        if result.get("success"):
            hub_response = result.get("hub_response", {})
            task_id = hub_response.get("task_id")
            print(f"[OK] Task ID: {task_id}")
            
            await asyncio.sleep(3)
            
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(f"{HUB_URL}/tasks/{task_id}")
                if resp.status_code == 200:
                    task_data = resp.json()
                    print(f"[OK] Task status: {task_data.get('status')}")
                    print(f"[OK] Task result: {task_data.get('result')}")
        
        return True, result
        
    except Exception as e:
        print(f"[FAIL] Escalation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


async def test_adapter_download():
    """Test adapter download from hub."""
    print("\n" + "="*60)
    print("TESTING ADAPTER DOWNLOAD")
    print("="*60)
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(f"{HUB_URL}/adapters/latest/version")
            resp.raise_for_status()
            version_data = resp.json()
            print(f"[OK] Latest adapter version: {version_data}")
            
            if version_data.get("version", 0) > 0:
                resp = await client.get(f"{HUB_URL}/adapters/latest/download")
                if resp.status_code == 200:
                    print(f"[OK] Adapter download available")
                else:
                    print(f"⚠ Adapter download status: {resp.status_code}")
            else:
                print("⚠ No adapter version available yet")
            
            return True, version_data
        except Exception as e:
            print(f"[FAIL] Adapter download test failed: {e}")
            return False, str(e)


async def test_moe_endpoints():
    """Test MoE-related endpoints."""
    print("\n" + "="*60)
    print("TESTING MOE ENDPOINTS")
    print("="*60)
    
    async with httpx.AsyncClient(timeout=30) as client:
        endpoints = [
            ("/moe/status"),
            ("/moe/representation-gap?cluster_threshold=5"),
        ]
        
        results = []
        for path in endpoints:
            try:
                resp = await client.get(f"{HUB_URL}{path}")
                resp.raise_for_status()
                data = resp.json()
                print(f"[OK] {path}: {data.get('num_experts', 'OK')}")
                results.append((path, True, data))
            except Exception as e:
                print(f"[ERR] {path}: {e}")
                results.append((path, False, str(e)))
        
        return results


async def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# EDGE HUB ADAPTIVE LEARNING SYSTEM - COMPREHENSIVE TEST")
    print("#"*60)
    
    all_results = {}
    
    all_results["hub_endpoints"] = await test_hub_endpoints()
    
    all_results["device_registration"] = await test_device_registration()
    
    all_results["novelty_detection"] = test_detect_novelty()
    
    all_results["local_adaptation"] = await test_local_adaptation()
    
    all_results["edge_to_hub_transfer"] = await test_edge_to_hub_weight_transfer()
    
    all_results["escalate_hub"] = await test_escalate_to_hub()
    
    all_results["adapter_download"] = await test_adapter_download()
    
    all_results["moe_endpoints"] = await test_moe_endpoints()
    
    print("\n" + "#"*60)
    print("# TEST SUMMARY")
    print("#"*60)
    
    test_results = [
        ("Hub Endpoints", all_results["hub_endpoints"]),
        ("Device Registration", all_results["device_registration"]),
        ("Novelty Detection", all_results["novelty_detection"]),
        ("Local Adaptation", all_results["local_adaptation"]),
        ("Edge-to-Hub Transfer", all_results["edge_to_hub_transfer"]),
        ("Escalate to Hub", all_results["escalate_hub"]),
        ("Adapter Download", all_results["adapter_download"]),
        ("MoE Endpoints", all_results["moe_endpoints"]),
    ]
    
    for name, result in test_results:
        status = "[OK] PASS" if result[0] else "[FAIL] FAIL"
        print(f"{status}: {name}")
    
    async with httpx.AsyncClient(timeout=30) as client:
        status_resp = await client.get(f"{HUB_URL}/status")
        final_status = status_resp.json()
        print(f"\nFinal Hub Status:")
        print(f"  - Total embeddings: {final_status.get('total_embeddings')}")
        print(f"  - Total devices: {final_status.get('total_devices')}")
        print(f"  - Global adapter version: {final_status.get('global_adapter_version')}")


if __name__ == "__main__":
    asyncio.run(main())
