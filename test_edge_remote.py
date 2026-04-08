"""
Test script for edge node with remote hub connectivity
Uses remote hub at 10.243.38.174:8000
"""

import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from edge_node.vision_agent import EdgeVisionNode
from edge_node.secure_transmitter import SecureTransmitter
from edge_node.camera_node import LiveCameraNode


REMOTE_HUB_URL = "http://10.243.38.174:8000"

async def test_edge_with_remote_hub():
    """Test edge node with remote hub."""
    print("=" * 60)
    print("EDGE NODE TEST WITH REMOTE HUB")
    print("=" * 60)
    print(f"Hub URL: {REMOTE_HUB_URL}")
    print()
    
    # Test 1: Initialize vision node
    print("[1] Initializing EdgeVisionNode...")
    try:
        vision_node = EdgeVisionNode(device="cpu", use_fp16=False)
        print("    [OK] EdgeVisionNode initialized")
    except Exception as e:
        print(f"    [ERR] {e}")
        return
    
    # Test 2: Novelty detection
    print("\n[2] Testing novelty detection...")
    try:
        from PIL import Image
        import numpy as np
        
        # Create test image
        test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        decision, scores, labels = vision_node.detect_novelty(
            test_img, 
            candidate_labels=["car", "truck", "person", "dog", "cat"]
        )
        
        print(f"    Decision: {decision}")
        print(f"    Top label: {labels[0]} ({scores[0]:.2%})")
        print("    [OK] Novelty detection working")
    except Exception as e:
        print(f"    [ERR] {e}")
    
    # Test 3: Secure transmission to remote hub
    print("\n[3] Testing secure transmission to remote hub...")
    try:
        from PIL import Image
        import numpy as np
        
        test_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        clip_emb = vision_node.extract_features(test_img)
        
        transmitter = SecureTransmitter(
            adapter_weights_path="edge_node/lora_adapter/test_adapter.bin",
            hub_url=REMOTE_HUB_URL,
            device_id="edge_test_001",
            key_path="keys/encryption.key",
            private_key_path="keys/private_key.pem",
            public_key_path="keys/public_key.pem",
        )
        
        result = await transmitter.transmit(
            clip_emb,
            metadata={
                "trigger": "escalate_hub",
                "adapter_version": 0,
            },
            sign_payload=True,
        )
        
        print(f"    Success: {result.get('success')}")
        
        if result.get('hub_response'):
            task_id = result['hub_response'].get('task_id')
            print(f"    Task ID: {task_id}")
            
            # Wait for task to complete
            await asyncio.sleep(2)
            
            import httpx
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(f"{REMOTE_HUB_URL}/tasks/{task_id}")
                if r.status_code == 200:
                    task_data = r.json()
                    print(f"    Task status: {task_data.get('status')}")
                    print(f"    Task result: {task_data.get('result')}")
                    print("    [OK] Task tracking working")
                else:
                    print(f"    [WARN] Task status: {r.status_code}")
        
    except Exception as e:
        print(f"    [ERR] {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Check hub status
    print("\n[4] Checking hub status...")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{REMOTE_HUB_URL}/status")
            data = r.json()
            print(f"    Status: {data.get('status')}")
            print(f"    Embeddings: {data.get('total_embeddings')}")
            print(f"    Devices: {data.get('total_devices')}")
            print("    [OK] Hub is operational")
    except Exception as e:
        print(f"    [ERR] {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("EDGE NODE TEST COMPLETE")
    print("=" * 60)
    print("\nTo start live camera inference, run:")
    print("  python -m edge_node.camera_node --hub-url http://10.243.38.174:8000")
    print("\nOr with options:")
    print("  python -m edge_node.camera_node --hub-url http://10.243.38.174:8000 --device-id my_device")


def test_camera_available():
    """Check if camera is available."""
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.release()
        return True
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Edge Node with Remote Hub")
    parser.add_argument("--hub-url", default=REMOTE_HUB_URL, help="Hub URL")
    parser.add_argument("--device-id", default=None, help="Device ID")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index")
    parser.add_argument("--inference-interval", type=float, default=1.0, help="Inference interval")
    args = parser.parse_args()
    
    print(f"\n[*] Using hub: {args.hub_url}")
    
    # Check if camera is available
    has_camera = test_camera_available()
    print(f"[*] Camera available: {has_camera}")
    
    if has_camera:
        print("\n[*] Starting live camera node...")
        print("[*] Press 'q' to quit")
        
        camera_node = LiveCameraNode(
            device_id=args.device_id,
            hub_url=args.hub_url,
            camera_index=args.camera_index,
            inference_interval=args.inference_interval,
        )
        
        camera_node.start()
    else:
        print("\n[*] No camera detected. Running test mode...")
        asyncio.run(test_edge_with_remote_hub())
