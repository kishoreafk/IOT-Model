"""
Quick test to verify edge node connects to remote hub
"""
import asyncio
import sys
import os
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from edge_node.camera_node import LiveCameraNode

REMOTE_HUB_URL = "http://10.243.38.174:8000"

async def test_quick():
    print("="*60)
    print("QUICK EDGE NODE TEST")
    print("="*60)
    
    # Check camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERR] Camera not available")
        return
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        print(f"[OK] Camera read: {frame.shape}")
    else:
        print("[ERR] Could not read camera frame")
        return
    
    # Create camera node
    print(f"\n[*] Creating LiveCameraNode with hub: {REMOTE_HUB_URL}")
    
    camera_node = LiveCameraNode(
        device_id="test_edge_001",
        hub_url=REMOTE_HUB_URL,
        camera_index=0,
        inference_interval=2.0,
    )
    
    print("[OK] LiveCameraNode created")
    print(f"    Device ID: {camera_node.device_id}")
    print(f"    Hub URL: {camera_node.hub_url}")
    
    # Try to register with hub
    print("\n[*] Registering with hub...")
    await camera_node._register_with_hub()
    
    # Process one frame
    print("\n[*] Processing one frame...")
    from PIL import Image
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        await camera_node._process_frame(pil_img, frame)
        
        print("[OK] Frame processed successfully")
    
    print("\n" + "="*60)
    print("EDGE NODE CONNECTED TO REMOTE HUB!")
    print("="*60)

asyncio.run(test_quick())
