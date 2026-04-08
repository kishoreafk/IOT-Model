"""
Direct camera node test - captures errors
"""
import sys
import os
from pathlib import Path
import asyncio
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from edge_node.camera_node import LiveCameraNode


async def run_camera():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub-url", default="http://10.243.38.174:8000")
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--inference-interval", type=float, default=1.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--no-hub", action="store_true", help="Run without hub connection (local only)")
    args = parser.parse_args()
    
    print("="*60)
    print("EDGE CAMERA NODE STARTING")
    print("="*60)
    print(f"Hub URL: {args.hub_url if not args.no_hub else 'DISABLED'}")
    print(f"Camera index: {args.camera_index}")
    print(f"Device: {args.device}")
    print(f"FP16: {args.use_fp16}")
    print(f"Inference interval: {args.inference_interval}s")
    print("="*60)
    print("Loading models...")
    
    try:
        camera = LiveCameraNode(
            device_id=args.device_id,
            hub_url=args.hub_url,
            camera_index=args.camera_index,
            inference_interval=args.inference_interval,
            device=args.device,
            use_fp16=args.use_fp16,
        )
        
        print(f"Device ID: {camera.device_id}")
        print("Models loaded successfully!")
        print("="*60)
        
        if args.no_hub:
            camera._skip_hub = True
        
        await camera._run()
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(run_camera())
    except KeyboardInterrupt:
        print("\nExiting...")
