import asyncio
import httpx

async def test_task_debug():
    async with httpx.AsyncClient(timeout=30) as client:
        # First create task via API call
        r = await client.get('http://localhost:8000/health')
        print("Health:", r.json())
        
        # Create a new task via transmission
        import sys, os
        sys.path.insert(0, os.getcwd())
        
        from edge_node.vision_agent import EdgeVisionNode
        from edge_node.secure_transmitter import SecureTransmitter
        from PIL import Image
        import numpy as np
        
        vision_node = EdgeVisionNode(device="cpu", use_fp16=False)
        img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        clip_emb = vision_node.extract_features(img)
        
        transmitter = SecureTransmitter(
            adapter_weights_path="edge_node/lora_adapter/test.bin",
            hub_url="http://localhost:8000",
            device_id="debug_test_device",
            key_path="keys/encryption.key",
            private_key_path="keys/private_key.pem",
            public_key_path="keys/public_key.pem",
        )
        
        result = await transmitter.transmit(
            clip_emb,
            metadata={"trigger": "escalate_hub", "adapter_version": 0},
            sign_payload=True,
        )
        
        task_id = result.get("hub_response", {}).get("task_id")
        print(f"\nNew task created: {task_id}")
        
        await asyncio.sleep(2)
        
        # Try to get task info - with explicit error handling
        try:
            r = await client.get(f'http://localhost:8000/tasks/{task_id}')
            print(f"Status: {r.status_code}")
            print(f"Response: {r.text}")
        except Exception as e:
            print(f"Error: {e}")

asyncio.run(test_task_debug())
