import asyncio
import httpx
import sys
sys.path.insert(0, '.')

from central_hub.task_tracker import TaskTracker

async def test_task_tracker():
    tt = TaskTracker()
    
    # Create a task
    tt.create("test_task_001", "test_type")
    print("Created task:", tt.get("test_task_001"))
    
    # Complete it
    tt.complete("test_task_001", {"result": "success"})
    print("Completed task:", tt.get("test_task_001"))
    
    # Now test via API
    async with httpx.AsyncClient(timeout=30) as client:
        # First check status
        r = await client.get('http://localhost:8000/status')
        print("\nStatus:", r.json())
        
        # Create new task via ingress
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
            device_id="task_test_device",
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
        print(f"\nTask ID: {task_id}")
        
        await asyncio.sleep(2)
        
        # Check task via API
        r = await client.get(f'http://localhost:8000/tasks/{task_id}')
        print(f"Task API status: {r.status_code}")
        if r.status_code == 200:
            print(f"Task data: {r.json()}")
        else:
            print(f"Task error: {r.text}")

asyncio.run(test_task_tracker())
