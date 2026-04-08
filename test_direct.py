import asyncio
import httpx

async def test_direct():
    """Test embedding addition directly."""
    async with httpx.AsyncClient(timeout=30) as client:
        print("1. Initial status:")
        r = await client.get('http://localhost:8000/status')
        print(r.json())
        
        print("\n2. Checking clusters:")
        r = await client.get('http://localhost:8000/clusters')
        print(r.json())
        
        print("\n3. Sending a test escalation (should add embedding):")
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
            device_id="direct_test_device",
            key_path="keys/encryption.key",
            private_key_path="keys/private_key.pem",
            public_key_path="keys/public_key.pem",
        )
        
        result = await transmitter.transmit(
            clip_emb,
            metadata={"trigger": "escalate_hub", "adapter_version": 0},
            sign_payload=True,
        )
        print(f"Transmission result: {result}")
        
        task_id = result.get("hub_response", {}).get("task_id")
        print(f"Task ID: {task_id}")
        
        print("\n4. Waiting 3 seconds...")
        await asyncio.sleep(3)
        
        print("\n5. Checking status after wait:")
        r = await client.get('http://localhost:8000/status')
        print(r.json())
        
        print("\n6. Checking clusters:")
        r = await client.get('http://localhost:8000/clusters')
        print(r.json())
        
        if task_id:
            print("\n7. Checking task status:")
            r = await client.get(f'http://localhost:8000/tasks/{task_id}')
            print(r.json())

asyncio.run(test_direct())
