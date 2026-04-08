import asyncio
import httpx
import sys
import os
sys.path.insert(0, os.getcwd())

async def test_direct_ingress():
    """Test ingress_update directly."""
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
        device_id="direct_test_001",
        key_path="keys/encryption.key",
        private_key_path="keys/private_key.pem",
        public_key_path="keys/public_key.pem",
    )
    
    result = await transmitter.transmit(
        clip_emb,
        metadata={"trigger": "escalate_hub", "adapter_version": 0},
        sign_payload=True,
    )
    
    print(f"Result: {result}")
    print(f"Success: {result.get('success')}")
    print(f"Hub response: {result.get('hub_response')}")
    
    if result.get('hub_response'):
        task_id = result['hub_response'].get('task_id')
        print(f"Task ID: {task_id}")
        
        await asyncio.sleep(2)
        
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f'http://localhost:8000/tasks/{task_id}')
            print(f"Task status: {r.status_code}")
            print(f"Task data: {r.text}")

asyncio.run(test_direct_ingress())
