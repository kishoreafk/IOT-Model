import asyncio
from edge_node.secure_transmitter import SecureTransmitter
from edge_node.vision_agent import EdgeVisionNode
from PIL import Image
import numpy as np

async def test():
    print("Creating vision node...")
    vn = EdgeVisionNode()
    
    st = SecureTransmitter(
        adapter_weights_path='edge_node/lora_adapter/test.bin',
        hub_url='http://10.243.38.174:8000',
        device_id='test_sign'
    )
    
    # Create dummy image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # Get embedding
    emb = vn.extract_features(img)
    print(f"Embedding shape: {emb.shape}")
    
    # Transmit
    print("Sending to hub...")
    result = await st.transmit(emb, metadata={'trigger': 'escalate_hub'})
    print(f"Result: {result}")

asyncio.run(test())
