import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from edge_node.vision_agent import EdgeVisionNode
from edge_node.adapter_sync import AdapterSyncClient

def test_hub_adapter():
    # Use config from codebase to match standard behavior
    node = EdgeVisionNode(device='cpu', use_fp16=False)
    
    # Create fake adapter metadata and weights (as if coming from hub)
    # The default class list length is 50, let's create a layer exactly matching the class size
    class_names = node._get_default_labels()
    num_classes = len(class_names)
    proj_layer = nn.Linear(512, num_classes)
    
    # Set weights to be deterministic so we can check output
    nn.init.zeros_(proj_layer.weight)
    nn.init.zeros_(proj_layer.bias)
    
    # Let's heavily weight 'centipede' which is at index 10 in the default list
    target_idx = 10
    proj_layer.bias.data[target_idx] = 100.0
    
    state_dict = proj_layer.state_dict()
    adapter_data = {
        "state_dict": state_dict,
        "class_names": class_names,
        "adapter_type": "projection_layer",
        "num_classes": num_classes
    }
    
    # Serialize it as bytes
    import io
    buf = io.BytesIO()
    torch.save(adapter_data, buf)
    adapter_bytes = buf.getvalue()
    
    # Init sync client
    client = AdapterSyncClient(
        device_id="test_device",
        hub_url="http://dummy",
        local_adapter_path="dummy.bin",
        vision_node=node
    )
    
    print("Testing hot swap...")
    # This should call our newly added code in adapter_sync.py
    client._hot_swap_adapter(adapter_bytes, 1)
    
    assert hasattr(node, 'hub_projection')
    assert isinstance(node.hub_projection, nn.Linear)
    print("✓ Projection layer cached as nn.Linear")
    
    # Test inference using dummy image
    img = Image.new('RGB', (224, 224), color='red')
    
    decision, scores, labels, pseudo_label = node.run_inference(img)
    print(f"Decision: {decision}")
    print(f"Top label: {labels[0]} with confidence {scores[0]}")
    
    assert labels[0] == class_names[target_idx]
    assert scores[0] > 0.99
    
    print("✓ Hub projection inference successful (target class detected as forced by weights)")
    print("ALL TESTS PASSED.")

if __name__ == "__main__":
    test_hub_adapter()
