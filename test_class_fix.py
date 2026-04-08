from edge_node.vision_agent import EdgeVisionNode
from PIL import Image
import numpy as np

edge = EdgeVisionNode(config_path='configs/vit_config.json')

# Check class alignment
class_names = edge._get_default_labels()
print(f"Edge classes: {len(class_names)} total")
print(f"  Class 0: {class_names[0]}")
print(f"  Class 1: {class_names[1]}")
print(f"  Class 24: {class_names[24]}")
print(f"  Class 49: {class_names[49]}")

print(f"\nBase ViT model:")
print(f"  num_labels: {edge.custom_vit.config.num_labels}")

# Test inference
dummy_img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
decision, scores, labels, pseudo = edge.run_inference(dummy_img)

print(f"\nInference result:")
print(f"  Decision: {decision}")
print(f"  Top 3 labels: {labels[:3]}")
print(f"  Top 3 scores: {[f'{s:.4f}' for s in scores[:3]]}")
print(f"\nExpected: Labels should be actual ImageNet classes like goldfish, salamander, etc.")
