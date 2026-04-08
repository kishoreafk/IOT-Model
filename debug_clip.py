from edge_node.vision_agent import EdgeVisionNode
from PIL import Image
import numpy as np

edge = EdgeVisionNode(config_path='configs/vit_config.json')

# Create a random image
dummy_img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))

# Manually run what detect_novelty does
class_names = edge._get_default_labels()
print(f'Class names: {len(class_names)} classes')
print(f'First 5: {class_names[:5]}')

# Get CLIP pseudo-label
try:
    clip_label = edge._get_clip_zero_shot_label(dummy_img, class_names)
    print(f'CLIP predicted: {clip_label!r}')
except Exception as e:
    print(f'CLIP error: {e}')

# Now run detect_novelty
decision, scores, labels, pseudo = edge.detect_novelty(dummy_img, class_names)
print(f'\ndetect_novelty result:')
print(f'  Decision: {decision}')
print(f'  Pseudo label: {pseudo}')
print(f'  Scores count: {len(scores)}')
print(f'  Labels count: {len(labels)}')
if labels:
    print(f'  First label: {labels[0]}')
    print(f'  All labels: {labels}')
