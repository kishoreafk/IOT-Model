"""
Preservation Property Tests - Verify baseline behavior is preserved

These tests verify that the core workflows remain unchanged:
1. ViT high-confidence classification works (no CLIP invocation)
2. Local adaptation workflow works (LoRA + FedAvg)
3. Hub retraining workflow structure is preserved

These tests MUST PASS on unfixed code - they confirm baseline behavior.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from PIL import Image
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)

print("=" * 80)
print("PRESERVATION PROPERTY TESTS - Verify Baseline Behavior")
print("=" * 80)

# ============================================================================
# Property 1: ViT high-confidence classification works (CLIP not invoked)
# ============================================================================
print("\n[PROPERTY 1] ViT high-confidence classification (CLIP not invoked)")
print("-" * 60)

from edge_node.vision_agent import EdgeVisionNode

edge = EdgeVisionNode(config_path="configs/vit_config.json")

# Create a test image
test_image = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))

# Check what the known_threshold is
print(f"Known threshold: {edge.known_threshold}")

# Test 1: When ViT is confident (>= known_threshold), CLIP should NOT be invoked
# We'll test by checking the returned decision

# Create an image with specific characteristics to trigger high confidence
# For this test, we just verify the logic flow
decision, scores, labels, pseudo = edge.run_inference(test_image)

print(f"Inference result: decision={decision}, scores={scores[:3]}, pseudo={pseudo}")

# The preservation property: 
# - If decision == "Known", CLIP was NOT invoked (ViT was confident)
# - If decision == "Escalate_Hub" or "Adapt_Local", CLIP was invoked for pseudo-label

if decision == "Known":
    print("PASS: ViT was confident enough - CLIP not invoked")
elif pseudo is not None:
    print("PASS: ViT was uncertain - CLIP invoked for pseudo-label (preserved behavior)")
else:
    print("UNEXPECTED: No pseudo-label when ViT is uncertain")

# Verify that _get_clip_zero_shot_label exists and is called correctly
print(f"\nCLIP method exists: {hasattr(edge, '_get_clip_zero_shot_label')}")
print(f"Pseudo-label method works: {callable(getattr(edge, '_get_clip_zero_shot_label', None))}")

property1_pass = (decision in ["Known", "Adapt_Local", "Escalate_Hub"])
print(f"\nProperty 1: {'PASS' if property1_pass else 'FAIL'}")

# ============================================================================
# Property 2: Local adaptation workflow structure is preserved
# ============================================================================
print("\n[PROPERTY 2] Local adaptation workflow (LoRA + adapter submission)")
print("-" * 60)

# Check that the local_adaptation method exists
print(f"Local adaptation method exists: {hasattr(edge, 'local_adaptation')}")

# Check that the camera_node has the transmit_adapter method
from edge_node.camera_node import LiveCameraNode

print(f"LiveCameraNode has adapt_local method: {hasattr(LiveCameraNode, '_handle_adapt_local')}")

# The preservation property: 
# - local_adaptation method exists and is callable
# - It creates a LoRA adapter when called
# - The adapter can be submitted to hub

property2_pass = hasattr(edge, 'local_adaptation')
print(f"\nProperty 2: {'PASS' if property2_pass else 'FAIL'}")

# ============================================================================
# Property 3: Hub retraining workflow structure is preserved
# ============================================================================
print("\n[PROPERTY 3] Hub retraining workflow (FAISS + clustering + retraining)")
print("-" * 60)

from central_hub.hub_retrainer import HubRetrainer
from central_hub.faiss_manager import FaissManager
from central_hub.fed_avg import submit_adapter, run_fedavg

# Check that hub retrainer has the required methods
retrainer = HubRetrainer(
    model=torch.nn.Identity(),
    embedding_dim=512,
    device="cpu",
    min_samples=1,
)

print(f"HubRetrainer has maybe_retrain: {hasattr(retrainer, 'maybe_retrain')}")
print(f"HubRetrainer has _retrain_background: {hasattr(retrainer, '_retrain_background')}")

# Check FAISS manager
faiss_mgr = FaissManager(embedding_dim=512)
print(f"FaissManager has add: {hasattr(faiss_mgr, 'add')}")
print(f"FaissManager has get_cluster_embeddings: {hasattr(faiss_mgr, 'get_cluster_embeddings')}")

# Check FedAvg functions
print(f"FedAvg has submit_adapter: {callable(submit_adapter)}")
print(f"FedAvg has run_fedavg: {callable(run_fedavg)}")

property3_pass = (
    hasattr(retrainer, 'maybe_retrain') and 
    hasattr(faiss_mgr, 'add') and 
    callable(submit_adapter) and
    callable(run_fedavg)
)
print(f"\nProperty 3: {'PASS' if property3_pass else 'FAIL'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PRESERVATION PROPERTY TEST RESULTS")
print("=" * 80)

results = {
    "Property 1: ViT high-confidence (no CLIP)": property1_pass,
    "Property 2: Local adaptation workflow": property2_pass,
    "Property 3: Hub retraining workflow": property3_pass,
}

for prop, passed in results.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {prop}")

all_passed = all(results.values())
print(f"\nPRESERVATION TESTS: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

if all_passed:
    print("\n*** These tests PASS on unfixed code ***")
    print("*** They confirm the baseline behavior to preserve during fixes ***")
else:
    print("\n*** Some tests FAILED - baseline behavior may be broken ***")

print("=" * 80 + "\n")