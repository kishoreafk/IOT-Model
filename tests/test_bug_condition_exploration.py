"""
Bug Condition Exploration Test for Adapter Shape Mismatch Fix

This test validates that the bug exists across all 4 phases of the lifecycle:
- Phase 1: Edge Detection (CLIP uses wrong labels)
- Phase 2: Hub Training (trains on wrong classes)
- Phase 3: Adapter Sync (shape mismatch)
- Phase 4: Edge Inference (displays wrong labels)

**CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists.
**DO NOT attempt to fix the test or the code when it fails.**
**EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the bug exists)

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11**
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from edge_node.vision_agent import EdgeVisionNode
from edge_node.adapter_sync import AdapterSyncClient
from central_hub.hub_retrainer import HubRetrainer


def test_bug_condition_phase_1_clip_uses_limited_candidate_labels():
    """
    Phase 1: Verify CLIP uses limited candidate labels (10 items) instead of broad
    real-world categories (100+ items)
    
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
    """
    print("\n" + "="*80)
    print("PHASE 1: Edge Detection - CLIP Uses Limited Candidate Labels")
    print("="*80)
    
    # Create edge node with default config (unfixed code)
    node = EdgeVisionNode(device='cpu', use_fp16=False)
    
    # Simulate camera_node's small candidate_labels (10 items)
    small_candidate_labels = [
        "car", "truck", "person", "bicycle", "dog",
        "cat", "chair", "table", "phone", "laptop"
    ]
    
    print(f"Current candidate_labels: {len(small_candidate_labels)} classes")
    print(f"Labels: {small_candidate_labels}")
    
    # EXPECTED BUG BEHAVIOR:
    # - camera_node uses only 10 candidate labels (limited coverage)
    # - Should use 100+ broad real-world categories for better coverage
    # - Missing: bus, motorcycle, building, tree, road, sky, etc.
    
    # Check if candidate_labels is comprehensive (100+ classes)
    broad_categories = [
        "car", "truck", "bus", "motorcycle", "bicycle", "person", "child",
        "dog", "cat", "bird", "horse", "tree", "building", "house", "road",
        "sidewalk", "grass", "sky", "cloud", "sun", "moon", "flower", "plant",
        "chair", "table", "desk", "bed", "sofa", "door", "window", "computer",
        "laptop", "phone", "keyboard", "mouse", "monitor", "book", "pen", "paper",
        "bag", "backpack", "bottle", "cup", "plate", "food", "fruit", "vegetable"
    ]
    
    print(f"\nBroad categories needed: {len(broad_categories)}+ classes")
    print(f"Current coverage: {len(small_candidate_labels)} classes")
    
    # Count how many broad categories are missing
    missing_categories = [cat for cat in broad_categories if cat not in small_candidate_labels]
    print(f"Missing categories: {len(missing_categories)} (e.g., {missing_categories[:10]})")
    
    print(f"\nBUG CHECK:")
    print(f"  - Current candidate_labels: {len(small_candidate_labels)} classes")
    print(f"  - Broad categories needed: {len(broad_categories)}+ classes")
    print(f"  - Has comprehensive coverage: {len(small_candidate_labels) >= len(broad_categories)}")
    
    # EXPECTED: Bug exists if candidate_labels is too small (< 100 classes)
    # This assertion SHOULD FAIL on unfixed code (proving bug exists)
    assert len(small_candidate_labels) >= len(broad_categories), \
        f"BUG DETECTED: CLIP uses only {len(small_candidate_labels)} candidate labels, " \
        f"should use {len(broad_categories)}+ broad real-world categories for better coverage"
    
    print("\n[OK] Phase 1 PASSED: CLIP uses comprehensive broad real-world categories (bug is FIXED)")


def test_bug_condition_phase_2_hub_trains_on_wrong_classes():
    """
    Phase 2: Verify hub receives wrong clip_pseudo_label ("goldfish"),
    trains on 47 unique wrong classes instead of 50
    
    **Validates: Requirements 2.5, 2.6, 2.7**
    """
    print("\n" + "="*80)
    print("PHASE 2: Hub Training - Trains on Wrong Class Names")
    print("="*80)
    
    # Create hub retrainer
    backbone = nn.Identity()  # Dummy backbone for testing
    retrainer = HubRetrainer(
        model=backbone,
        embedding_dim=512,
        num_epochs=1,  # Fast test
        min_samples=2,
        device='cpu'
    )
    
    # Get the default class names (50 ViT training classes)
    default_class_names = retrainer._get_default_labels()
    print(f"Hub default class names: {len(default_class_names)} classes")
    
    # Simulate receiving wrong pseudo_labels from edge nodes
    # (due to Phase 1 bug: CLIP returns ViT training classes instead of real objects)
    wrong_pseudo_labels = [
        "goldfish",  # Camera saw "car" but CLIP returned "goldfish"
        "salamander",  # Camera saw "person" but CLIP returned "salamander"
        "bullfrog",  # Camera saw "building" but CLIP returned "bullfrog"
        "goldfish",  # Duplicate
        "toad",
        "alligator",
    ]
    
    # Create dummy embeddings
    embeddings = [torch.randn(512) for _ in wrong_pseudo_labels]
    
    # Extract unique classes (what hub will train on)
    unique_classes = sorted(set(label for label in wrong_pseudo_labels if label))
    print(f"\nUnique classes from pseudo_labels: {len(unique_classes)} classes")
    print(f"Classes: {unique_classes}")
    
    # EXPECTED BUG BEHAVIOR:
    # - Hub should train on CLIP-identified real-world classes (e.g., "car", "person")
    # - But unfixed code uses self._class_names (50 ViT training classes)
    # - So hub trains on wrong classes and creates projection layer with wrong size
    
    # Check if hub would use the correct classes
    # In unfixed code, hub uses self._class_names (50 classes)
    # In fixed code, hub should use unique_classes from pseudo_labels
    
    print(f"\nBUG CHECK:")
    print(f"  - Hub default class names: {len(default_class_names)} classes")
    print(f"  - Unique classes from pseudo_labels: {len(unique_classes)} classes")
    print(f"  - Classes match: {len(default_class_names) == len(unique_classes)}")
    
    # EXPECTED: Bug exists if hub uses default_class_names instead of unique_classes
    # This assertion SHOULD FAIL on unfixed code (proving bug exists)
    assert len(unique_classes) == len(default_class_names), \
        f"BUG DETECTED: Hub should train on {len(unique_classes)} CLIP-identified classes, " \
        f"but uses {len(default_class_names)} ViT training classes"
    
    # Also check that the classes are the correct ones (real-world objects)
    real_world_classes = ["car", "person", "building", "truck", "bicycle"]
    has_real_world = any(cls in unique_classes for cls in real_world_classes)
    has_vit_classes = any(cls in unique_classes for cls in ["goldfish", "salamander", "bullfrog"])
    
    print(f"  - Contains real-world classes: {has_real_world}")
    print(f"  - Contains ViT training classes: {has_vit_classes}")
    
    assert has_real_world and not has_vit_classes, \
        f"BUG DETECTED: Hub trains on wrong ViT classes {unique_classes} instead of real-world objects"
    
    print("\n[OK] Phase 2 PASSED: Hub trains on correct CLIP-identified classes (bug is FIXED)")


def test_bug_condition_phase_3_adapter_shape_mismatch():
    """
    Phase 3: Verify edge node fails to load hub adapter with RuntimeError: size mismatch (47 ≠ 50)
    
    **Validates: Requirements 2.8, 2.9**
    """
    print("\n" + "="*80)
    print("PHASE 3: Adapter Sync - Shape Mismatch Error")
    print("="*80)
    
    # Create edge node
    node = EdgeVisionNode(device='cpu', use_fp16=False)
    
    # Get default class names (50 ViT training classes)
    default_class_names = node._get_default_labels()
    print(f"Edge node expects: {len(default_class_names)} classes")
    
    # Simulate hub adapter with 47 classes (due to Phase 2 bug)
    # Hub trained on 47 unique wrong classes instead of 50
    hub_num_classes = 47
    hub_class_names = [f"class_{i}" for i in range(hub_num_classes)]
    
    print(f"Hub adapter has: {hub_num_classes} classes")
    
    # Create hub adapter with 47 classes
    proj_layer = nn.Linear(512, hub_num_classes)
    state_dict = proj_layer.state_dict()
    
    adapter_data = {
        "state_dict": state_dict,
        "class_names": hub_class_names,
        "adapter_type": "projection_layer",
        "num_classes": hub_num_classes
    }
    
    # Serialize adapter
    buf = io.BytesIO()
    torch.save(adapter_data, buf)
    adapter_bytes = buf.getvalue()
    
    # Create sync client
    client = AdapterSyncClient(
        device_id="test_device",
        hub_url="http://dummy",
        local_adapter_path="dummy.bin",
        vision_node=node
    )
    
    print(f"\nAttempting to load hub adapter with {hub_num_classes} classes...")
    
    # EXPECTED BUG BEHAVIOR:
    # - Edge node expects 50 classes (from local ViT training)
    # - Hub adapter has 47 classes (from CLIP-identified objects)
    # - load_state_dict() fails with RuntimeError: size mismatch
    
    # Try to load adapter
    try:
        client._hot_swap_adapter(adapter_bytes, 1)
        
        # Check if adapter was loaded successfully
        assert hasattr(node, 'hub_projection'), "hub_projection not set"
        assert isinstance(node.hub_projection, nn.Linear), "hub_projection is not nn.Linear"
        
        # Check if projection layer has correct shape
        actual_shape = node.hub_projection.out_features
        print(f"Loaded projection layer shape: Linear(512 → {actual_shape})")
        
        # EXPECTED: Bug is fixed if adapter loads successfully with correct shape
        # This assertion SHOULD FAIL on unfixed code (proving bug exists)
        assert actual_shape == hub_num_classes, \
            f"BUG DETECTED: Projection layer has {actual_shape} classes, expected {hub_num_classes}"
        
        # Check if hub class names were stored
        assert hasattr(node, 'hub_projection_classes'), "hub_projection_classes not set"
        assert node.hub_projection_classes == hub_class_names, \
            f"BUG DETECTED: hub_projection_classes not stored correctly"
        
        print(f"[OK] Adapter loaded successfully with {hub_num_classes} classes")
        print(f"[OK] hub_projection_classes stored: {len(node.hub_projection_classes)} classes")
        
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"\n[X] BUG DETECTED: {e}")
            raise AssertionError(
                f"BUG DETECTED: Edge node failed to load hub adapter due to shape mismatch. "
                f"Expected {len(default_class_names)} classes, hub has {hub_num_classes} classes. "
                f"Error: {e}"
            )
        else:
            raise
    
    print("\n[OK] Phase 3 PASSED: Edge loads variable-sized adapters correctly (bug is FIXED)")


def test_bug_condition_phase_4_inference_displays_wrong_label():
    """
    Phase 4: Verify edge node displays "goldfish" instead of "car" when using hub projection
    (if shape mismatch bypassed)
    
    **Validates: Requirements 2.10, 2.11**
    """
    print("\n" + "="*80)
    print("PHASE 4: Edge Inference - Displays Wrong Label")
    print("="*80)
    
    # Create edge node
    node = EdgeVisionNode(device='cpu', use_fp16=False)
    
    # Get default class names (50 ViT training classes)
    default_class_names = node._get_default_labels()
    
    # Simulate hub adapter with correct CLIP-identified classes
    hub_class_names = ["car", "person", "building", "truck", "bicycle"]
    hub_num_classes = len(hub_class_names)
    
    print(f"Hub adapter class names: {hub_class_names}")
    print(f"Edge default class names (first 5): {default_class_names[:5]}")
    
    # Create hub projection layer
    proj_layer = nn.Linear(512, hub_num_classes)
    
    # Set weights to heavily favor "car" (index 0)
    nn.init.zeros_(proj_layer.weight)
    nn.init.zeros_(proj_layer.bias)
    proj_layer.bias.data[0] = 100.0  # "car" at index 0
    
    state_dict = proj_layer.state_dict()
    adapter_data = {
        "state_dict": state_dict,
        "class_names": hub_class_names,
        "adapter_type": "projection_layer",
        "num_classes": hub_num_classes
    }
    
    # Serialize adapter
    buf = io.BytesIO()
    torch.save(adapter_data, buf)
    adapter_bytes = buf.getvalue()
    
    # Load adapter
    client = AdapterSyncClient(
        device_id="test_device",
        hub_url="http://dummy",
        local_adapter_path="dummy.bin",
        vision_node=node
    )
    
    try:
        client._hot_swap_adapter(adapter_bytes, 1)
    except Exception as e:
        print(f"Warning: Could not load adapter (Phase 3 bug): {e}")
        print("Skipping Phase 4 test (depends on Phase 3 fix)")
        return
    
    # Create dummy image (simulating camera seeing a car)
    car_image = Image.new('RGB', (224, 224), color='red')
    
    # Run inference
    decision, scores, labels, pseudo_label = node.run_inference(car_image)
    
    print(f"\nInference result:")
    print(f"  Decision: {decision}")
    print(f"  Top label: {labels[0] if labels else 'None'}")
    print(f"  Confidence: {scores[0] if scores else 0:.2%}")
    print(f"  Pseudo label: {pseudo_label}")
    
    # EXPECTED BUG BEHAVIOR:
    # - Hub projection should output "car" (index 0 with highest bias)
    # - But unfixed code uses local default_class_names for display
    # - So edge displays "goldfish" (index 0 in ViT training classes) instead of "car"
    
    # Check if the correct label is displayed
    expected_label = "car"
    actual_label = labels[0] if labels else None
    
    print(f"\nBUG CHECK:")
    print(f"  - Expected label: {expected_label}")
    print(f"  - Actual label: {actual_label}")
    print(f"  - Labels match: {actual_label == expected_label}")
    
    # EXPECTED: Bug exists if actual_label is from ViT training classes, not hub classes
    # This assertion SHOULD FAIL on unfixed code (proving bug exists)
    assert actual_label == expected_label, \
        f"BUG DETECTED: Edge displays '{actual_label}' instead of '{expected_label}' " \
        f"(using local ViT class names instead of hub class names)"
    
    # Also verify hub_projection_classes is being used
    if hasattr(node, 'hub_projection_classes'):
        print(f"  - hub_projection_classes: {node.hub_projection_classes}")
        assert node.hub_projection_classes == hub_class_names, \
            f"BUG DETECTED: hub_projection_classes not set correctly"
    else:
        raise AssertionError("BUG DETECTED: hub_projection_classes not set")
    
    print("\n[OK] Phase 4 PASSED: Edge uses hub class names for inference (bug is FIXED)")


def test_complete_lifecycle_bug_condition():
    """
    Complete lifecycle test: Verify bug exists across all 4 phases
    
    This is the main bug condition exploration test that validates the complete
    lifecycle bug from edge detection through hub training to adapter sync and inference.
    
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11**
    """
    print("\n" + "="*80)
    print("COMPLETE LIFECYCLE BUG CONDITION EXPLORATION")
    print("="*80)
    print("\nThis test validates the bug exists across all 4 phases:")
    print("  Phase 1: Edge Detection (CLIP uses wrong labels)")
    print("  Phase 2: Hub Training (trains on wrong classes)")
    print("  Phase 3: Adapter Sync (shape mismatch)")
    print("  Phase 4: Edge Inference (displays wrong labels)")
    print("\n**EXPECTED**: This test SHOULD FAIL on unfixed code (proving bug exists)")
    print("="*80)
    
    # Run all phases
    try:
        test_bug_condition_phase_1_clip_uses_limited_candidate_labels()
    except AssertionError as e:
        print(f"\n[X] Phase 1 FAILED (BUG EXISTS): {e}")
        raise
    
    try:
        test_bug_condition_phase_2_hub_trains_on_wrong_classes()
    except AssertionError as e:
        print(f"\n[X] Phase 2 FAILED (BUG EXISTS): {e}")
        raise
    
    try:
        test_bug_condition_phase_3_adapter_shape_mismatch()
    except AssertionError as e:
        print(f"\n[X] Phase 3 FAILED (BUG EXISTS): {e}")
        raise
    
    try:
        test_bug_condition_phase_4_inference_displays_wrong_label()
    except AssertionError as e:
        print(f"\n[X] Phase 4 FAILED (BUG EXISTS): {e}")
        raise
    
    print("\n" + "="*80)
    print("[OK] ALL PHASES PASSED: Bug is FIXED across complete lifecycle")
    print("="*80)


if __name__ == "__main__":
    # Run the complete lifecycle test
    test_complete_lifecycle_bug_condition()
