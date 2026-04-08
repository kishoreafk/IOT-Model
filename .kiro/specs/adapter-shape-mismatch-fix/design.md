# Adapter Shape Mismatch Bugfix Design

## Overview

This bugfix addresses a critical lifecycle issue in the federated learning system where edge nodes send incorrect class names to the hub, causing shape mismatches when loading hub-trained adapters. The bug manifests across four phases:

1. **Phase 1 (Edge Detection)**: CLIP uses wrong candidate labels (50 ViT training classes instead of broad real-world categories)
2. **Phase 2 (Hub Training)**: Hub trains on wrong class names, creating adapters with N classes (actual objects seen) instead of 50 classes
3. **Phase 3 (Adapter Sync)**: Edge nodes fail to load hub adapters due to shape mismatch (N ≠ 50)
4. **Phase 4 (Edge Inference)**: Edge nodes use wrong class_names for displaying results

The fix ensures CLIP identifies real-world objects using broad categories, the hub trains on these CLIP-identified classes, edge nodes properly handle variable-sized adapters, and inference uses the hub's class names.

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug - when CLIP uses the 50 ViT training classes as candidate labels instead of broad real-world categories, causing wrong pseudo-labels to be sent to the hub
- **Property (P)**: The desired behavior - CLIP should use broad real-world categories, hub should train on CLIP-identified classes, edge should load variable-sized adapters, and inference should use hub class names
- **Preservation**: Existing ViT high-confidence classification, local adaptation, FedAvg, and hub retraining workflows that must remain unchanged
- **candidate_labels**: The list of class names passed to CLIP for zero-shot classification (currently wrong: uses 50 ViT training classes; should be: broad real-world categories)
- **clip_pseudo_label**: The class name identified by CLIP and sent to the hub in ingress updates (currently wrong: ViT training class; should be: actual object name)
- **hub_projection**: The projection layer trained by the hub that maps CLIP embeddings (512-dim) to class logits (N classes where N = number of unique CLIP-identified objects)
- **hub_projection_classes**: The list of class names corresponding to the hub projection layer's output classes (currently missing or not used; should be: stored and used for inference)
- **class_names.txt**: The file containing 50 ViT training classes (goldfish, salamander, etc.) - currently misused as CLIP candidate labels
- **shape mismatch**: RuntimeError when loading hub adapter because projection layer has N classes but edge expects 50 classes

## Bug Details

### Bug Condition

The bug manifests when the camera sees a real-world object that was NOT in the ViT training set (e.g., "car", "person", "building"). The system incorrectly uses the 50 ViT training classes as CLIP candidate labels, causing CLIP to return the closest match from those 50 classes instead of the actual object name. This wrong class name propagates through the entire lifecycle.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type (Image, candidate_labels)
  OUTPUT: boolean
  
  RETURN input.candidate_labels == class_names_from_file("configs/class_names.txt")
         AND input.image_contains_real_world_object NOT IN input.candidate_labels
         AND CLIP_identifies_closest_match_from_50_classes(input)
         AND hub_trains_on_wrong_class_names()
         AND edge_fails_to_load_hub_adapter_due_to_shape_mismatch()
END FUNCTION
```

### Examples

**Example 1: Car Detection**
- Camera sees: "car"
- ViT confidence: 0.35 (uncertain, triggers CLIP)
- CLIP candidate_labels: ["goldfish", "salamander", "bullfrog", ...] (50 ViT training classes)
- CLIP returns: "goldfish" (closest match from 50 classes)
- Sent to hub: clip_pseudo_label="goldfish"
- Hub trains on: "goldfish" (wrong - should be "car")
- Result: Hub has 47 unique classes (actual objects), edge expects 50 classes → shape mismatch

**Example 2: Person Detection**
- Camera sees: "person"
- ViT confidence: 0.42 (uncertain, triggers CLIP)
- CLIP candidate_labels: ["goldfish", "salamander", ...] (50 ViT training classes)
- CLIP returns: "salamander" (closest match)
- Sent to hub: clip_pseudo_label="salamander"
- Hub trains on: "salamander" (wrong - should be "person")
- Result: Wrong training data accumulates at hub

**Example 3: Adapter Loading Failure**
- Hub trains on 47 unique CLIP-identified classes (actual objects seen by cameras)
- Hub creates projection layer: Linear(512 → 47)
- Edge node receives adapter with 47 classes
- Edge node tries to load into existing projection layer: Linear(512 → 50)
- Result: RuntimeError: size mismatch for weight: copying a param with shape torch.Size([47, 512]) from checkpoint, the shape in current model is torch.Size([50, 512])

**Example 4: Inference Display Error**
- Edge node successfully loads hub adapter (if shape mismatch is fixed)
- Hub adapter has class_names: ["car", "person", "building", ...]
- Edge node performs inference using hub projection
- Edge node displays result using local class_names.txt: "goldfish"
- Result: User sees "goldfish" instead of "car"

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- ViT high-confidence classification (>= known_threshold) must continue to work exactly as before
- Local adaptation (trigger="adapt_local") with LoRA fine-tuning must continue to work
- FedAvg aggregation of edge adapters must continue to work
- Hub retraining on escalated embeddings must continue to work
- Secure transmission with encryption and signing must continue to work
- Adapter sync polling and hot-swapping must continue to work

**Scope:**
All inputs where the ViT model has high confidence (>= known_threshold) should be completely unaffected by this fix. This includes:
- Images containing objects from the 50 ViT training classes with high confidence
- Local adaptation workflow when ViT confidence is in adapt_local range
- FedAvg workflow when edge nodes send adapter weights
- Hub retraining workflow structure (only the class names change, not the workflow)

## Hypothesized Root Cause

Based on the bug description and code analysis, the root causes are:

1. **Wrong CLIP Candidate Labels (Phase 1)**: 
   - `camera_node.py` initializes `candidate_labels` with a small hardcoded list (10 items) or uses default
   - `vision_agent.py` uses `_get_default_labels()` which loads from `class_names.txt` (50 ViT training classes)
   - CLIP should use broad real-world categories (100+ classes) like ["car", "truck", "person", "bicycle", "dog", "cat", "tree", "building", "road", "sky", ...]
   - Current: CLIP tries to match "car" against ["goldfish", "salamander", ...] → returns "goldfish"

2. **Hub Trains on Wrong Class Names (Phase 2)**:
   - `hub_retrainer.py` receives `clip_pseudo_label` from edge nodes
   - These labels are wrong (e.g., "goldfish" instead of "car") due to Phase 1 bug
   - Hub accumulates 47 unique wrong class names and trains projection layer with 47 classes
   - Hub creates `Linear(512 → 47)` instead of `Linear(512 → 50)`

3. **Missing Class Names Metadata (Phase 2)**:
   - `hub_retrainer.py` creates adapter with metadata including `class_names`
   - However, edge nodes don't properly extract or store this metadata
   - Edge nodes need to know which class index corresponds to which class name

4. **Shape Mismatch on Adapter Load (Phase 3)**:
   - `adapter_sync.py` tries to load hub adapter state_dict into existing projection layer
   - Existing projection layer has shape `Linear(512 → 50)` (from local ViT training)
   - Hub adapter has shape `Linear(512 → 47)` (from CLIP-identified classes)
   - `load_state_dict()` fails with size mismatch error
   - Fix: Recreate projection layer with correct shape before loading

5. **Wrong Class Names for Inference (Phase 4)**:
   - `vision_agent.py` uses `self.hub_projection_classes` for inference
   - This field is set in `adapter_sync.py` when loading adapter metadata
   - However, inference code falls back to local `_get_default_labels()` if not set
   - Result: Displays "goldfish" instead of "car"

## Correctness Properties

Property 1: Bug Condition - CLIP Uses Broad Real-World Categories

_For any_ image where the ViT model is uncertain (confidence < known_threshold), the fixed system SHALL invoke CLIP zero-shot classification using a broad set of real-world object categories (100+ classes including "car", "person", "building", "tree", "dog", "cat", etc.) NOT limited to the 50 ViT training classes, and SHALL send the CLIP-identified object name as clip_pseudo_label to the hub.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

Property 2: Hub Training - Trains on CLIP-Identified Classes

_For any_ set of ingress updates with trigger="escalate_hub", the fixed hub SHALL train a projection layer on the N unique CLIP-identified class names received from edge nodes (where N is the number of distinct real-world objects seen), SHALL include the class_names list in the adapter metadata, and SHALL create a projection layer with shape Linear(512 → N).

**Validates: Requirements 2.5, 2.6, 2.7**

Property 3: Adapter Loading - Handles Variable-Sized Adapters

_For any_ hub adapter with N classes (where N ≠ 50), the fixed edge node SHALL successfully load the adapter by: (1) extracting the num_classes metadata, (2) creating a new projection layer with shape Linear(512 → N), (3) loading the state_dict into the new projection layer, (4) storing the hub's class_names for inference use.

**Validates: Requirements 2.8, 2.9**

Property 4: Inference Display - Uses Hub Class Names

_For any_ inference performed using the hub projection layer, the fixed edge node SHALL use the hub's class_names (stored from adapter metadata) to label the output, NOT the local ViT class_names from class_names.txt, ensuring users see the actual CLIP-identified object name (e.g., "car") instead of a ViT training class (e.g., "goldfish").

**Validates: Requirements 2.10, 2.11**

Property 5: Preservation - ViT High-Confidence Classification

_For any_ image where the ViT model has high confidence (>= known_threshold), the fixed system SHALL produce exactly the same classification result as the original system, using the ViT model without invoking CLIP, preserving all existing high-confidence classification behavior.

**Validates: Requirements 3.1**

Property 6: Preservation - Local Adaptation and FedAvg

_For any_ local adaptation event (trigger="adapt_local"), the fixed system SHALL produce exactly the same behavior as the original system: fine-tune the LoRA adapter, save adapter weights, send to hub, and trigger FedAvg aggregation, preserving all existing local adaptation and FedAvg workflows.

**Validates: Requirements 3.2, 3.3, 3.4**

Property 7: Preservation - Hub Retraining Workflow

_For any_ escalate_hub event, the fixed system SHALL preserve the existing hub retraining workflow structure: store embeddings in FAISS, cluster them, trigger retraining when clusters reach minimum size, train projection layer, and broadcast to edge nodes. Only the class names used for training change (from wrong ViT classes to correct CLIP classes), not the workflow itself.

**Validates: Requirements 3.5, 3.6, 3.7**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File 1**: `edge_node/camera_node.py`

**Function**: `__init__`

**Specific Changes**:
1. **Expand CLIP Candidate Labels**: Replace the small hardcoded list (10 items) with a comprehensive set of 100+ real-world object categories
   - Current: `["car", "truck", "person", "bicycle", "dog", "cat", "chair", "table", "phone", "laptop"]`
   - Fixed: Include common objects, animals, vehicles, buildings, natural elements, etc.
   - Examples: "car", "truck", "bus", "motorcycle", "bicycle", "person", "child", "dog", "cat", "bird", "tree", "building", "house", "road", "sidewalk", "grass", "sky", "cloud", "sun", "moon", "flower", "plant", "chair", "table", "desk", "bed", "sofa", "door", "window", "computer", "laptop", "phone", "keyboard", "mouse", "monitor", "book", "pen", "paper", "bag", "backpack", "bottle", "cup", "plate", "food", "fruit", "vegetable", etc.

**File 2**: `edge_node/vision_agent.py`

**Function**: `run_inference`

**Specific Changes**:
1. **Use Broad Candidate Labels for CLIP**: When calling `_get_clip_zero_shot_label()`, pass the broad candidate_labels from camera_node instead of using `_get_default_labels()` (which loads the 50 ViT training classes)
   - Current: Uses `candidate_labels` parameter but falls back to `_get_default_labels()` internally
   - Fixed: Ensure CLIP always uses the broad candidate_labels passed from camera_node

2. **Store Hub Class Names**: When hub_projection is loaded, ensure `hub_projection_classes` is properly set and used for inference
   - Current: `hub_projection_classes` is set in adapter_sync but may not be used consistently
   - Fixed: Always use `hub_projection_classes` when hub_projection is active

3. **Use Hub Class Names for Inference**: In the hub_projection inference branch, use `projection_classes` (hub's class names) instead of local class names
   - Current: Falls back to `candidate_labels` or `_get_default_labels()`
   - Fixed: Use `self.hub_projection_classes` when available

**File 3**: `edge_node/adapter_sync.py`

**Function**: `_hot_swap_adapter`

**Specific Changes**:
1. **Extract Adapter Metadata**: Parse the loaded adapter data to extract `num_classes` and `class_names`
   - Current: Extracts metadata but doesn't fully handle projection layer shape mismatch
   - Fixed: Extract `num_classes` from metadata (e.g., 47 instead of 50)

2. **Recreate Projection Layer with Correct Shape**: Instead of trying to load into existing projection layer, create a new one with the correct shape
   - Current: `proj_layer = nn.Linear(512, num_classes)` but uses hardcoded `num_classes = 50`
   - Fixed: Use `num_classes` from adapter metadata (e.g., 47)
   - Create: `proj_layer = nn.Linear(512, num_classes).to(self.vision_node.device)`

3. **Load State Dict into New Projection Layer**: Load the adapter state_dict into the newly created projection layer
   - Current: Tries to load with `strict=True` which fails on shape mismatch
   - Fixed: Create new layer first, then load with `strict=True` (should succeed now)

4. **Store Hub Class Names**: Store the `class_names` from adapter metadata in `vision_node.hub_projection_classes`
   - Current: Already implemented but ensure it's always set
   - Fixed: Verify `self.vision_node.hub_projection_classes = class_names` is executed

**File 4**: `central_hub/hub_retrainer.py`

**Function**: `_retrain_background`

**Specific Changes**:
1. **Use CLIP-Identified Class Names**: Instead of using `self._class_names` (loaded from class_names.txt with 50 ViT training classes), build the class list from the unique pseudo_labels received from edge nodes
   - Current: Uses `self._class_names` which has 50 ViT training classes
   - Fixed: Extract unique class names from `pseudo_labels` parameter (e.g., ["car", "person", "building", ...])
   - Build: `unique_classes = sorted(set(label for label in pseudo_labels if label))`
   - Create: `label_to_idx = {name: i for i, name in enumerate(unique_classes)}`

2. **Create Projection Layer with Correct Size**: Create projection layer with N classes where N = len(unique_classes)
   - Current: `projection = nn.Linear(self.embedding_dim, len(self._class_names))` uses 50 classes
   - Fixed: `projection = nn.Linear(self.embedding_dim, len(unique_classes))` uses N classes

3. **Include Class Names in Adapter Metadata**: Ensure the adapter metadata includes the correct class_names list
   - Current: Already includes `class_names: self._class_names` (wrong - 50 ViT classes)
   - Fixed: Include `class_names: unique_classes` (correct - N CLIP-identified classes)

4. **Handle Empty or Missing Pseudo-Labels**: If no pseudo_labels are provided, fall back to a reasonable default or skip training
   - Current: May crash if pseudo_labels is None or empty
   - Fixed: Check if pseudo_labels is valid before building unique_classes

## Testing Strategy

### Validation Approach

The testing strategy follows a three-phase approach:
1. **Exploratory Bug Condition Checking**: Surface counterexamples on unfixed code to confirm root cause
2. **Fix Checking**: Verify the fix works correctly for all buggy inputs
3. **Preservation Checking**: Verify existing behavior is unchanged for non-buggy inputs

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that simulate the complete lifecycle (edge detection → hub training → adapter sync → edge inference) on the UNFIXED code to observe failures and understand the root cause.

**Test Cases**:
1. **Phase 1 Test - Wrong CLIP Labels**: Simulate edge node seeing "car", verify CLIP uses 50 ViT training classes as candidates, verify CLIP returns "goldfish" instead of "car" (will fail on unfixed code - wrong label returned)

2. **Phase 2 Test - Hub Training on Wrong Classes**: Send multiple ingress updates with wrong clip_pseudo_labels (e.g., "goldfish", "salamander"), verify hub trains on these wrong classes, verify hub creates projection layer with N classes where N ≠ 50 (will fail on unfixed code - hub has wrong class count)

3. **Phase 3 Test - Adapter Shape Mismatch**: Simulate hub sending adapter with 47 classes to edge node expecting 50 classes, verify edge node fails to load adapter with RuntimeError: size mismatch (will fail on unfixed code - shape mismatch error)

4. **Phase 4 Test - Wrong Inference Display**: Simulate edge node loading hub adapter (if shape mismatch is bypassed), verify edge node displays "goldfish" instead of "car" when using hub projection (will fail on unfixed code - wrong label displayed)

**Expected Counterexamples**:
- CLIP returns "goldfish" when camera sees "car" (due to wrong candidate labels)
- Hub trains on 47 unique wrong class names instead of correct CLIP-identified classes
- Edge node fails to load hub adapter with RuntimeError: size mismatch
- Edge node displays "goldfish" instead of "car" in inference results

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed system produces the expected behavior.

**Pseudocode:**
```
FOR ALL image WHERE isBugCondition(image) DO
  // Phase 1: Edge Detection
  candidate_labels := broad_real_world_categories()  // 100+ classes
  clip_label := CLIP_zero_shot(image, candidate_labels)
  ASSERT clip_label IN broad_real_world_categories()
  ASSERT clip_label NOT IN vit_training_classes()
  
  // Phase 2: Hub Training
  hub_receives_ingress_update(clip_label)
  unique_classes := extract_unique_classes_from_all_ingress_updates()
  hub_projection := train_projection_layer(unique_classes)
  ASSERT hub_projection.shape == (512, len(unique_classes))
  ASSERT adapter_metadata.class_names == unique_classes
  
  // Phase 3: Adapter Sync
  edge_receives_adapter(hub_projection, adapter_metadata)
  ASSERT edge_loads_adapter_successfully()
  ASSERT edge.hub_projection.shape == (512, len(unique_classes))
  ASSERT edge.hub_projection_classes == unique_classes
  
  // Phase 4: Edge Inference
  inference_result := edge_inference_with_hub_projection(image)
  ASSERT inference_result.label IN unique_classes
  ASSERT inference_result.label == clip_label
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed system produces the same result as the original system.

**Pseudocode:**
```
FOR ALL image WHERE NOT isBugCondition(image) DO
  // High-confidence ViT classification
  IF vit_confidence(image) >= known_threshold THEN
    ASSERT fixed_system(image) == original_system(image)
  END IF
  
  // Local adaptation workflow
  IF vit_confidence(image) IN [adapt_threshold, known_threshold) THEN
    ASSERT fixed_local_adaptation(image) == original_local_adaptation(image)
    ASSERT fixed_fedavg() == original_fedavg()
  END IF
  
  // Hub retraining workflow structure
  IF trigger == "escalate_hub" THEN
    ASSERT fixed_hub_workflow_structure() == original_hub_workflow_structure()
    // Only class names change, not workflow
  END IF
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs

**Test Plan**: Observe behavior on UNFIXED code first for high-confidence ViT classification, local adaptation, and FedAvg, then write property-based tests capturing that behavior.

**Test Cases**:
1. **ViT High-Confidence Preservation**: Generate random images from the 50 ViT training classes, verify ViT classifies with high confidence, verify CLIP is NOT invoked, verify classification result is identical to unfixed code

2. **Local Adaptation Preservation**: Generate random images triggering adapt_local (confidence in [0.60, 0.85)), verify LoRA fine-tuning occurs, verify adapter weights are sent to hub, verify FedAvg is triggered, verify behavior is identical to unfixed code

3. **FedAvg Preservation**: Send multiple adapter weights from edge nodes, verify FedAvg aggregates them correctly, verify global adapter version is bumped, verify behavior is identical to unfixed code

4. **Hub Retraining Workflow Preservation**: Send escalate_hub updates, verify FAISS stores embeddings, verify clustering occurs, verify retraining is triggered at minimum cluster size, verify workflow structure is identical to unfixed code (only class names differ)

### Unit Tests

- Test CLIP zero-shot classification with broad candidate labels (100+ classes)
- Test hub projection layer creation with variable number of classes (N ≠ 50)
- Test edge adapter loading with shape mismatch handling (recreate projection layer)
- Test edge inference using hub class names instead of local class names
- Test adapter metadata extraction (num_classes, class_names)
- Test unique class extraction from pseudo_labels in hub retrainer
- Test edge cases: empty pseudo_labels, None pseudo_labels, duplicate class names

### Property-Based Tests

- Generate random real-world object images, verify CLIP identifies them correctly using broad categories
- Generate random sets of ingress updates with varying class names, verify hub trains on correct unique classes
- Generate random hub adapters with varying class counts (10, 20, 47, 100), verify edge loads them successfully
- Generate random inference scenarios with hub projection, verify correct class names are displayed
- Generate random high-confidence ViT images, verify preservation of original behavior
- Generate random local adaptation scenarios, verify preservation of LoRA and FedAvg workflows

### Integration Tests

- Test full lifecycle: edge detection (CLIP with broad labels) → hub training (on CLIP classes) → adapter sync (variable size) → edge inference (hub class names)
- Test multiple edge nodes sending different CLIP-identified classes to hub
- Test hub accumulating 47 unique classes from multiple edge nodes
- Test edge nodes receiving and loading hub adapter with 47 classes
- Test edge nodes displaying correct CLIP-identified class names in inference
- Test mixed scenario: some images with high ViT confidence (preserved), some with low confidence (CLIP with broad labels)
- Test adapter version bumping and edge node sync after hub retraining
- Test visual feedback in camera_node showing correct class names
