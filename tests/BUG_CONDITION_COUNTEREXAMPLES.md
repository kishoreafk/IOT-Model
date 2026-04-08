# Bug Condition Exploration - Counterexamples Found

**Test File**: `tests/test_bug_condition_exploration.py`  
**Test Date**: Run on unfixed code  
**Status**: **BUGS CONFIRMED** - Test failures prove the bug exists across multiple phases

## Summary

The bug condition exploration test successfully identified bugs in **2 out of 4 phases** of the federated learning lifecycle:

- ✅ **Phase 1 (Edge Detection)**: BUG CONFIRMED - CLIP uses only 10 candidate labels instead of 100+
- ✅ **Phase 2 (Hub Training)**: BUG CONFIRMED - Hub uses 50 ViT training classes instead of unique CLIP-identified classes
- ❌ **Phase 3 (Adapter Sync)**: NO BUG - Adapter loading works correctly (already fixed or not present)
- ❌ **Phase 4 (Edge Inference)**: NO BUG - Inference uses hub class names correctly (already fixed or not present)

## Detailed Counterexamples

### Phase 1: Edge Detection - Limited Candidate Labels

**Bug Description**: CLIP uses only 10 candidate labels instead of comprehensive 100+ broad real-world categories

**Counterexample**:
```
Current candidate_labels: 10 classes
Labels: ['car', 'truck', 'person', 'bicycle', 'dog', 'cat', 'chair', 'table', 'phone', 'laptop']

Broad categories needed: 47+ classes
Current coverage: 10 classes
Missing categories: 37 (e.g., ['bus', 'motorcycle', 'child', 'bird', 'horse', 'tree', 'building', 'house', 'road', 'sidewalk'])

BUG CHECK:
  - Current candidate_labels: 10 classes
  - Broad categories needed: 47+ classes
  - Has comprehensive coverage: False
```

**Assertion Failure**:
```
AssertionError: BUG DETECTED: CLIP uses only 10 candidate labels, should use 47+ broad real-world categories for better coverage
assert 10 >= 47
```

**Impact**: 
- Limited coverage means CLIP cannot identify many real-world objects
- Missing categories include: bus, motorcycle, building, tree, road, sky, cloud, sun, moon, flower, plant, desk, bed, sofa, door, window, computer, keyboard, mouse, monitor, book, pen, paper, bag, backpack, bottle, cup, plate, food, fruit, vegetable, etc.
- This severely limits the system's ability to handle diverse real-world scenarios

**Requirements Validated**: 2.1, 2.2, 2.3, 2.4

---

### Phase 2: Hub Training - Wrong Class Names

**Bug Description**: Hub trains on 50 ViT training classes instead of unique CLIP-identified classes from pseudo_labels

**Counterexample**:
```
Hub default class names: 50 classes

Unique classes from pseudo_labels: 5 classes
Classes: ['alligator', 'bullfrog', 'goldfish', 'salamander', 'toad']

BUG CHECK:
  - Hub default class names: 50 classes
  - Unique classes from pseudo_labels: 5 classes
  - Classes match: False
```

**Assertion Failure**:
```
AssertionError: BUG DETECTED: Hub should train on 5 CLIP-identified classes, but uses 50 ViT training classes
assert 5 == 50
```

**Impact**:
- Hub creates projection layer with wrong number of classes (50 instead of actual unique classes)
- Hub ignores the actual CLIP-identified classes from edge nodes
- This causes shape mismatch when edge nodes try to load hub adapters
- Training data is misaligned with the actual classes seen by cameras

**Requirements Validated**: 2.5, 2.6, 2.7

---

### Phase 3: Adapter Sync - Shape Mismatch (NO BUG FOUND)

**Expected Bug**: Edge node should fail to load hub adapter with RuntimeError: size mismatch (47 ≠ 50)

**Actual Result**: **PASSED** - Adapter loaded successfully

```
Edge node expects: 50 classes
Hub adapter has: 47 classes

Attempting to load hub adapter with 47 classes...
Loaded projection layer shape: Linear(512 → 47)
[OK] Adapter loaded successfully with 47 classes
[OK] hub_projection_classes stored: 47 classes
```

**Analysis**: 
- The adapter_sync.py code already handles variable-sized adapters correctly
- It creates a new projection layer with the correct shape from metadata
- It stores hub_projection_classes for inference use
- This phase appears to be already fixed or the bug was not present

**Requirements Validated**: 2.8, 2.9

---

### Phase 4: Edge Inference - Wrong Label Display (NO BUG FOUND)

**Expected Bug**: Edge node should display wrong label (e.g., "goldfish" instead of "car") when using hub projection

**Actual Result**: **PASSED** - Correct label displayed

```
Hub adapter class names: ['car', 'person', 'building', 'truck', 'bicycle']
Edge default class names (first 5): ['n01443537 goldfish Carassius auratus', ...]

Inference result:
  Decision: Known
  Top label: car
  Confidence: 100.00%
  Pseudo label: car

BUG CHECK:
  - Expected label: car
  - Actual label: car
  - Labels match: True
  - hub_projection_classes: ['car', 'person', 'building', 'truck', 'bicycle']
```

**Analysis**:
- The vision_agent.py code already uses hub_projection_classes for inference
- It correctly displays the hub's class names instead of local ViT class names
- This phase appears to be already fixed or the bug was not present

**Requirements Validated**: 2.10, 2.11

---

## Root Cause Analysis

Based on the counterexamples, the confirmed bugs are:

### 1. Limited CLIP Candidate Labels (Phase 1)
**Location**: `edge_node/camera_node.py` line 75-85
```python
self.candidate_labels = candidate_labels or [
    "car", "truck", "person", "bicycle", "dog",
    "cat", "chair", "table", "phone", "laptop",
]
```

**Fix Required**: Expand to 100+ broad real-world categories including:
- Vehicles: car, truck, bus, motorcycle, bicycle, etc.
- People: person, child, adult, etc.
- Animals: dog, cat, bird, horse, etc.
- Buildings: building, house, store, office, etc.
- Natural elements: tree, grass, sky, cloud, sun, moon, flower, plant, etc.
- Indoor objects: chair, table, desk, bed, sofa, door, window, etc.
- Electronics: computer, laptop, phone, keyboard, mouse, monitor, etc.
- Everyday items: book, pen, paper, bag, backpack, bottle, cup, plate, etc.
- Food: food, fruit, vegetable, bread, meat, etc.

### 2. Hub Uses Wrong Class Names (Phase 2)
**Location**: `central_hub/hub_retrainer.py` line 175-180 (in `_retrain_background` method)
```python
# Current (buggy):
label_to_idx = {name.lower(): i for i, name in enumerate(self._class_names)}

# Should be:
unique_classes = sorted(set(label for label in pseudo_labels if label))
label_to_idx = {name.lower(): i for i, name in enumerate(unique_classes)}
```

**Fix Required**: 
- Extract unique class names from `pseudo_labels` parameter
- Build projection layer with N classes where N = len(unique_classes)
- Include `class_names: unique_classes` in adapter metadata (not `self._class_names`)

---

## Test Execution Summary

**Command**: `python -m pytest tests/test_bug_condition_exploration.py::test_complete_lifecycle_bug_condition -v -s`

**Result**: **FAILED** (as expected - proves bugs exist)

**Phases Failed**: 2 out of 4
- Phase 1: FAILED ✅ (bug confirmed)
- Phase 2: FAILED ✅ (bug confirmed)
- Phase 3: PASSED ❌ (no bug found)
- Phase 4: PASSED ❌ (no bug found)

**Next Steps**:
1. Implement fixes for Phase 1 (expand candidate_labels)
2. Implement fixes for Phase 2 (use unique CLIP classes)
3. Re-run test to verify fixes work
4. Test should PASS after fixes are implemented

---

## Conclusion

The bug condition exploration test successfully identified and documented the bugs in the federated learning lifecycle. The counterexamples prove that:

1. **Phase 1 Bug Exists**: CLIP uses only 10 candidate labels instead of 100+ broad categories
2. **Phase 2 Bug Exists**: Hub trains on 50 ViT training classes instead of unique CLIP-identified classes
3. **Phase 3 Already Fixed**: Adapter loading handles variable-sized adapters correctly
4. **Phase 4 Already Fixed**: Inference uses hub class names correctly

The test is working as intended - it FAILS on unfixed code, proving the bugs exist. After implementing the fixes, the test should PASS, confirming the bugs are resolved.
