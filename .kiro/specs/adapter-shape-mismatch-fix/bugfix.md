# Bugfix Requirements Document

## Introduction

The edge node is sending WRONG class names to the hub in ingress updates, and the system fails to properly handle the complete federated learning lifecycle. This bug has multiple manifestations across the workflow:

**Phase 1 - Edge Detection (CLIP Labeling):** When the camera sees a real-world object (like a car or person), CLIP correctly identifies it using zero-shot classification. However, the system then sends a class name from the base ViT training set (the 50 classes in class_names.txt like "goldfish", "salamander") instead of the CLIP-identified object name.

**Phase 2 - Hub Training:** This causes the hub to train on incorrect data: when the camera sees "car" and CLIP identifies it as "car", the hub receives something like "goldfish" as the pseudo_label. The hub then trains on these wrong class names, which explains why it has 47 unique classes (the actual objects seen by cameras) while edge nodes expect 50 classes (the base ViT training set).

**Phase 3 - Adapter Sync:** When the hub sends back an adapter with 47 classes, edge nodes fail to load it due to shape mismatch (expecting 50 classes). Even if the adapter loads, edge nodes don't properly store or use the hub's class_names metadata.

**Phase 4 - Edge Inference:** When edge nodes perform inference with the hub adapter, they use the wrong class_names (the local ViT training classes) instead of the hub's CLIP-identified classes, resulting in displaying "goldfish" instead of "car" to users.

The root cause is that the system uses class_names.txt (the 50 ViT training classes) as candidate labels for CLIP zero-shot classification AND as the source for pseudo_labels sent to the hub AND for displaying inference results. But CLIP should be identifying ANY real-world object, not just the 50 classes the ViT was trained on, and the complete lifecycle should properly propagate these CLIP-identified class names from edge → hub → back to edge.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN the camera sees a real-world object (e.g., "car") that was NOT in the ViT training set THEN the ViT model is uncertain (low confidence)

1.2 WHEN the ViT is uncertain THEN CLIP performs zero-shot classification using candidate_labels from class_names.txt (the 50 ViT training classes)

1.3 WHEN CLIP tries to identify a "car" using only the 50 ViT training classes as candidates THEN it returns the closest match from those 50 classes (e.g., "goldfish" or "salamander") instead of "car"

1.4 WHEN the edge node sends an ingress update to the hub with trigger="escalate_hub" THEN it sends the wrong class name (e.g., "goldfish") as clip_pseudo_label instead of the actual object seen (e.g., "car")

1.5 WHEN the hub receives ingress updates with wrong class names THEN it trains on garbage data (e.g., training on "goldfish" when the camera actually saw a "car")

1.6 WHEN the hub accumulates training data from multiple cameras seeing real-world objects THEN it ends up with 47 unique classes (the actual objects seen) that don't match the 50 ViT training classes expected by edge nodes

1.7 WHEN the hub trains a projection layer on 47 unique CLIP-identified classes THEN it does NOT include the class_names list in the adapter metadata (or edge nodes fail to extract it)

1.8 WHEN the edge node receives a hub adapter with 47 classes (where 47 ≠ 50) THEN it fails to load the adapter due to shape mismatch error (RuntimeError: size mismatch)

1.9 WHEN the edge node attempts to load a hub adapter with different number of classes THEN it tries to load the state_dict into the existing projection layer without recreating it with the correct shape

1.10 WHEN the edge node performs inference using a hub projection layer (if it loads) THEN it uses the local ViT class_names from class_names.txt instead of the hub's CLIP-identified class names

1.11 WHEN the edge node displays inference results to the user THEN it shows a ViT training class name (e.g., "goldfish") instead of the actual CLIP-identified object (e.g., "car")

### Expected Behavior (Correct)

2.1 WHEN the camera sees a real-world object (e.g., "car") that was NOT in the ViT training set THEN the ViT model SHALL be uncertain (low confidence)

2.2 WHEN the ViT is uncertain THEN CLIP SHALL perform zero-shot classification using a broad set of real-world object categories (e.g., "car", "person", "building", "tree", "dog", "cat", etc.) NOT limited to the 50 ViT training classes

2.3 WHEN CLIP identifies a real-world object THEN it SHALL return the actual object name (e.g., "car") from the broad candidate set

2.4 WHEN the edge node sends an ingress update to the hub with trigger="escalate_hub" THEN it SHALL send the CLIP-identified object name (e.g., "car") as clip_pseudo_label

2.5 WHEN the hub receives ingress updates with CLIP-identified object names THEN it SHALL train on the actual objects seen by cameras (e.g., "car", "person", "building")

2.6 WHEN the hub accumulates training data from multiple cameras THEN it SHALL have N unique classes where N is the number of distinct real-world objects seen by cameras (e.g., 47 classes if cameras have seen 47 different object types)

2.7 WHEN the hub trains a projection layer on N unique CLIP-identified classes THEN it SHALL include the class_names list in the adapter metadata

2.8 WHEN the edge node receives a hub adapter with N classes (where N ≠ 50) THEN it SHALL successfully load the adapter by creating a new projection layer with shape (512 → N)

2.9 WHEN the edge node loads a hub adapter with class_names metadata THEN it SHALL store these class_names for use in inference

2.10 WHEN the edge node performs inference using the hub projection layer THEN it SHALL use the hub's class_names (not the local ViT class_names) to label the output

2.11 WHEN the edge node displays inference results to the user THEN it SHALL show the CLIP-identified class name from the hub adapter (e.g., "car") not a ViT training class (e.g., "goldfish")

### Unchanged Behavior (Regression Prevention)

3.1 WHEN the ViT model has high confidence (>= known_threshold) on an object from its training set THEN it SHALL CONTINUE TO classify using the ViT model without invoking CLIP

3.2 WHEN the edge node performs local adaptation (trigger="adapt_local") THEN it SHALL CONTINUE TO fine-tune the LoRA adapter and send adapter weights to the hub

3.3 WHEN the edge node sends ingress updates to the hub THEN it SHALL CONTINUE TO include all required fields (embedding, metadata, trigger type, etc.)

3.4 WHEN the hub receives ingress updates with trigger="adapt_local" THEN it SHALL CONTINUE TO process adapter weights through FedAvg

3.5 WHEN the hub receives ingress updates with trigger="escalate_hub" THEN it SHALL CONTINUE TO store embeddings in FAISS, cluster them, and trigger retraining when clusters reach minimum size

3.6 WHEN the hub trains on accumulated embeddings THEN it SHALL CONTINUE TO create a projection layer that maps CLIP embeddings to class logits

3.7 WHEN edge nodes sync adapters from the hub THEN they SHALL CONTINUE TO download and apply the global adapter for inference
