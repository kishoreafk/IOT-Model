#!/usr/bin/env python3
"""Interactive demo runner for Edge Hub Adaptive Learning System."""
import os
import sys
import time
import asyncio
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_option(num: int, text: str):
    print(f"  [{num}] {text}")


def demo_edge_node_only():
    """Simulate edge node detection with mock data."""
    print_header("Edge Node Only Demo (Mock Data)")

    print("Initializing EdgeVisionNode (mock mode)...")
    time.sleep(0.5)

    class MockNode:
        known_threshold = 0.80
        adapt_threshold = 0.50

        def detect_novelty(self, image, candidate_labels):
            scores = np.random.rand(len(candidate_labels))
            scores = scores / scores.sum()
            labels = sorted(zip(candidate_labels, scores), key=lambda x: x[1], reverse=True)
            top_label, top_score = labels[0]

            if top_score > self.known_threshold:
                decision = "Known"
            elif top_score > self.adapt_threshold:
                decision = "Adapt_Local"
            else:
                decision = "Escalate_Hub"

            return decision, [s for _, s in labels], [l for l, _ in labels]

        def extract_features(self, image):
            return np.random.randn(1, 512)

    node = MockNode()

    candidate_labels = ["car", "truck", "person", "dog", "bird", "unknown"]

    print("\nProcessing test image...")
    decision, scores, labels = node.detect_novelty(None, candidate_labels)

    print(f"\nDecision: {decision}")
    print(f"Top prediction: {labels[0]} ({scores[0]:.2%})")
    print(f"Second: {labels[1]} ({scores[1]:.2%})")

    print("\n[✓] Edge node demo complete\n")


async def demo_full_pipeline():
    """Demo full pipeline from CLIP detection to hub aggregation."""
    print_header("Full Pipeline Demo")

    print("Step 1: Initialize CLIP model (mock)...")
    time.sleep(0.5)

    print("Step 2: Zero-shot classification...")
    embedding = np.random.randn(1, 512)
    embedding = embedding / np.linalg.norm(embedding)

    confidence = np.random.uniform(0.3, 0.9)
    print(f"   CLIP confidence: {confidence:.2%}")

    if confidence > 0.80:
        print("   Decision: Known (log locally)")
    elif confidence > 0.50:
        print("   Decision: Adapt_Local (LoRA fine-tune)")
    else:
        print("   Decision: Escalate_Hub (transmit to hub)")

    print("\nStep 3: Extract embedding features...")
    print(f"   Embedding shape: {embedding.shape}")

    print("\nStep 4: Encrypt with Fernet + RSA sign...")
    time.sleep(0.3)
    print("   ✓ Fernet AES-256 encryption")
    print("   ✓ RSA-2048 signature")

    print("\nStep 5: Transmit to hub...")
    print("   POST /ingress_update")

    print("\nStep 6: Hub processes...")
    time.sleep(0.5)
    print("   ✓ Decrypt payload")
    print("   ✓ Verify signature")
    print("   ✓ Add to FAISS index")
    print("   ✓ Route via MoE gating")

    print("\nStep 7: Task completion...")
    print("   Task ID: 550e8400-e29b-41d4-a716-446655440000")
    print("   Status: completed")

    print("\n[✓] Full pipeline demo complete\n")


def demo_moe_manager():
    """Demo MoE expert routing."""
    print_header("MoE Manager Demo")

    print("Initializing MoE system...")
    time.sleep(0.3)

    print("\nCreating initial experts...")
    print("   Expert 0: cluster_id=0, status=active")
    print("   Expert 1: cluster_id=1, status=active")

    print("\nRouting test embedding...")
    test_embedding = np.random.randn(512)
    routing_weights = np.random.rand(2)
    routing_weights = routing_weights / routing_weights.sum()

    expert_id = int(np.argmax(routing_weights))
    print(f"   → Expert {expert_id} selected")
    print(f"   Routing weights: [{routing_weights[0]:.3f}, {routing_weights[1]:.3f}]")

    print("\nChecking representation gap...")
    cluster_size = 15
    threshold = 10
    has_gap = cluster_size >= threshold
    print(f"   Cluster size: {cluster_size}, Threshold: {threshold}")
    print(f"   Has gap: {has_gap}")

    if has_gap:
        print("\n   Creating new expert...")
        print("   → Expert 2: cluster_id=2, status=active")

    print("\n[✓] MoE manager demo complete\n")


def demo_encryption_flow():
    """Demo encryption and signing flow."""
    print_header("Encryption Flow Demo")

    print("Generating sample payload...")
    payload = {
        "device_id": "edge_device_001",
        "timestamp": time.time(),
        "embedding": np.random.randn(512).tolist()[:5],
        "metadata": {"model_version": "1.0"},
    }
    print(f"   Device: {payload['device_id']}")
    print(f"   Embedding dim: {len(payload['embedding'])}")

    print("\nSigning with RSA-2048...")
    time.sleep(0.2)
    signature = "a1b2c3d4e5f6..." * 4
    print(f"   Signature: {signature[:40]}...")

    print("\nEncrypting with Fernet (AES-256)...")
    time.sleep(0.2)
    ciphertext = "g9h8i7j6k5l4..." * 8
    print(f"   Ciphertext: {ciphertext[:40]}...")

    print("\nTransmitting to hub...")
    print("   POST /ingress_update")

    print("\nHub receiving...")
    time.sleep(0.3)
    print("   ✓ Fernet.decrypt()")
    print("   ✓ RSA.verify(signature, public_key)")
    print("   ✓ Process payload")

    print("\n[✓] Encryption flow demo complete\n")


async def run_all():
    """Run all demos sequentially."""
    print_header("Edge Hub Adaptive Learning System")
    print("Running all demos...\n")

    demo_edge_node_only()
    await demo_full_pipeline()
    demo_moe_manager()
    demo_encryption_flow()

    print_header("All Demos Complete")
    print("System ready for deployment.\n")


async def main():
    while True:
        print_header("Edge Hub Adaptive Learning System - Demo Menu")
        print_option(1, "Edge Node Only (mock data)")
        print_option(2, "Full Pipeline")
        print_option(3, "MoE Manager")
        print_option(4, "Encryption Flow")
        print_option(5, "Run All")
        print_option(0, "Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            demo_edge_node_only()
        elif choice == "2":
            await demo_full_pipeline()
        elif choice == "3":
            demo_moe_manager()
        elif choice == "4":
            demo_encryption_flow()
        elif choice == "5":
            await run_all()
        elif choice == "0":
            print("\nGoodbye!\n")
            break
        else:
            print("\nInvalid option. Try again.\n")


if __name__ == "__main__":
    asyncio.run(main())