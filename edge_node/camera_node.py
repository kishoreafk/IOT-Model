"""
camera_node.py
─────────────
Live webcam inference loop for EdgeVisionNode.
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import asyncio
import logging
import time
import uuid

from edge_node.vision_agent import EdgeVisionNode
from edge_node.secure_transmitter import SecureTransmitter
from edge_node.adapter_sync import AdapterSyncClient

logger = logging.getLogger(__name__)


class LiveCameraNode:
    """
    Wraps EdgeVisionNode with a real-time webcam loop.

    Parameters
    ----------
    device_id : str
        Unique identifier for this edge node.
    hub_url : str
        Base URL of the Central Hub.
    camera_index : int
        OpenCV camera index (0 = default webcam).
    inference_interval : float
        Minimum seconds between inference calls.
    candidate_labels : list[str]
        CLIP zero-shot label candidates.
    key_path / private_key_path / public_key_path : str
        Paths to cryptographic keys.
    adapter_poll_interval : int
        Seconds between adapter version checks.
    """

    def __init__(
        self,
        device_id: Optional[str] = None,
        hub_url: str = os.getenv("HUB_URL", "http://localhost:8000"),
        camera_index: int = int(os.getenv("CAMERA_INDEX", "0")),
        inference_interval: float = float(os.getenv("INFERENCE_INTERVAL", "0.5")),
        candidate_labels: Optional[list] = None,
        key_path: str = os.getenv("EDGE_ENCRYPTION_KEY_PATH", "keys/encryption.key"),
        private_key_path: str = os.getenv("EDGE_PRIVATE_KEY_PATH", "keys/private_key.pem"),
        public_key_path: str = os.getenv("EDGE_PUBLIC_KEY_PATH", "keys/public_key.pem"),
        adapter_poll_interval: int = int(os.getenv("ADAPTER_POLL_INTERVAL", "30")),
        device: str = os.getenv("DEVICE", "auto"),
        use_fp16: bool = os.getenv("USE_FP16", "true").lower() == "true",
    ):
        self.device_id = device_id or str(uuid.uuid4())
        self.hub_url = hub_url
        self.camera_index = camera_index
        self.inference_interval = inference_interval
        self.candidate_labels = candidate_labels or [
            "car", "truck", "person", "bicycle", "dog",
            "cat", "chair", "table", "phone", "laptop",
        ]
        self.adapter_poll_interval = adapter_poll_interval
        self._last_inference_time = 0.0
        self._running = False
        self._skip_hub = False

        self.vision_node = EdgeVisionNode(device=device, use_fp16=use_fp16)

        self.transmitter = SecureTransmitter(
            adapter_weights_path=str(
                Path("edge_node/lora_adapter") / f"{self.device_id}_adapter.bin"
            ),
            hub_url=hub_url,
            device_id=self.device_id,
            key_path=key_path,
            private_key_path=private_key_path,
            public_key_path=public_key_path,
        )

        self.sync_client = AdapterSyncClient(
            device_id=self.device_id,
            hub_url=hub_url,
            local_adapter_path=self.transmitter.adapter_weights_path,
            vision_node=self.vision_node,
            poll_interval=adapter_poll_interval,
        )

        logger.info(f"[LiveCameraNode] device_id={self.device_id} hub={hub_url}")

    def start(self):
        """Start the camera loop (blocks until stopped or camera error)."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                return loop.create_task(self._run())
        except RuntimeError:
            pass
        
        asyncio.run(self._run())

    def stop(self):
        self._running = False
        logger.info("[LiveCameraNode] Stop requested.")

    async def _run(self):
        if not self._skip_hub:
            print("[LiveCameraNode] Attempting to register with hub...")
            try:
                await asyncio.wait_for(self._register_with_hub(), timeout=5.0)
            except asyncio.TimeoutError:
                print("[LiveCameraNode] Hub registration timeout, continuing anyway...")
            except Exception as e:
                print(f"[LiveCameraNode] Hub registration failed: {e}")
        else:
            print("[LiveCameraNode] Skipping hub registration (--no-hub mode)")

        print(f"[LiveCameraNode] Opening camera {self.camera_index}...")
        
        # Try multiple backends
        cap = None
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(self.camera_index, backend)
            if cap.isOpened():
                print(f"[LiveCameraNode] Camera opened with backend: {backend}")
                break
            if cap:
                cap.release()
        
        if not cap or not cap.isOpened():
            print(f"[ERROR] Cannot open camera index {self.camera_index}")
            print("Available camera indices to try: 0, 1, 2")
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"[LiveCameraNode] Camera {self.camera_index} opened successfully!")
        print("[LiveCameraNode] Press 'q' in the window to quit")

        self._running = True
        sync_task = None
        if not self._skip_hub:
            sync_task = asyncio.create_task(self.sync_client.poll_loop())

        window_name = f"Edge Node [{self.device_id[:8]}]"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            frame_count = 0
            while self._running:
                ret, bgr_frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.1)
                    continue

                frame_count += 1
                if frame_count == 1:
                    print(f"[LiveCameraNode] First frame captured! Resolution: {bgr_frame.shape[1]}x{bgr_frame.shape[0]}")

                now = time.monotonic()
                should_run_inference = now - self._last_inference_time >= self.inference_interval
                
                if should_run_inference:
                    self._last_inference_time = now
                    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb)
                    
                    try:
                        decision, scores, labels = self.vision_node.detect_novelty(
                            pil_image, candidate_labels=self.candidate_labels
                        )
                        
                        top_label = labels[0] if labels else "unknown"
                        top_score = scores[0] if scores else 0.0
                        
                        self._draw_hud(bgr_frame, decision, top_label, top_score)
                        print(f"[{decision}] {top_label} ({top_score:.1%})")
                        
                        if decision == "Adapt_Local" and not self._skip_hub:
                            asyncio.create_task(self._handle_adapt_local(pil_image, top_label))
                        elif decision == "Escalate_Hub" and not self._skip_hub:
                            asyncio.create_task(self._handle_escalate_hub(pil_image))
                    except Exception as e:
                        print(f"[ERROR] Inference failed: {e}")
                
                cv2.imshow(window_name, bgr_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[LiveCameraNode] Quit requested")
                    self.stop()
                    break
                
                await asyncio.sleep(0.001)

        except KeyboardInterrupt:
            print("\n[LiveCameraNode] Interrupted by user")
        except Exception as e:
            print(f"[ERROR] Camera loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False
            cap.release()
            if sync_task:
                sync_task.cancel()
                try:
                    await sync_task
                except asyncio.CancelledError:
                    pass
            cv2.destroyAllWindows()
            await asyncio.sleep(0.1)
            print("[LiveCameraNode] Camera released and windows closed.")

    async def _process_frame(self, pil_image: Image.Image, display_frame: np.ndarray):
        """Run inference and route to correct action."""
        try:
            decision, scores, labels, pseudo_label = self.vision_node.run_inference(
                pil_image, candidate_labels=self.candidate_labels
            )
        except Exception as e:
            logger.error(f"[LiveCameraNode] Inference error: {e}")
            return

        top_label = labels[0] if labels else "unknown"
        top_score = scores[0] if scores else 0.0

        self._draw_hud(display_frame, decision, top_label, top_score)
        cv2.imshow(f"Edge Node [{self.device_id[:8]}]", display_frame)

        logger.debug(f"[{decision}] {top_label} ({top_score:.2%})")

        if decision == "Adapt_Local":
            clip_pseudo_label = pseudo_label if pseudo_label else top_label
            asyncio.create_task(self._handle_adapt_local(pil_image, clip_pseudo_label))
        elif decision == "Escalate_Hub":
            clip_pseudo_label = pseudo_label if pseudo_label else top_label
            asyncio.create_task(self._handle_escalate_hub(pil_image, clip_pseudo_label))

    async def _handle_adapt_local(self, image: Image.Image, pseudo_label: str):
        """LoRA fine-tune on the edge device, then push adapter weights to hub."""
        logger.info(f"[Adapt_Local] Fine-tuning locally for label='{pseudo_label}'")
        try:
            self.vision_node.local_adaptation(image, pseudo_label=pseudo_label)
            self.vision_node._save_adapter_weights()
        except Exception as e:
            logger.error(f"[Adapt_Local] LoRA training failed: {e}")
            return

        clip_embedding = self.vision_node.extract_features(image)

        result = await self.transmitter.transmit(
            clip_embedding,
            metadata={
                "trigger": "adapt_local",
                "pseudo_label": pseudo_label,
                "adapter_version": self.sync_client.local_version,
            },
            sign_payload=True,
        )

        if result.get("success"):
            task_id = result["hub_response"].get("task_id")
            logger.info(f"[Adapt_Local] Adapter sent to hub. task_id={task_id}")
        else:
            logger.warning(f"[Adapt_Local] Hub transmission failed: {result}")

    async def _handle_escalate_hub(self, image: Image.Image, clip_pseudo_label: str = None):
        """Extract ViT embedding and send to hub for clustering and retraining."""
        logger.info("[Escalate_Hub] Sending embedding to hub for retraining…")
        try:
            vit_embedding = self.vision_node.extract_features(image)
        except Exception as e:
            logger.error(f"[Escalate_Hub] Embedding extraction failed: {e}")
            return

        result = await self.transmitter.transmit(
            vit_embedding,
            metadata={
                "trigger": "escalate_hub",
                "clip_pseudo_label": clip_pseudo_label,
                "adapter_version": self.sync_client.local_version,
            },
            sign_payload=True,
        )

        if result.get("success"):
            task_id = result["hub_response"].get("task_id")
            logger.info(f"[Escalate_Hub] Embedding sent. Hub will retrain. task_id={task_id}")
        else:
            logger.warning(f"[Escalate_Hub] Hub transmission failed: {result}")

    async def _register_with_hub(self):
        """Register this edge device with the hub's AdapterRegistry."""
        import httpx

        payload = {
            "device_id": self.device_id,
            "public_key_path": self.transmitter.public_key_path,
            "adapter_version": 0,
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(
                    f"{self.hub_url}/devices/register", json=payload
                )
                r.raise_for_status()
                logger.info(f"[LiveCameraNode] Registered with hub: {r.json()}")
        except Exception as e:
            logger.warning(f"[LiveCameraNode] Hub registration failed (will retry): {e}")

    @staticmethod
    def _draw_hud(
        frame: np.ndarray, decision: str, label: str, score: float
    ):
        """Overlay inference results on the display frame."""
        color_map = {
            "Known": (0, 200, 0),
            "Adapt_Local": (0, 165, 255),
            "Escalate_Hub": (0, 0, 220),
        }
        color = color_map.get(decision, (200, 200, 200))

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (20, 20, 20), -1)
        cv2.putText(
            frame, f"{decision}  {label} ({score:.1%})",
            (10, 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2,
        )
if __name__ == "__main__":
    node = LiveCameraNode()
    node.run()