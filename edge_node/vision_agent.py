import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor, ViTForImageClassification

logger = logging.getLogger(__name__)


class EdgeVisionNode:
    """Edge device vision agent with CLIP zero-shot detection and LoRA adaptation."""

    def __init__(
        self,
        device: str = "auto",
        use_fp16: bool = True,
        config_path: str = "configs/model_config.yaml",
        lora_adapter_path: Optional[str] = None,
    ):
        self.device = self._resolve_device(device)
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.config = self._load_config(config_path)
        self.lora_adapter_path = lora_adapter_path

        self.clip_model = None
        self.clip_processor = None
        self.custom_vit = None
        self.lora_model = None
        self.hub_projection = None  # Projection layer from hub training
        self.hub_projection_classes = None  # Class names from hub for projection layer

        self._init_clip()
        self._init_custom_vit()
        if lora_adapter_path and os.path.exists(lora_adapter_path):
            self._load_adapter()

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        return {}

    def _init_clip(self):
        """Initialize CLIP model for zero-shot classification."""
        clip_config = self.config.get("clip", {})
        model_name = clip_config.get("model_name", "openai/clip-vit-base-patch32")

        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)

        if self.use_fp16:
            self.clip_model = self.clip_model.half()

        self.clip_model.eval()

    def _init_custom_vit(self):
        """Initialize custom ViT model using HuggingFace transformers."""
        vit_config = self.config.get("custom_vit", {})
        if not vit_config.get("enabled", True):
            return

        vit_thresholds = vit_config.get("thresholds", {})
        self.known_threshold = vit_thresholds.get("known", 0.85)
        self.adapt_threshold = vit_thresholds.get("adapt", 0.60)

        architecture = vit_config.get("architecture", "vit_base_patch16_224")
        num_classes = vit_config.get("num_classes", 50)
        weights_path = vit_config.get("weights_path", "model/best_vit_model.pth")

        try:
            self.custom_vit = ViTForImageClassification.from_pretrained(
                architecture,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        except Exception:
            self.custom_vit = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )

        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("module."):
                        k = k[7:]
                    new_state_dict[k] = v
                self.custom_vit.load_state_dict(new_state_dict, strict=False)
                print(f"[EdgeVisionNode] Loaded custom weights from {weights_path}")
            except Exception as e:
                print(f"[EdgeVisionNode] Warning: Could not load weights: {e}")

        self.custom_vit = self.custom_vit.to(self.device)

        if self.use_fp16:
            self.custom_vit = self.custom_vit.half()

        self.custom_vit.eval()

    def _load_adapter(self):
        """Load pre-trained LoRA adapter weights."""
        if not self.lora_model and self.custom_vit:
            try:
                self.lora_model = PeftModel.from_pretrained(
                    self.custom_vit, os.path.dirname(self.lora_adapter_path)
                )
                self.lora_model.eval()
            except Exception as e:
                print(f"Warning: Could not load adapter: {e}")

    def run_inference(
        self,
        image: Image.Image,
        candidate_labels: Optional[List[str]] = None,
    ) -> Tuple[str, List[float], List[str], Optional[str]]:
        """
        Run inference on custom ViT model first.
        If hub_projection is available (from hub training), use it on CLIP embeddings.
        If ViT is uncertain, use CLIP zero-shot to get pseudo-labels for fine-tuning.

        Returns:
            decision: One of 'Known', 'Adapt_Local', 'Escalate_Hub'
            scores: Confidence scores for each label
            labels: Sorted labels by confidence
            pseudo_label: The label suggested by CLIP when ViT is uncertain (None otherwise)
        """
        if candidate_labels is None:
            candidate_labels = self._get_default_labels()

        # If hub_projection exists (from hub), apply it on CLIP embeddings
        if hasattr(self, 'hub_projection') and self.hub_projection is not None:
            try:
                clip_embedding = self.extract_features(image)
                clip_embedding = clip_embedding.unsqueeze(0).to(self.device)
                
                # Use hub's class names if available, otherwise fall back to local ones
                projection_classes = (
                    self.hub_projection_classes 
                    if self.hub_projection_classes 
                    else candidate_labels
                )
                
                with torch.no_grad():
                    if isinstance(self.hub_projection, dict):
                        # Legacy/fallback behavior
                        proj_layer = nn.Linear(512, len(projection_classes)).to(self.device)
                        proj_layer.load_state_dict(self.hub_projection, strict=False)
                        if self.use_fp16:
                            proj_layer = proj_layer.half()
                        proj_layer.eval()
                        self.hub_projection = proj_layer  # Cache it
                    else:
                        proj_layer = self.hub_projection
                        
                    if self.use_fp16:
                        clip_embedding = clip_embedding.half()
                        
                    logits = proj_layer(clip_embedding)
                    probs = torch.softmax(logits, dim=-1)
                    
                    top_prob, top_idx = probs.max(dim=-1)
                    proj_confidence = top_prob.item()
                    proj_label_idx = top_idx.item()
                    
                    if proj_label_idx < len(projection_classes):
                        proj_label = projection_classes[proj_label_idx]
                    else:
                        proj_label = f"class_{proj_label_idx}"
                    
                    logger.warning(
                        f"[Hub Projection] {proj_label} ({proj_confidence:.2%}) "
                        f"using hub-trained adapter (classes: {len(projection_classes)})"
                    )
                    
                    # Use hub projection result
                    if proj_confidence > 0.5:
                        return "Known", [proj_confidence], [proj_label], proj_label
                    else:
                        return "Escalate_Hub", [proj_confidence], [proj_label], proj_label
            except Exception as e:
                logger.debug(f"Hub projection inference failed: {e}")

        # Standard ViT inference
        model = self.lora_model if self.lora_model else self.custom_vit
        if model is None:
            raise ValueError("No model available for inference")

        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        img_tensor = transform(image).unsqueeze(0).to(self.device)

        if self.use_fp16:
            img_tensor = img_tensor.half()

        with torch.no_grad():
            outputs = model(img_tensor)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
            probs = torch.softmax(logits, dim=-1)

            top_prob, top_idx = probs.max(dim=-1)
            vit_confidence = top_prob.item()
            vit_label_idx = top_idx.item()

        class_names = self._get_default_labels()
        if vit_label_idx < len(class_names):
            vit_label = class_names[vit_label_idx]
        else:
            vit_label = f"class_{vit_label_idx}"

        vit_scores = probs[0].cpu().tolist()
        vit_labels = class_names[: len(vit_scores)]

        pseudo_label = None

        if vit_confidence >= self.known_threshold:
            decision = "Known"
        elif vit_confidence >= self.adapt_threshold:
            decision = "Adapt_Local"
            # Use CLIP with BROAD real-world categories, not limited ViT classes
            broad_categories = self._get_broad_clip_categories()
            pseudo_label = self._get_clip_zero_shot_label(image, broad_categories)
        else:
            decision = "Escalate_Hub"
            # Use CLIP with BROAD real-world categories, not limited ViT classes
            broad_categories = self._get_broad_clip_categories()
            pseudo_label = self._get_clip_zero_shot_label(image, broad_categories)

        return decision, vit_scores, vit_labels, pseudo_label

    def _get_clip_zero_shot_label(
        self,
        image: Image.Image,
        candidate_labels: List[str],
    ) -> str:
        """
        Use CLIP for zero-shot classification when ViT is uncertain.
        Returns the top predicted label from CLIP.
        """
        text_inputs = self.clip_processor(
            text=candidate_labels,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            image_inputs = self.clip_processor(
                images=image,
                return_tensors="pt",
            ).to(self.device)

            image_features = self.clip_model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.T
            probs = logits.softmax(dim=-1)

            top_prob, top_idx = probs.max(dim=-1)
            top_label = candidate_labels[top_idx.item()]

        return top_label

    def detect_novelty(
        self,
        image: Image.Image,
        candidate_labels: Optional[List[str]] = None,
    ) -> Tuple[str, List[float], List[str]]:
        """
        Legacy method - runs combined inference.
        Now calls run_inference() internally.

        Returns:
            decision: One of 'Known', 'Adapt_Local', 'Escalate_Hub'
            scores: Confidence scores for each label
            labels: Sorted labels by confidence
        """
        decision, scores, labels, _ = self.run_inference(image, candidate_labels)
        return decision, scores, labels

    def _get_default_labels(self) -> List[str]:
        """Load default class labels from config (ViT training classes)."""
        class_names_path = "configs/class_names.txt"
        if os.path.exists(class_names_path):
            with open(class_names_path, "r") as f:
                return [line.strip() for line in f.readlines()]
        # Fallback: Tiny ImageNet 50 classes matching ViT training
        return [
            "goldfish", "salamander", "bullfrog", "toad", "alligator",
            "boa", "trilobite", "scorpion", "spider", "tarantula",
            "centipede", "goose", "koala", "jellyfish", "coral",
            "snail", "slug", "nudibranch", "lobster", "crayfish",
            "stork", "penguin", "albatross", "dugong", "chihuahua",
            "terrier", "retriever", "retriever", "shepherd", "poodle",
            "tabby", "persian", "cat", "cougar", "lion",
            "bear", "ladybug", "fly", "bee", "grasshopper",
            "stick_insect", "cockroach", "mantis", "dragonfly", "butterfly",
            "butterfly", "cucumber", "guinea_pig", "pig", "ox"
        ]

    def _get_broad_clip_categories(self) -> List[str]:
        """Load broad real-world categories for CLIP zero-shot classification."""
        broad_categories_path = "configs/broad_categories.txt"
        if os.path.exists(broad_categories_path):
            with open(broad_categories_path, "r") as f:
                categories = [line.strip() for line in f.readlines() 
                            if line.strip() and not line.strip().startswith('#')]
            return categories
        # Fallback broad categories if file not found
        return [
            "person", "car", "truck", "bus", "motorcycle", "bicycle",
            "building", "road", "tree", "dog", "cat", "phone",
            "laptop", "chair", "table", "door", "window", "food"
        ]

    def extract_features(self, image: Image.Image) -> torch.Tensor:
        """
        Extract CLIP embedding features from image for hub transmission.
        CLIP provides 512-dim embeddings that match FAISS index.
        """
        inputs = self.clip_processor(
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.squeeze()

    def local_adaptation(
        self,
        image: Image.Image,
        pseudo_label: str,
        num_epochs: int = 5,
    ):
        """
        Perform local LoRA fine-tuning on the image.

        Args:
            image: Input PIL Image
            pseudo_label: Label to use for pseudo-labeling
            num_epochs: Number of training epochs
        """
        if self.custom_vit is None:
            raise ValueError("Custom ViT not initialized")

        lora_config = self.config.get("lora", {})
        target_modules = lora_config.get(
            "target_modules", ["query", "key", "value", "dense"]
        )

        peft_config = LoraConfig(
            r=lora_config.get("r", 8),
            lora_alpha=lora_config.get("alpha", 16),
            target_modules=target_modules,
            lora_dropout=lora_config.get("dropout", 0.1),
            bias=lora_config.get("bias", "none"),
            task_type=lora_config.get("task_type", "IMAGE_CLS"),
        )

        if self.lora_model is None:
            self.lora_model = get_peft_model(self.custom_vit, peft_config)
        else:
            self.lora_model.train()

        self.lora_model.train()

        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        img_tensor = transform(image).unsqueeze(0).to(self.device)

        if self.use_fp16:
            img_tensor = img_tensor.half()

        optimizer = torch.optim.Adam(self.lora_model.parameters(), lr=1e-4)

        # Resolve pseudo_label to a class index
        class_names = self._get_default_labels()
        target_idx = 0  # fallback
        if pseudo_label:
            pseudo_lower = pseudo_label.lower()
            for i, name in enumerate(class_names):
                if pseudo_lower in name.lower() or name.lower() in pseudo_lower:
                    target_idx = i
                    break

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self.lora_model(img_tensor)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
                target = torch.tensor([target_idx]).to(self.device)
                loss = nn.functional.cross_entropy(logits, target)
            else:
                target = torch.tensor([target_idx]).to(self.device)
                loss = nn.functional.cross_entropy(outputs, target)
            loss.backward()
            optimizer.step()

        self.lora_model.eval()

    def _save_adapter_weights(self):
        """Save LoRA adapter weights to file."""
        if self.lora_model is None:
            return

        os.makedirs(os.path.dirname(self.lora_adapter_path), exist_ok=True)
        self.lora_model.save_pretrained(os.path.dirname(self.lora_adapter_path))

        adapter_file = os.path.join(
            os.path.dirname(self.lora_adapter_path), "adapter_model.bin"
        )
        if os.path.exists(adapter_file) and self.lora_adapter_path != adapter_file:
            import shutil

            shutil.copy(adapter_file, self.lora_adapter_path)

    def get_adapter_weights(self) -> Optional[Dict[str, Any]]:
        """Get current adapter weights as dictionary."""
        if self.lora_model is None:
            return None
        return self.lora_model.state_dict()

    def classify_image(self, image: Image.Image) -> Tuple[str, float]:
        """
        Classify image using custom ViT or LoRA-adapted model.

        Returns:
            label: Predicted class label
            confidence: Prediction confidence
        """
        model = self.lora_model if self.lora_model else self.custom_vit
        if model is None:
            raise ValueError("No model available for classification")

        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        img_tensor = transform(image).unsqueeze(0).to(self.device)

        if self.use_fp16:
            img_tensor = img_tensor.half()

        with torch.no_grad():
            outputs = model(img_tensor)
            # Handle both timm and transformers output formats
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
            probs = torch.softmax(logits, dim=-1)
            top_prob, top_idx = probs.max(dim=-1)

        class_names = self._get_default_labels()
        label = (
            class_names[top_idx.item()]
            if top_idx.item() < len(class_names)
            else f"class_{top_idx.item()}"
        )
        confidence = top_prob.item()

        return label, confidence
