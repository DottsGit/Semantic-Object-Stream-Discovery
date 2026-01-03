"""DINOv2 feature extraction module for unsupervised object representation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from loguru import logger

from src.detection.detector import Detection


@dataclass
class ObjectFeature:
    """Feature vector for a detected object."""

    detection: Detection
    embedding: np.ndarray  # DINOv2 embedding vector
    crop_image: np.ndarray | None = None  # Optional: store the cropped image

    @property
    def vector(self) -> np.ndarray:
        """Alias for embedding."""
        return self.embedding


class DINOv2Extractor:
    """Feature extractor using DINOv2 vision transformer."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: str = "cuda",
        batch_size: int = 16,
        image_size: int = 224,
    ):
        """Initialize DINOv2 feature extractor.

        Args:
            model_name: HuggingFace model name for DINOv2
                - facebook/dinov2-small (384-dim)
                - facebook/dinov2-base (768-dim)
                - facebook/dinov2-large (1024-dim)
                - facebook/dinov2-giant (1536-dim)
            device: Device to run on (cuda, cpu, mps)
            batch_size: Batch size for feature extraction
            image_size: Input image size (DINOv2 uses 224 or 518)
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.image_size = image_size

        self._model = None
        self._processor = None
        self._embedding_dim: int | None = None

    def load(self) -> None:
        """Load the DINOv2 model and processor."""
        try:
            from transformers import AutoImageProcessor, AutoModel

            logger.info(f"Loading DINOv2 model: {self.model_name}")

            self._processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=True)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

            # Get embedding dimension from config
            self._embedding_dim = self._model.config.hidden_size
            logger.info(
                f"Loaded DINOv2 with {self._embedding_dim}-dim embeddings on {self.device}"
            )

        except ImportError:
            logger.error("transformers not installed. Run: pip install transformers")
            raise

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is None:
            self.load()
        return self._embedding_dim or 768

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess a crop for DINOv2.

        Args:
            crop: BGR image crop from OpenCV

        Returns:
            RGB image resized for model input
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        resized = cv2.resize(rgb, (self.image_size, self.image_size))

        return resized

    def extract(
        self,
        image: np.ndarray,
        detections: list[Detection],
        store_crops: bool = False,
    ) -> list[ObjectFeature]:
        """Extract DINOv2 features for detected objects.

        Args:
            image: Full frame image (BGR)
            detections: List of detections to extract features for
            store_crops: Whether to store cropped images in results

        Returns:
            List of ObjectFeature with embeddings
        """
        if self._model is None:
            self.load()

        if not detections:
            return []

        # Crop and preprocess all detections
        crops = []
        valid_detections = []

        for det in detections:
            try:
                crop = det.crop_from(image, padding=5)
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue
                crops.append(self._preprocess_crop(crop))
                valid_detections.append((det, crop if store_crops else None))
            except Exception as e:
                logger.warning(f"Failed to crop detection: {e}")
                continue

        if not crops:
            return []

        # Process in batches
        all_embeddings = []

        for i in range(0, len(crops), self.batch_size):
            batch_crops = crops[i : i + self.batch_size]

            # Use HuggingFace processor
            inputs = self._processor(images=batch_crops, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Use CLS token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)

        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)

        # Create ObjectFeature instances
        features = []
        for (det, crop), embedding in zip(valid_detections, all_embeddings):
            features.append(
                ObjectFeature(
                    detection=det,
                    embedding=embedding,
                    crop_image=crop,
                )
            )

        return features

    def extract_single(
        self, image: np.ndarray, detection: Detection, store_crop: bool = False
    ) -> ObjectFeature | None:
        """Extract feature for a single detection.

        Args:
            image: Full frame image
            detection: Single detection
            store_crop: Whether to store the crop

        Returns:
            ObjectFeature or None if extraction fails
        """
        features = self.extract(image, [detection], store_crop)
        return features[0] if features else None


class FeatureBuffer:
    """Buffer for collecting features during warmup phase."""

    def __init__(self, max_size: int = 10000):
        """Initialize feature buffer.

        Args:
            max_size: Maximum number of features to store
        """
        self.max_size = max_size
        self._features: list[ObjectFeature] = []
        self._embeddings: np.ndarray | None = None
        self._dirty = True  # Track if embeddings array needs rebuild

    def add(self, features: list[ObjectFeature]) -> None:
        """Add features to the buffer."""
        self._features.extend(features)

        # Trim if over max size (keep most recent)
        if len(self._features) > self.max_size:
            self._features = self._features[-self.max_size :]

        self._dirty = True

    def add_single(self, feature: ObjectFeature) -> None:
        """Add a single feature."""
        self.add([feature])

    @property
    def embeddings(self) -> np.ndarray:
        """Get all embeddings as a numpy array."""
        if self._dirty or self._embeddings is None:
            if self._features:
                self._embeddings = np.vstack([f.embedding for f in self._features])
            else:
                self._embeddings = np.array([])
            self._dirty = False
        return self._embeddings

    @property
    def features(self) -> list[ObjectFeature]:
        """Get all features."""
        return self._features

    def __len__(self) -> int:
        return len(self._features)

    def clear(self) -> None:
        """Clear the buffer."""
        self._features = []
        self._embeddings = None
        self._dirty = True
