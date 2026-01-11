"""
YOLO-based wildlife detection module.

Uses YOLOv8 from Ultralytics to detect wildlife subjects in images.
Supports batch inference for efficient processing.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from .utils import load_image_safely


@dataclass
class BoundingBox:
    """Represents a detected object's bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str

    def area(self) -> int:
        """Calculate bounding box area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def width(self) -> int:
        """Get bounding box width."""
        return self.x2 - self.x1

    def height(self) -> int:
        """Get bounding box height."""
        return self.y2 - self.y1


@dataclass
class DetectionResult:
    """Results from YOLO detection on a single image."""
    image_path: str
    boxes: List[BoundingBox]
    has_detection: bool
    inference_time: float

    def get_largest_box(self) -> Optional[BoundingBox]:
        """Get the largest bounding box by area."""
        if not self.boxes:
            return None
        return max(self.boxes, key=lambda b: b.area())


class WildlifeDetector:
    """YOLO-based detector for wildlife subjects."""

    # COCO dataset class names
    COCO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
        35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
        39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
        45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
        50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
        55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
        60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
        65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
        70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
        75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }

    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        """
        Initialize wildlife detector.

        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger('BlurDetection.Detector')

        self.model = None
        self.device = self._determine_device()
        self.target_classes = config['yolo']['target_classes']

        self.logger.info(f"Detector initialized - Device: {self.device}")
        self.logger.info(f"Target classes: {self._get_target_class_names()}")

    def _determine_device(self) -> str:
        """Determine which device to use for inference."""
        device_config = self.config['yolo']['device']

        if device_config == 'auto':
            # Auto-detect best available device
            try:
                import torch
                if torch.cuda.is_available():
                    self.logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
                    self.logger.info(f"CUDA version: {torch.version.cuda}")
                    return 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.logger.info("Apple MPS GPU detected")
                    return 'mps'
                else:
                    self.logger.warning("No GPU detected, using CPU (slower)")
                    return 'cpu'
            except ImportError:
                self.logger.warning("PyTorch not available, using CPU")
                return 'cpu'
        else:
            return device_config

    def _get_target_class_names(self) -> List[str]:
        """Get names of target wildlife classes."""
        return [self.COCO_CLASSES.get(class_id, f"class_{class_id}")
                for class_id in self.target_classes]

    def load_model(self) -> None:
        """Load YOLO model."""
        model_path = self.config['yolo']['model']
        self.logger.info(f"Loading YOLO model: {model_path}")

        start_time = time.time()

        try:
            # Load model (auto-downloads if not found and auto_download=True)
            self.model = YOLO(model_path)

            # Set device
            self.model.to(self.device)

            load_time = time.time() - start_time
            self.logger.info(f"Model loaded successfully in {load_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            if not Path(model_path).exists() and not self.config['yolo']['auto_download']:
                self.logger.error(
                    "Model file not found. Set 'auto_download: true' in config "
                    "or download manually from Ultralytics."
                )
            raise

    def detect_batch(self, image_paths: List[str]) -> List[DetectionResult]:
        """
        Run batch detection on multiple images.

        Args:
            image_paths: List of paths to images

        Returns:
            List of DetectionResult objects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.logger.debug(f"Running batch detection on {len(image_paths)} images")
        start_time = time.time()

        try:
            # Pre-load images (supports RAW files via load_image_safely)
            loaded_images = []
            for path in image_paths:
                img = load_image_safely(path, self.logger)
                if img is not None:
                    loaded_images.append(img)
                else:
                    # If image fails to load, append None and handle later
                    loaded_images.append(None)

            # Run batch inference on loaded images
            results = self.model.predict(
                source=loaded_images,
                conf=self.config['yolo']['confidence_threshold'],
                iou=self.config['yolo']['iou_threshold'],
                max_det=self.config['yolo']['max_detections'],
                imgsz=self.config['yolo']['image_size'],
                verbose=False,
                stream=False
            )

            # Process results
            detection_results = []
            for image_path, result in zip(image_paths, results):
                det_result = self._process_single_result(image_path, result)
                detection_results.append(det_result)

            batch_time = time.time() - start_time
            self.logger.debug(
                f"Batch detection completed in {batch_time:.2f}s "
                f"({len(image_paths)/batch_time:.1f} imgs/sec)"
            )

            return detection_results

        except Exception as e:
            self.logger.error(f"Batch detection failed: {e}")
            # Return empty results for all images
            return [
                DetectionResult(path, [], False, 0.0)
                for path in image_paths
            ]

    def detect_single(self, image_path: str) -> DetectionResult:
        """
        Run detection on a single image.

        Args:
            image_path: Path to the image

        Returns:
            DetectionResult object
        """
        results = self.detect_batch([image_path])
        return results[0]

    def _process_single_result(self, image_path: str, result) -> DetectionResult:
        """
        Process YOLO result for a single image.

        Args:
            image_path: Path to the image
            result: YOLO result object

        Returns:
            DetectionResult object
        """
        start_time = time.time()

        # Extract bounding boxes
        boxes = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                # Get box coordinates and metadata
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                # Filter by target classes
                if class_id in self.target_classes:
                    class_name = self.COCO_CLASSES.get(class_id, f"class_{class_id}")

                    bbox = BoundingBox(
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name
                    )
                    boxes.append(bbox)

        inference_time = time.time() - start_time

        return DetectionResult(
            image_path=image_path,
            boxes=boxes,
            has_detection=len(boxes) > 0,
            inference_time=inference_time
        )

    def filter_by_class(self, detection: DetectionResult,
                       class_ids: Optional[List[int]] = None) -> DetectionResult:
        """
        Filter detection results to keep only specified classes.

        Args:
            detection: DetectionResult to filter
            class_ids: List of class IDs to keep (default: target_classes)

        Returns:
            New DetectionResult with filtered boxes
        """
        if class_ids is None:
            class_ids = self.target_classes

        filtered_boxes = [box for box in detection.boxes if box.class_id in class_ids]

        return DetectionResult(
            image_path=detection.image_path,
            boxes=filtered_boxes,
            has_detection=len(filtered_boxes) > 0,
            inference_time=detection.inference_time
        )

    def get_largest_box(self, boxes: List[BoundingBox]) -> Optional[BoundingBox]:
        """
        Get the largest bounding box by area.

        Args:
            boxes: List of BoundingBox objects

        Returns:
            Largest BoundingBox, or None if list is empty
        """
        if not boxes:
            return None
        return max(boxes, key=lambda b: b.area())

    def get_detection_summary(self, detection: DetectionResult) -> str:
        """
        Get a human-readable summary of detection results.

        Args:
            detection: DetectionResult object

        Returns:
            Summary string
        """
        if not detection.has_detection:
            return "No wildlife detected"

        class_counts = {}
        for box in detection.boxes:
            class_counts[box.class_name] = class_counts.get(box.class_name, 0) + 1

        summary_parts = [f"{count} {name}(s)" for name, count in class_counts.items()]
        return "Detected: " + ", ".join(summary_parts)
