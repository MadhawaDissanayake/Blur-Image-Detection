"""
Sharpness analysis module using Laplacian variance.

Calculates sharpness of image regions (crops) to determine if subjects are
in focus or blurry.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detector import BoundingBox, DetectionResult


@dataclass
class SharpnessResult:
    """Results from sharpness analysis."""
    image_path: str
    laplacian_variance: float
    is_sharp: bool
    strategy_used: str  # 'single', 'largest', 'average', 'best', 'worst'
    num_subjects: int
    crop_size: Optional[Tuple[int, int]] = None


class SharpnessAnalyzer:
    """Analyzes image sharpness using Laplacian variance."""

    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        """
        Initialize sharpness analyzer.

        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger('BlurDetection.Sharpness')

        self.threshold = config['sharpness']['threshold']
        self.multi_subject_strategy = config['sharpness']['multi_subject_strategy']
        self.crop_padding = config['sharpness']['crop_padding']
        self.min_crop_width = config['sharpness']['min_crop_width']
        self.min_crop_height = config['sharpness']['min_crop_height']

        # Percentile-based sharpness detection
        self.method = config['sharpness'].get('method', 'variance')
        self.percentile = config['sharpness'].get('percentile', 95.0)

        self.logger.info(f"Sharpness analyzer initialized - Method: {self.method}")
        self.logger.info(f"Threshold: {self.threshold}")
        if self.method == 'percentile':
            self.logger.info(f"Percentile: {self.percentile}")
        self.logger.info(f"Multi-subject strategy: {self.multi_subject_strategy}")

    def calculate_laplacian_variance(self, image: np.ndarray) -> float:
        """
        Calculate Laplacian-based sharpness metric.

        The Laplacian operator highlights regions of rapid intensity change,
        which correspond to edges. Sharp images have strong edges, while
        blurry images have weak edges.

        Two methods are supported:
        - 'variance': Variance of the Laplacian (traditional method)
        - 'percentile': Percentile of absolute Laplacian values (focuses on sharpest regions)

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Sharpness metric value (higher = sharper)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate Laplacian
        # CV_64F = 64-bit float for precision (prevents overflow)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        if self.method == 'percentile':
            # Use percentile of absolute Laplacian values
            # This focuses on the strongest edges (sharpest parts like eyes, beaks)
            # while ignoring smooth areas (uniform feathers, backgrounds)
            abs_laplacian = np.abs(laplacian)
            percentile_value = np.percentile(abs_laplacian, self.percentile)
            return percentile_value
        else:
            # Traditional variance method
            variance = laplacian.var()
            return variance

    def crop_to_bbox(self, image: np.ndarray, bbox: BoundingBox,
                    padding: Optional[float] = None) -> np.ndarray:
        """
        Crop image to bounding box with optional padding.

        Args:
            image: Input image
            bbox: Bounding box to crop to
            padding: Padding as percentage of box dimension (default: from config)

        Returns:
            Cropped image
        """
        if padding is None:
            padding = self.crop_padding

        height, width = image.shape[:2]

        # Calculate padding in pixels
        bbox_width = bbox.width()
        bbox_height = bbox.height()
        pad_x = int(bbox_width * padding)
        pad_y = int(bbox_height * padding)

        # Apply padding and clip to image bounds
        x1 = max(0, bbox.x1 - pad_x)
        y1 = max(0, bbox.y1 - pad_y)
        x2 = min(width, bbox.x2 + pad_x)
        y2 = min(height, bbox.y2 + pad_y)

        # Crop
        crop = image[y1:y2, x1:x2]

        return crop

    def is_sharp(self, variance: float) -> bool:
        """
        Determine if variance indicates a sharp image.

        Args:
            variance: Laplacian variance value

        Returns:
            True if sharp, False if blurry
        """
        return variance >= self.threshold

    def validate_crop_size(self, crop: np.ndarray) -> bool:
        """
        Check if crop meets minimum size requirements.

        Args:
            crop: Cropped image

        Returns:
            True if crop is large enough, False otherwise
        """
        if crop.size == 0:
            return False

        height, width = crop.shape[:2]

        return (width >= self.min_crop_width and
                height >= self.min_crop_height)

    def analyze_detection(self, image: np.ndarray,
                         detection: DetectionResult) -> SharpnessResult:
        """
        Analyze sharpness for a detection result.

        Args:
            image: Original image (BGR format)
            detection: Detection result with bounding boxes

        Returns:
            SharpnessResult object
        """
        if not detection.has_detection or not detection.boxes:
            # No detection - return default result
            return SharpnessResult(
                image_path=detection.image_path,
                laplacian_variance=0.0,
                is_sharp=False,
                strategy_used='none',
                num_subjects=0,
                crop_size=None
            )

        num_subjects = len(detection.boxes)

        if num_subjects == 1:
            # Single subject - straightforward
            bbox = detection.boxes[0]
            variance, crop_size = self._analyze_single_box(image, bbox)
            strategy = 'single'

        else:
            # Multiple subjects - use strategy
            variance, crop_size = self._analyze_multiple_subjects(
                image, detection.boxes
            )
            strategy = self.multi_subject_strategy

        is_sharp = self.is_sharp(variance)

        return SharpnessResult(
            image_path=detection.image_path,
            laplacian_variance=variance,
            is_sharp=is_sharp,
            strategy_used=strategy,
            num_subjects=num_subjects,
            crop_size=crop_size
        )

    def _analyze_single_box(self, image: np.ndarray,
                           bbox: BoundingBox) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Analyze sharpness for a single bounding box.

        Args:
            image: Original image
            bbox: Bounding box

        Returns:
            Tuple of (variance, crop_size)
        """
        # Crop to bounding box
        crop = self.crop_to_bbox(image, bbox)

        # Validate crop size
        if not self.validate_crop_size(crop):
            self.logger.warning(
                f"Crop too small: {crop.shape[1]}x{crop.shape[0]} "
                f"(min: {self.min_crop_width}x{self.min_crop_height})"
            )
            return 0.0, None

        # Calculate variance
        variance = self.calculate_laplacian_variance(crop)
        crop_size = (crop.shape[1], crop.shape[0])  # (width, height)

        return variance, crop_size

    def _analyze_multiple_subjects(self, image: np.ndarray,
                                   boxes: List[BoundingBox]) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Analyze sharpness for multiple subjects using configured strategy.

        Args:
            image: Original image
            boxes: List of bounding boxes

        Returns:
            Tuple of (variance, crop_size)
        """
        strategy = self.multi_subject_strategy

        if strategy == 'largest':
            # Use largest bounding box
            largest_box = max(boxes, key=lambda b: b.area())
            return self._analyze_single_box(image, largest_box)

        elif strategy in ['average', 'best', 'worst']:
            # Calculate variance for all boxes
            variances = []
            crop_sizes = []

            for box in boxes:
                crop = self.crop_to_bbox(image, box)

                if self.validate_crop_size(crop):
                    variance = self.calculate_laplacian_variance(crop)
                    variances.append(variance)
                    crop_sizes.append((crop.shape[1], crop.shape[0]))

            if not variances:
                return 0.0, None

            if strategy == 'average':
                # Average variance across all subjects
                final_variance = np.mean(variances)
                crop_size = crop_sizes[0]  # Representative size

            elif strategy == 'best':
                # Use sharpest subject (maximum variance)
                max_idx = np.argmax(variances)
                final_variance = variances[max_idx]
                crop_size = crop_sizes[max_idx]

            else:  # worst
                # Use blurriest subject (minimum variance)
                min_idx = np.argmin(variances)
                final_variance = variances[min_idx]
                crop_size = crop_sizes[min_idx]

            return final_variance, crop_size

        else:
            # Fallback to largest if unknown strategy
            self.logger.warning(f"Unknown strategy '{strategy}', using 'largest'")
            largest_box = max(boxes, key=lambda b: b.area())
            return self._analyze_single_box(image, largest_box)

    def get_sharpness_category(self, variance: float) -> str:
        """
        Categorize sharpness level based on variance.

        Args:
            variance: Laplacian variance

        Returns:
            Category string: 'very_sharp', 'sharp', 'acceptable', 'soft', 'blurry', 'very_blurry'
        """
        threshold = self.threshold

        if variance >= threshold * 2:
            return 'very_sharp'
        elif variance >= threshold * 1.3:
            return 'sharp'
        elif variance >= threshold:
            return 'acceptable'
        elif variance >= threshold * 0.7:
            return 'soft'
        elif variance >= threshold * 0.4:
            return 'blurry'
        else:
            return 'very_blurry'

    def get_result_summary(self, result: SharpnessResult) -> str:
        """
        Get human-readable summary of sharpness result.

        Args:
            result: SharpnessResult object

        Returns:
            Summary string
        """
        if result.num_subjects == 0:
            return "No subjects detected"

        category = self.get_sharpness_category(result.laplacian_variance)
        classification = "SHARP" if result.is_sharp else "BLURRY"

        summary = (
            f"{classification} ({category}) - "
            f"Variance: {result.laplacian_variance:.1f} "
            f"(threshold: {self.threshold})"
        )

        if result.num_subjects > 1:
            summary += f" - {result.num_subjects} subjects ({result.strategy_used} strategy)"

        return summary
