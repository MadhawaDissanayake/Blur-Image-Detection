"""
Main processing orchestrator for blur detection.

Coordinates detection, sharpness analysis, and file organization.
"""

import csv
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from .detector import WildlifeDetector, DetectionResult
from .sharpness import SharpnessAnalyzer, SharpnessResult
from .file_manager import FileManager
from .utils import load_image_safely, format_time


@dataclass
class ProcessingResult:
    """Results from processing a single image."""
    image_path: str
    status: str  # 'sharp', 'blurry', 'no_detection', 'error'
    detection: Optional[DetectionResult]
    sharpness: Optional[SharpnessResult]
    error_message: Optional[str]
    processing_time: float


@dataclass
class ProcessingReport:
    """Summary report from batch processing."""
    total_images: int
    sharp_count: int
    blurry_count: int
    no_detection_count: int
    error_count: int
    total_time: float
    average_time_per_image: float
    results: List[ProcessingResult]

    def format_summary(self) -> str:
        """Format summary as human-readable string."""
        lines = [
            "",
            "=" * 70,
            "PROCESSING SUMMARY",
            "=" * 70,
            f"Total Images Processed: {self.total_images}",
            f"",
            f"Sharp:          {self.sharp_count:5d} ({self._percent(self.sharp_count)}%)",
            f"Blurry:         {self.blurry_count:5d} ({self._percent(self.blurry_count)}%)",
            f"No Detection:   {self.no_detection_count:5d} ({self._percent(self.no_detection_count)}%)",
            f"Errors:         {self.error_count:5d} ({self._percent(self.error_count)}%)",
            f"",
            f"Total Time: {format_time(self.total_time)}",
            f"Average: {self.average_time_per_image:.2f}s per image",
            "=" * 70,
            ""
        ]
        return "\n".join(lines)

    def _percent(self, count: int) -> str:
        """Calculate percentage with formatting."""
        if self.total_images == 0:
            return "0.0"
        return f"{(count / self.total_images) * 100:.1f}"


class BlurDetectionProcessor:
    """Main processor coordinating all components."""

    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        """
        Initialize blur detection processor.

        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger('BlurDetection.Processor')

        # Initialize components
        self.detector = WildlifeDetector(config, self.logger)
        self.sharpness_analyzer = SharpnessAnalyzer(config, self.logger)
        self.file_manager = FileManager(config, self.logger)

        self.batch_size = config['processing']['batch_size']
        self.show_progress = config['logging']['show_progress']

        self.logger.info("Processor initialized")

    def process_all(self) -> ProcessingReport:
        """
        Process all images in input directory.

        Returns:
            ProcessingReport with results
        """
        self.logger.info("Starting blur detection processing")
        start_time = time.time()

        # Load YOLO model
        self.detector.load_model()

        # Create output structure
        self.file_manager.create_output_structure()

        # Scan input directory
        image_paths = self.file_manager.scan_input_directory()

        if not image_paths:
            self.logger.warning("No images found in input directory")
            return ProcessingReport(
                total_images=0,
                sharp_count=0,
                blurry_count=0,
                no_detection_count=0,
                error_count=0,
                total_time=0.0,
                average_time_per_image=0.0,
                results=[]
            )

        # Process in batches
        all_results = []
        batches = [image_paths[i:i + self.batch_size]
                  for i in range(0, len(image_paths), self.batch_size)]

        self.logger.info(f"Processing {len(image_paths)} images in {len(batches)} batches")

        # Process with progress bar
        if self.show_progress:
            with tqdm(total=len(image_paths), desc="Processing", unit="img") as pbar:
                for batch in batches:
                    batch_results = self.process_batch(batch)
                    all_results.extend(batch_results)
                    pbar.update(len(batch))
        else:
            for i, batch in enumerate(batches, 1):
                self.logger.info(f"Processing batch {i}/{len(batches)}")
                batch_results = self.process_batch(batch)
                all_results.extend(batch_results)

        # Generate report
        total_time = time.time() - start_time
        report = self._generate_report(all_results, total_time)

        # Save results if configured
        if self.config['advanced']['save_scores']:
            self.save_scores_csv(all_results)

        if self.config['output']['generate_report']:
            self.save_report(report)

        self.logger.info(f"Processing complete - {len(image_paths)} images in {format_time(total_time)}")

        return report

    def process_batch(self, image_paths: List[str]) -> List[ProcessingResult]:
        """
        Process a batch of images.

        Args:
            image_paths: List of image paths

        Returns:
            List of ProcessingResult objects
        """
        batch_start = time.time()

        # Run YOLO detection on batch
        try:
            detections = self.detector.detect_batch(image_paths)
        except Exception as e:
            self.logger.error(f"Batch detection failed: {e}")
            # Return error results for all images
            return [
                ProcessingResult(
                    image_path=path,
                    status='error',
                    detection=None,
                    sharpness=None,
                    error_message=str(e),
                    processing_time=0.0
                )
                for path in image_paths
            ]

        # Process each image
        results = []
        for image_path, detection in zip(image_paths, detections):
            result = self.process_single_image(image_path, detection)
            results.append(result)

        batch_time = time.time() - batch_start
        self.logger.debug(
            f"Batch processed: {len(image_paths)} images in {batch_time:.2f}s "
            f"({len(image_paths)/batch_time:.1f} imgs/sec)"
        )

        return results

    def process_single_image(self, image_path: str,
                            detection: DetectionResult) -> ProcessingResult:
        """
        Process a single image.

        Args:
            image_path: Path to the image
            detection: Detection result from YOLO

        Returns:
            ProcessingResult object
        """
        start_time = time.time()

        try:
            if not detection.has_detection:
                # No wildlife detected
                self.logger.debug(f"No detection: {Path(image_path).name}")

                # Organize file based on no_detection_action
                self.file_manager.organize_file(image_path, None, has_detection=False)

                return ProcessingResult(
                    image_path=image_path,
                    status='no_detection',
                    detection=detection,
                    sharpness=None,
                    error_message=None,
                    processing_time=time.time() - start_time
                )

            # Wildlife detected - analyze sharpness
            image = load_image_safely(image_path, self.logger)

            if image is None:
                raise ValueError("Failed to load image")

            sharpness_result = self.sharpness_analyzer.analyze_detection(image, detection)

            # Determine status
            status = 'sharp' if sharpness_result.is_sharp else 'blurry'

            # Log result
            self.logger.debug(
                f"{status.upper()}: {Path(image_path).name} - "
                f"Variance: {sharpness_result.laplacian_variance:.1f}"
            )

            # Save visualization if enabled
            if self.config['advanced']['save_visualizations']:
                self._save_visualization(image, image_path, detection, sharpness_result)

            # Organize file
            self.file_manager.organize_file(
                image_path,
                is_sharp=sharpness_result.is_sharp,
                has_detection=True
            )

            return ProcessingResult(
                image_path=image_path,
                status=status,
                detection=detection,
                sharpness=sharpness_result,
                error_message=None,
                processing_time=time.time() - start_time
            )

        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")

            # Handle error based on configuration
            if self.config['advanced']['error_handling'] == 'stop':
                raise

            return ProcessingResult(
                image_path=image_path,
                status='error',
                detection=detection,
                sharpness=None,
                error_message=str(e),
                processing_time=time.time() - start_time
            )

    def _save_visualization(self, image: np.ndarray, image_path: str,
                           detection: DetectionResult, sharpness: SharpnessResult) -> None:
        """
        Save visualization image with bounding boxes and labels.

        Args:
            image: Original image array
            image_path: Path to original image
            detection: Detection result with bounding boxes
            sharpness: Sharpness analysis result
        """
        try:
            # Create output directory
            vis_dir = Path(self.config['advanced']['visualizations_dir'])
            vis_dir.mkdir(parents=True, exist_ok=True)

            # Create a copy of the image for drawing
            vis_image = image.copy()

            # Draw bounding boxes
            for i, box in enumerate(detection.boxes):
                # Box coordinates
                x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2

                # Color: Green for sharp, Red for blurry
                color = (0, 255, 0) if sharpness.is_sharp else (0, 0, 255)

                # Draw rectangle
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)

                # Prepare label text
                label_parts = [
                    f"{box.class_name}",
                    f"conf: {box.confidence:.2f}"
                ]

                # Add "LARGEST" indicator if this is the largest box in multi-subject
                if len(detection.boxes) > 1:
                    largest = detection.get_largest_box()
                    if largest and box == largest:
                        label_parts.append("LARGEST")

                label = " | ".join(label_parts)

                # Calculate label background size
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )

                # Draw label background
                cv2.rectangle(
                    vis_image,
                    (x1, y1 - label_h - baseline - 10),
                    (x1 + label_w + 10, y1),
                    color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    vis_image,
                    label,
                    (x1 + 5, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            # Add sharpness info at the top
            status_text = f"{'SHARP' if sharpness.is_sharp else 'BLURRY'} | Variance: {sharpness.laplacian_variance:.1f} | Threshold: {self.config['sharpness']['threshold']}"
            status_color = (0, 255, 0) if sharpness.is_sharp else (0, 0, 255)

            # Draw status background
            (status_w, status_h), baseline = cv2.getTextSize(
                status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
            )
            cv2.rectangle(
                vis_image,
                (10, 10),
                (status_w + 30, status_h + baseline + 30),
                status_color,
                -1
            )

            # Draw status text
            cv2.putText(
                vis_image,
                status_text,
                (20, status_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

            # Save visualization
            filename = Path(image_path).stem + '_visualization.jpg'
            output_path = vis_dir / filename

            # Resize if image is too large (for faster saving)
            max_dimension = 2000
            h, w = vis_image.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                vis_image = cv2.resize(vis_image, (new_w, new_h))

            cv2.imwrite(str(output_path), vis_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            self.logger.debug(f"Saved visualization: {filename}")

        except Exception as e:
            self.logger.warning(f"Failed to save visualization for {Path(image_path).name}: {e}")

    def _generate_report(self, results: List[ProcessingResult],
                        total_time: float) -> ProcessingReport:
        """
        Generate processing report from results.

        Args:
            results: List of ProcessingResult objects
            total_time: Total processing time in seconds

        Returns:
            ProcessingReport object
        """
        # Count by status
        sharp_count = sum(1 for r in results if r.status == 'sharp')
        blurry_count = sum(1 for r in results if r.status == 'blurry')
        no_detection_count = sum(1 for r in results if r.status == 'no_detection')
        error_count = sum(1 for r in results if r.status == 'error')

        # Calculate average time
        avg_time = total_time / len(results) if results else 0.0

        return ProcessingReport(
            total_images=len(results),
            sharp_count=sharp_count,
            blurry_count=blurry_count,
            no_detection_count=no_detection_count,
            error_count=error_count,
            total_time=total_time,
            average_time_per_image=avg_time,
            results=results
        )

    def save_scores_csv(self, results: List[ProcessingResult]) -> None:
        """
        Save detailed sharpness scores to CSV file.

        Args:
            results: List of ProcessingResult objects
        """
        csv_file = Path(self.config['advanced']['scores_file'])
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # Header
                writer.writerow([
                    'Image Path',
                    'Status',
                    'Laplacian Variance',
                    'Threshold',
                    'Is Sharp',
                    'Num Subjects',
                    'Strategy',
                    'Processing Time (s)',
                    'Error Message'
                ])

                # Data
                for result in results:
                    if result.sharpness:
                        variance = result.sharpness.laplacian_variance
                        is_sharp = result.sharpness.is_sharp
                        num_subjects = result.sharpness.num_subjects
                        strategy = result.sharpness.strategy_used
                    else:
                        variance = 0.0
                        is_sharp = False
                        num_subjects = 0
                        strategy = 'N/A'

                    writer.writerow([
                        result.image_path,
                        result.status,
                        f"{variance:.2f}",
                        self.config['sharpness']['threshold'],
                        is_sharp,
                        num_subjects,
                        strategy,
                        f"{result.processing_time:.3f}",
                        result.error_message or ''
                    ])

            self.logger.info(f"Scores saved to: {csv_file}")

        except Exception as e:
            self.logger.error(f"Failed to save scores CSV: {e}")

    def save_report(self, report: ProcessingReport) -> None:
        """
        Save processing report to file.

        Args:
            report: ProcessingReport object
        """
        report_format = self.config['output']['report_format']
        output_dir = Path(self.config['paths']['output_dir'])

        if report_format == 'txt':
            report_file = output_dir / 'processing_report.txt'
            try:
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report.format_summary())
                self.logger.info(f"Report saved to: {report_file}")
            except Exception as e:
                self.logger.error(f"Failed to save report: {e}")

        elif report_format == 'json':
            report_file = output_dir / 'processing_report.json'
            try:
                # Convert to dict (excluding full results to keep file small)
                report_dict = {
                    'total_images': report.total_images,
                    'sharp_count': report.sharp_count,
                    'blurry_count': report.blurry_count,
                    'no_detection_count': report.no_detection_count,
                    'error_count': report.error_count,
                    'total_time': report.total_time,
                    'average_time_per_image': report.average_time_per_image
                }

                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(report_dict, f, indent=2)

                self.logger.info(f"Report saved to: {report_file}")
            except Exception as e:
                self.logger.error(f"Failed to save JSON report: {e}")
