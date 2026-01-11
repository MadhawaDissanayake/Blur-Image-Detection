"""
File management module for organizing images.

Handles scanning input directories, moving/copying files to output folders,
and maintaining folder structure.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

from .utils import validate_image_file


class FileManager:
    """Manages file operations for the blur detection system."""

    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        """
        Initialize file manager.

        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger('BlurDetection.FileManager')

        self.input_dir = Path(config['paths']['input_dir'])
        self.output_dir = Path(config['paths']['output_dir'])
        self.preserve_structure = config['paths']['preserve_structure']

        self.sharp_folder = config['output']['sharp_folder']
        self.blurry_folder = config['output']['blurry_folder']
        self.no_detection_folder = config['output']['no_detection_folder']

        self.file_operation = config['output']['file_operation']
        self.image_extensions = config['processing']['image_extensions']

        self.logger.info(f"File manager initialized")
        self.logger.info(f"Input: {self.input_dir}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info(f"Operation: {self.file_operation}")

    def scan_input_directory(self) -> List[str]:
        """
        Recursively scan input directory for image files.

        Returns:
            List of absolute paths to image files
        """
        self.logger.info(f"Scanning input directory: {self.input_dir}")

        image_paths = []

        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                file_path = os.path.join(root, file)

                if validate_image_file(file_path, self.image_extensions):
                    image_paths.append(file_path)

        self.logger.info(f"Found {len(image_paths)} images")

        return image_paths

    def create_output_structure(self) -> None:
        """Create output directory structure."""
        self.logger.info("Creating output directory structure")

        # Create main folders
        (self.output_dir / self.sharp_folder).mkdir(parents=True, exist_ok=True)
        (self.output_dir / self.blurry_folder).mkdir(parents=True, exist_ok=True)
        (self.output_dir / self.no_detection_folder).mkdir(parents=True, exist_ok=True)

        self.logger.info("Output structure created")

    def get_output_path(self, original_path: str, category: str) -> Path:
        """
        Calculate output path for a file based on category.

        Args:
            original_path: Original file path
            category: Category ('sharp', 'blurry', 'no_detection')

        Returns:
            Target output path
        """
        original = Path(original_path)

        # Determine folder based on category
        if category == 'sharp':
            category_folder = self.sharp_folder
        elif category == 'blurry':
            category_folder = self.blurry_folder
        elif category == 'no_detection':
            category_folder = self.no_detection_folder
        else:
            raise ValueError(f"Unknown category: {category}")

        if self.preserve_structure:
            # Preserve original folder structure
            try:
                rel_path = original.relative_to(self.input_dir)
                target = self.output_dir / category_folder / rel_path
            except ValueError:
                # If path is not relative to input_dir, use flat structure
                target = self.output_dir / category_folder / original.name
        else:
            # Flat structure
            target = self.output_dir / category_folder / original.name

        # Create parent directories if needed
        target.parent.mkdir(parents=True, exist_ok=True)

        return target

    def handle_duplicate_filename(self, target_path: Path) -> Path:
        """
        Handle duplicate filenames by appending a counter.

        Args:
            target_path: Proposed target path

        Returns:
            Available target path (may have counter appended)
        """
        if not target_path.exists():
            return target_path

        # File exists - append counter
        counter = 1
        stem = target_path.stem
        suffix = target_path.suffix
        parent = target_path.parent

        while True:
            new_path = parent / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1

    def move_file(self, src: str, category: str) -> bool:
        """
        Move file to appropriate output folder.

        Args:
            src: Source file path
            category: Category ('sharp', 'blurry', 'no_detection')

        Returns:
            True if successful, False otherwise
        """
        try:
            src_path = Path(src)
            if not src_path.exists():
                self.logger.error(f"Source file not found: {src}")
                return False

            # Get target path
            target_path = self.get_output_path(src, category)

            # Handle duplicates
            target_path = self.handle_duplicate_filename(target_path)

            # Perform operation
            if self.file_operation == 'move':
                shutil.move(str(src_path), str(target_path))
                self.logger.debug(f"Moved: {src_path.name} → {category}/")
            elif self.file_operation == 'copy':
                shutil.copy2(str(src_path), str(target_path))
                self.logger.debug(f"Copied: {src_path.name} → {category}/")
            else:
                self.logger.error(f"Unknown file operation: {self.file_operation}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Failed to {self.file_operation} {src}: {e}")
            return False

    def organize_file(self, image_path: str, is_sharp: Optional[bool],
                     has_detection: bool) -> bool:
        """
        Organize a file based on sharpness and detection results.

        Args:
            image_path: Path to the image file
            is_sharp: True if sharp, False if blurry, None if no analysis
            has_detection: True if wildlife was detected

        Returns:
            True if successful, False otherwise
        """
        # Determine category
        if not has_detection:
            action = self.config['output']['no_detection_action']

            if action == 'skip':
                self.logger.debug(f"Skipping (no detection): {Path(image_path).name}")
                return True
            elif action == 'blurry':
                category = 'blurry'
            else:  # 'move'
                category = 'no_detection'

        elif is_sharp:
            category = 'sharp'
        else:
            category = 'blurry'

        # Check if dry run
        if self.config['advanced']['dry_run']:
            self.logger.info(f"[DRY RUN] Would {self.file_operation} to {category}/: {Path(image_path).name}")
            return True

        # Move/copy file
        return self.move_file(image_path, category)

    def get_output_summary(self) -> dict:
        """
        Get summary of files in output directories.

        Returns:
            Dictionary with counts for each category
        """
        summary = {}

        for category, folder in [
            ('sharp', self.sharp_folder),
            ('blurry', self.blurry_folder),
            ('no_detection', self.no_detection_folder)
        ]:
            folder_path = self.output_dir / folder
            if folder_path.exists():
                count = sum(1 for _ in folder_path.rglob('*') if _.is_file())
                summary[category] = count
            else:
                summary[category] = 0

        return summary

    def cleanup_empty_directories(self, path: Optional[Path] = None) -> None:
        """
        Remove empty directories recursively.

        Args:
            path: Path to clean up (default: input directory)
        """
        if path is None:
            path = self.input_dir

        if not path.exists() or not path.is_dir():
            return

        try:
            # Remove empty subdirectories
            for subdir in sorted(path.rglob('*'), reverse=True):
                if subdir.is_dir() and not any(subdir.iterdir()):
                    subdir.rmdir()
                    self.logger.debug(f"Removed empty directory: {subdir}")

        except Exception as e:
            self.logger.warning(f"Error cleaning up directories: {e}")
