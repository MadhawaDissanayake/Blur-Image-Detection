"""
Utility functions for the blur detection system.

Includes logging setup, image loading (with RAW file support), validation,
and helper functions.
"""

import logging
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

import cv2
import numpy as np


def setup_logging(config: dict) -> logging.Logger:
    """
    Configure logging for console and file output.

    Args:
        config: Configuration dictionary containing logging settings

    Returns:
        Configured logger instance
    """
    # Get logging configuration
    log_level = config.get('logging', {}).get('level', 'INFO')
    console_level = config.get('logging', {}).get('console_level', log_level)
    file_level = config.get('logging', {}).get('file_level', 'DEBUG')
    log_format = config.get('logging', {}).get(
        'format',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    date_format = config.get('logging', {}).get('date_format', '%Y-%m-%d %H:%M:%S')

    # Create logger
    logger = logging.getLogger('BlurDetection')
    logger.setLevel(logging.DEBUG)  # Capture all levels, filters applied to handlers

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_formatter = logging.Formatter(log_format, datefmt=date_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = config.get('paths', {}).get('log_file')
    if log_file is None:
        # Auto-generate timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f'blur_detection_{timestamp}.log'

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(getattr(logging, file_level.upper()))
    file_formatter = logging.Formatter(log_format, datefmt=date_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized - Console: {console_level}, File: {file_level}")
    logger.info(f"Log file: {log_file}")

    return logger


def is_raw_file(path: str) -> bool:
    """
    Check if a file is a RAW image format.

    Args:
        path: Path to the image file

    Returns:
        True if the file is a RAW format, False otherwise
    """
    raw_extensions = (
        '.arw', '.nef', '.cr2', '.cr3', '.dng',
        '.raf', '.orf', '.rw2', '.pef', '.srw'
    )
    return Path(path).suffix.lower() in raw_extensions


def validate_image_file(path: str, supported_extensions: list = None) -> bool:
    """
    Validate if a file is a supported image format.

    Args:
        path: Path to the image file
        supported_extensions: List of supported file extensions (optional)

    Returns:
        True if the file has a supported extension, False otherwise
    """
    if supported_extensions is None:
        # Default supported extensions
        supported_extensions = [
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
            '.ARW', '.arw', '.NEF', '.nef', '.CR2', '.cr2',
            '.CR3', '.cr3', '.DNG', '.dng'
        ]

    file_ext = Path(path).suffix.lower()
    supported_lower = [ext.lower() for ext in supported_extensions]

    return file_ext in supported_lower


def load_image_safely(path: str, logger: Optional[logging.Logger] = None) -> Optional[np.ndarray]:
    """
    Load image supporting both standard formats and RAW files.

    This function can read:
    - Standard formats (JPEG, PNG, BMP, TIFF) via OpenCV
    - RAW formats (ARW, NEF, CR2, DNG, etc.) via rawpy

    Args:
        path: Path to the image file
        logger: Optional logger for debugging

    Returns:
        Image as numpy array in BGR format (OpenCV standard), or None if failed
    """
    if logger is None:
        logger = logging.getLogger('BlurDetection')

    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        return None

    try:
        if is_raw_file(path):
            # RAW file - use rawpy
            logger.debug(f"Loading RAW file: {path}")
            import rawpy

            with rawpy.imread(path) as raw:
                # Extract RGB numpy array from RAW
                # postprocess() applies demosaicing and color correction
                rgb = raw.postprocess(
                    use_camera_wb=True,  # Use camera white balance
                    half_size=False,     # Full resolution
                    no_auto_bright=False,  # Allow auto brightness
                    output_bps=8         # 8-bit output (standard)
                )

                # Convert RGB to BGR for OpenCV compatibility
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                logger.debug(f"RAW file loaded successfully: {bgr.shape}")
                return bgr

        else:
            # Standard image - use OpenCV
            logger.debug(f"Loading standard image: {path}")
            image = cv2.imread(path)

            if image is None:
                logger.error(f"Failed to load image with OpenCV: {path}")
                return None

            logger.debug(f"Image loaded successfully: {image.shape}")
            return image

    except ImportError as e:
        logger.error(f"Missing library for RAW file support: {e}")
        logger.error("Install rawpy with: pip install rawpy")
        return None

    except Exception as e:
        logger.error(f"Error loading image {path}: {str(e)}")
        return None


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "1h 23m 45s" or "12.3s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def format_bytes(bytes_count: int) -> str:
    """
    Format bytes into human-readable size string.

    Args:
        bytes_count: Number of bytes

    Returns:
        Formatted size string (e.g., "1.5 MB", "2.3 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f} PB"


def check_gpu_available() -> bool:
    """
    Check if GPU (CUDA) is available for PyTorch.

    Returns:
        True if GPU is available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_system_info() -> dict:
    """
    Get system information including GPU details.

    Returns:
        Dictionary containing system information
    """
    info = {
        'gpu_available': False,
        'gpu_name': None,
        'cuda_version': None,
        'pytorch_version': None
    }

    try:
        import torch
        info['pytorch_version'] = torch.__version__

        if torch.cuda.is_available():
            info['gpu_available'] = True
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['cuda_version'] = torch.version.cuda

    except ImportError:
        pass

    return info


def validate_paths(config: dict, logger: logging.Logger) -> bool:
    """
    Validate that required paths exist or can be created.

    Args:
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        True if all paths are valid, False otherwise
    """
    paths_config = config.get('paths', {})

    # Check input directory
    input_dir = paths_config.get('input_dir')
    if not input_dir:
        logger.error("Input directory not specified in config")
        return False

    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return False

    if not os.path.isdir(input_dir):
        logger.error(f"Input path is not a directory: {input_dir}")
        return False

    logger.info(f"Input directory validated: {input_dir}")

    # Check/create output directory
    output_dir = paths_config.get('output_dir')
    if not output_dir:
        logger.error("Output directory not specified in config")
        return False

    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory ready: {output_dir}")
    except Exception as e:
        logger.error(f"Cannot create output directory {output_dir}: {e}")
        return False

    return True


class ProgressTracker:
    """Simple progress tracker for logging."""

    def __init__(self, total: int, logger: logging.Logger, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.logger = logger
        self.description = description
        self.start_time = datetime.now()

    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.current += n
        if self.current % 10 == 0 or self.current == self.total:
            percent = (self.current / self.total) * 100 if self.total > 0 else 0
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.current / elapsed if elapsed > 0 else 0
            self.logger.info(
                f"{self.description}: {self.current}/{self.total} "
                f"({percent:.1f}%) - {rate:.1f} imgs/sec"
            )

    def finish(self):
        """Mark progress as complete."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(
            f"{self.description} complete: {self.current} images "
            f"in {format_time(elapsed)}"
        )
