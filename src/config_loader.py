"""
Configuration loader and validator for the blur detection system.

Handles loading YAML configuration files, applying defaults, and validating settings.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file with validation.

    Args:
        config_path: Path to the configuration YAML file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
        ValueError: If configuration is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Copy config/config.example.yaml to config/config.yaml and edit it."
        )

    # Load YAML file
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Configuration file is empty")

    # Apply defaults for missing values
    config = apply_defaults(config)

    # Validate configuration
    validate_config(config)

    return config


def apply_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply default values for missing configuration options.

    Args:
        config: Partial configuration dictionary

    Returns:
        Configuration dictionary with defaults applied
    """
    defaults = {
        'paths': {
            'input_dir': '',
            'output_dir': 'output',
            'preserve_structure': False,
            'log_file': None
        },
        'yolo': {
            'model': 'yolov8n.pt',
            'auto_download': True,
            'target_classes': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            'confidence_threshold': 0.25,
            'iou_threshold': 0.45,
            'max_detections': 10,
            'image_size': 640,
            'device': 'auto'
        },
        'sharpness': {
            'threshold': 150.0,
            'multi_subject_strategy': 'largest',
            'min_crop_width': 50,
            'min_crop_height': 50,
            'crop_padding': 0.05
        },
        'processing': {
            'batch_size': 16,
            'num_workers': 4,
            'image_extensions': [
                '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
                '.ARW', '.arw', '.NEF', '.nef', '.CR2', '.cr2',
                '.CR3', '.cr3', '.DNG', '.dng'
            ],
            'case_insensitive': True,
            'skip_processed': False
        },
        'output': {
            'sharp_folder': 'sharp',
            'blurry_folder': 'blurry',
            'no_detection_folder': 'no_detection',
            'no_detection_action': 'move',
            'file_operation': 'move',
            'generate_report': True,
            'report_format': 'txt'
        },
        'logging': {
            'level': 'INFO',
            'console_level': 'INFO',
            'file_level': 'DEBUG',
            'show_progress': True,
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'date_format': '%Y-%m-%d %H:%M:%S'
        },
        'performance': {
            'use_gpu': True,
            'half_precision': False,
            'preload_model': True,
            'max_memory_mb': None
        },
        'advanced': {
            'dry_run': False,
            'save_visualizations': False,
            'visualizations_dir': 'output/visualizations',
            'save_scores': True,
            'scores_file': 'output/sharpness_scores.csv',
            'error_handling': 'skip',
            'verify_operations': False
        }
    }

    # Merge user config with defaults (user config takes precedence)
    for section, section_defaults in defaults.items():
        if section not in config:
            config[section] = {}
        for key, default_value in section_defaults.items():
            if key not in config[section]:
                config[section][key] = default_value

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration values.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate paths
    if not config['paths']['input_dir']:
        raise ValueError("Input directory must be specified in config")

    if not config['paths']['output_dir']:
        raise ValueError("Output directory must be specified in config")

    # Validate YOLO settings
    yolo = config['yolo']
    if yolo['confidence_threshold'] < 0 or yolo['confidence_threshold'] > 1:
        raise ValueError("YOLO confidence_threshold must be between 0 and 1")

    if yolo['iou_threshold'] < 0 or yolo['iou_threshold'] > 1:
        raise ValueError("YOLO iou_threshold must be between 0 and 1")

    if yolo['image_size'] % 32 != 0:
        raise ValueError("YOLO image_size must be a multiple of 32")

    if yolo['device'] not in ['auto', 'cuda', 'cpu', 'mps']:
        raise ValueError("YOLO device must be one of: auto, cuda, cpu, mps")

    # Validate sharpness settings
    sharpness = config['sharpness']
    if sharpness['threshold'] <= 0:
        raise ValueError("Sharpness threshold must be positive")

    valid_strategies = ['largest', 'average', 'best', 'worst']
    if sharpness['multi_subject_strategy'] not in valid_strategies:
        raise ValueError(
            f"Multi-subject strategy must be one of: {', '.join(valid_strategies)}"
        )

    if sharpness['min_crop_width'] < 1 or sharpness['min_crop_height'] < 1:
        raise ValueError("Minimum crop dimensions must be at least 1 pixel")

    if sharpness['crop_padding'] < 0 or sharpness['crop_padding'] > 1:
        raise ValueError("Crop padding must be between 0 and 1")

    # Validate processing settings
    processing = config['processing']
    if processing['batch_size'] < 1:
        raise ValueError("Batch size must be at least 1")

    if processing['num_workers'] < 0:
        raise ValueError("Number of workers cannot be negative")

    # Validate output settings
    output = config['output']
    valid_actions = ['move', 'skip', 'blurry']
    if output['no_detection_action'] not in valid_actions:
        raise ValueError(
            f"No detection action must be one of: {', '.join(valid_actions)}"
        )

    valid_operations = ['move', 'copy']
    if output['file_operation'] not in valid_operations:
        raise ValueError(
            f"File operation must be one of: {', '.join(valid_operations)}"
        )

    valid_formats = ['txt', 'json', 'csv']
    if output['report_format'] not in valid_formats:
        raise ValueError(
            f"Report format must be one of: {', '.join(valid_formats)}"
        )

    # Validate logging settings
    valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if config['logging']['level'].upper() not in valid_log_levels:
        raise ValueError(
            f"Log level must be one of: {', '.join(valid_log_levels)}"
        )

    # Validate advanced settings
    advanced = config['advanced']
    valid_error_handling = ['skip', 'stop']
    if advanced['error_handling'] not in valid_error_handling:
        raise ValueError(
            f"Error handling must be one of: {', '.join(valid_error_handling)}"
        )


def create_output_directories(config: Dict[str, Any]) -> None:
    """
    Create output directory structure based on configuration.

    Args:
        config: Configuration dictionary
    """
    output_dir = Path(config['paths']['output_dir'])
    output_config = config['output']

    # Create main output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create classification folders
    (output_dir / output_config['sharp_folder']).mkdir(exist_ok=True)
    (output_dir / output_config['blurry_folder']).mkdir(exist_ok=True)
    (output_dir / output_config['no_detection_folder']).mkdir(exist_ok=True)

    # Create visualization directory if enabled
    if config['advanced']['save_visualizations']:
        vis_dir = Path(config['advanced']['visualizations_dir'])
        vis_dir.mkdir(parents=True, exist_ok=True)


def get_config_value(config: Dict[str, Any], key_path: str, default=None) -> Any:
    """
    Safely get a configuration value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the value (e.g., 'yolo.confidence_threshold')
        default: Default value if key doesn't exist

    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def print_config_summary(config: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Print a summary of key configuration settings.

    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("=" * 70)
    logger.info("Configuration Summary")
    logger.info("=" * 70)

    logger.info(f"Input Directory: {config['paths']['input_dir']}")
    logger.info(f"Output Directory: {config['paths']['output_dir']}")

    logger.info(f"YOLO Model: {config['yolo']['model']}")
    logger.info(f"YOLO Device: {config['yolo']['device']}")
    logger.info(f"YOLO Confidence: {config['yolo']['confidence_threshold']}")

    logger.info(f"Sharpness Threshold: {config['sharpness']['threshold']}")
    logger.info(f"Multi-Subject Strategy: {config['sharpness']['multi_subject_strategy']}")

    logger.info(f"Batch Size: {config['processing']['batch_size']}")
    logger.info(f"File Operation: {config['output']['file_operation']}")

    if config['advanced']['dry_run']:
        logger.warning("DRY RUN MODE - Files will not be moved/copied")

    logger.info("=" * 70)
