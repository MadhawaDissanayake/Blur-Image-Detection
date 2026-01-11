"""
Main entry point for the YOLO-Crop + Laplacian Blur Detection system.

Usage:
    python src/main.py --config config/config.yaml
    python src/main.py --dry-run
    python src/main.py --visualize
"""

import argparse
import sys
from pathlib import Path

from .config_loader import load_config, print_config_summary, create_output_directories
from .processor import BlurDetectionProcessor
from .utils import setup_logging, get_system_info, validate_paths


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLO-Crop + Laplacian Blur Detection for Wildlife Photography",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config config/config.yaml
  %(prog)s --dry-run
  %(prog)s --visualize
  %(prog)s --config custom_config.yaml --dry-run

For more information, see README.md
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Process images but do not move/copy files (for testing)'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Save visualization images with bounding boxes'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Blur Detection System 1.0.0'
    )

    return parser.parse_args()


def print_welcome():
    """Print welcome message."""
    print("")
    print("=" * 70)
    print("  YOLO-Crop + Laplacian Blur Detection")
    print("  Wildlife Photography Sharpness Analyzer")
    print("=" * 70)
    print("")


def check_system_requirements(logger):
    """
    Check system requirements and display warnings.

    Args:
        logger: Logger instance
    """
    system_info = get_system_info()

    logger.info("System Information:")
    logger.info(f"  PyTorch Version: {system_info.get('pytorch_version', 'Not installed')}")

    if system_info['gpu_available']:
        logger.info(f"  GPU: {system_info['gpu_name']}")
        logger.info(f"  CUDA Version: {system_info['cuda_version']}")

        # Check for RTX 5070 Ti compatibility
        if system_info['gpu_name'] and '5070' in system_info['gpu_name']:
            if system_info['cuda_version'] and system_info['cuda_version'].startswith('12.8'):
                logger.info("  RTX 5070 Ti detected with CUDA 12.8 - Optimal configuration!")
            else:
                logger.warning(
                    "  RTX 5070 Ti detected but CUDA version is not 12.8. "
                    "Install PyTorch nightly for best performance."
                )
    else:
        logger.warning("  No GPU detected - processing will use CPU (slower)")

        if not system_info.get('pytorch_version'):
            logger.error("  PyTorch not installed! Install with:")
            logger.error("    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128")
            return False

    return True


def main():
    """Main entry point."""
    # Print welcome
    print_welcome()

    # Parse arguments
    args = parse_arguments()

    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        # Apply CLI overrides
        if args.dry_run:
            config['advanced']['dry_run'] = True
            print("DRY RUN MODE: Files will not be moved/copied")

        if args.visualize:
            config['advanced']['save_visualizations'] = True
            print("Visualization mode enabled")

        # Setup logging
        logger = setup_logging(config)
        logger.info("Blur detection system started")

        # Print configuration summary
        print_config_summary(config, logger)

        # Check system requirements
        if not check_system_requirements(logger):
            logger.error("System requirements not met. Exiting.")
            return 1

        # Validate paths
        if not validate_paths(config, logger):
            logger.error("Path validation failed. Exiting.")
            return 1

        # Create output directories
        create_output_directories(config)

        # Initialize processor
        logger.info("Initializing blur detection processor...")
        processor = BlurDetectionProcessor(config, logger)

        # Process all images
        logger.info("Starting image processing...")
        report = processor.process_all()

        # Display summary
        print(report.format_summary())
        logger.info("Processing completed successfully")

        # Return exit code based on errors
        if report.error_count > 0:
            logger.warning(f"{report.error_count} images had errors")
            return 1
        else:
            return 0

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nTo create a configuration file:")
        print("  1. Copy config/config.example.yaml to config/config.yaml")
        print("  2. Edit config/config.yaml with your paths and settings")
        print("  3. Run again")
        return 1

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        return 130

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
