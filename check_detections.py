"""Check what YOLO actually detected in the test images."""
import sys
sys.path.insert(0, 'src')

from src.config_loader import load_config
from src.detector import WildlifeDetector
from src.utils import setup_logging, load_image_safely
import os

# Load config
config = load_config('config/config.yaml')
logger = setup_logging(config)

# Initialize detector
detector = WildlifeDetector(config, logger)
detector.load_model()

# Get test images
test_dir = r'TestImages'
images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.ARW')]

print("\n" + "="*70)
print("YOLO Detection Results for Test Images")
print("="*70 + "\n")

for img_path in images:
    filename = os.path.basename(img_path)
    print(f"\n{filename}:")

    # Load image
    image = load_image_safely(img_path, logger)
    if image is None:
        print("  ERROR: Could not load image")
        continue

    # Detect
    result = detector.detect_single(img_path)

    if result.has_detection:
        print(f"  ✓ Detected {len(result.boxes)} wildlife subject(s):")
        for i, box in enumerate(result.boxes, 1):
            area = box.area()
            print(f"    {i}. {box.class_name} (confidence: {box.confidence:.2f}, area: {area:,} pixels)")
    else:
        print("  ✗ No wildlife detected")

print("\n" + "="*70)
print("\nTarget classes configured:")
print(", ".join(detector._get_target_class_names()))
print("="*70 + "\n")
