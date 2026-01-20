<<<<<< HEAD
# Blur-Image-Detection
AI-powered wildlife photo sorter that uses YOLOv8 subject detection and Laplacian variance to automatically cull blurry images while preserving artistic bokeh.

Wildlife Focus-Sorter is a high-performance Python tool designed for photographers to automate the culling process. Unlike standard blur detectors, this tool uses Computer Vision (YOLOv8) to identify the subject (bird, animal, or person) and only evaluates sharpness within that specific region. This ensures that photos with intentional background blur (bokeh) are kept, while shots with missed focus are moved to a separate directory.
=======
# YOLO-Crop + Laplacian Blur Detection for Wildlife Photography

Automatically detect and organize blurry vs sharp wildlife photos by focusing on the subject (bird/animal), not the bokeh background.

## Features

- **Smart Subject Detection**: Uses YOLOv8n to identify wildlife subjects (birds, mammals)
- **Focused Sharpness Analysis**: Calculates blur only on the subject, ignoring background bokeh
- **RAW File Support**: Processes Sony ARW, Nikon NEF, Canon CR2/CR3, and Adobe DNG files
- **RTX 5070 Ti Optimized**: Configured for NVIDIA's latest Blackwell architecture
- **Batch Processing**: Efficiently processes 16-32 images at once with GPU acceleration
- **Configurable**: All settings via YAML configuration file
- **Production-Ready**: Comprehensive logging, error handling, and progress tracking

## Quick Start

### 1. System Requirements

- **Python**: 3.8 or higher
- **GPU**: NVIDIA RTX 5070 Ti (or any CUDA-capable GPU)
- **Driver**: NVIDIA driver 570+ for RTX 5070 Ti
- **OS**: Windows, Linux, or macOS
- **Memory**: 8GB+ RAM recommended

### 2. Installation

#### Step 1: Create Virtual Environment

```bash
cd C:\DATA\Projects\Photography\BlurImageDetection

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
# source venv/bin/activate
```

#### Step 2: Install PyTorch (CRITICAL for RTX 5070 Ti)

The RTX 5070 Ti requires PyTorch nightly with CUDA 12.8 support. Standard PyTorch WILL NOT WORK.

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### Step 3: Verify GPU Support

```python
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'CUDA Version: {torch.version.cuda}')"
```

Expected output:
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 5070 Ti
CUDA Version: 12.8
```

#### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configuration

Copy the example configuration:

```bash
copy config\config.example.yaml config\config.yaml
```

Edit `config/config.yaml` and set your paths:

```yaml
paths:
  # Your source photo directory
  input_dir: "C:/DATA/Projects/Photography/TestImages"

  # Where to save organized results
  output_dir: "C:/DATA/Projects/Photography/BlurImageDetection/output"

sharpness:
  # Blur threshold (adjust based on results)
  threshold: 150.0
```

### 4. Run Processing

```bash
# Process images
python -m src.main

# Dry run (test without moving files)
python -m src.main --dry-run

# Custom config file
python -m src.main --config my_config.yaml
```

## How It Works

### Processing Pipeline

```
1. Scan Input Directory
   ↓
2. Load Images (RAW or Standard)
   ↓
3. YOLO Detection (Batch)
   ├─ Detect Wildlife Subjects
   ├─ Filter by Confidence (0.25)
   └─ Extract Bounding Boxes
   ↓
4. Sharpness Analysis
   ├─ Crop to Subject Bounding Box
   ├─ Calculate Laplacian Variance
   └─ Compare to Threshold (150.0)
   ↓
5. Organize Files
   ├─ Sharp → output/sharp/
   ├─ Blurry → output/blurry/
   └─ No Detection → output/no_detection/
```

### The Algorithm

**YOLO-Crop + Laplacian Method**:

1. **YOLO Detection**: Identifies wildlife subjects in the image
2. **Smart Cropping**: Extracts the region around the detected subject
3. **Laplacian Variance**: Calculates sharpness only on the cropped region
4. **Classification**: Compares variance to threshold to determine sharp vs blurry

This approach ignores the intentionally blurred background (bokeh) and focuses only on whether the subject is in focus.

## Configuration Guide

### Key Settings

| Setting | Description | Default | Recommended |
|---------|-------------|---------|-------------|
| `sharpness.threshold` | Laplacian variance threshold | 150.0 | 100-250 for wildlife |
| `processing.batch_size` | Images per batch | 16 | 16-32 for RTX 5070 Ti |
| `yolo.confidence_threshold` | Detection confidence | 0.25 | 0.20-0.30 |
| `sharpness.multi_subject_strategy` | Multiple subjects handling | largest | largest/average/best/worst |
| `output.no_detection_action` | No subject detected | move | move/skip/blurry |
| `output.file_operation` | File handling | move | move/copy |

### Threshold Tuning

The `sharpness.threshold` value determines what's considered sharp vs blurry.

**Start with default (150) and adjust**:
- **Too many false blurry**: Decrease threshold (try 100-120)
- **Too many false sharp**: Increase threshold (try 180-250)

**Process TestImages first**:
```bash
# Process test images with score logging
python -m src.main --dry-run
```

Check `output/sharpness_scores.csv` to see variance values and tune threshold accordingly.

### Multi-Subject Strategies

When multiple animals are detected in one image:

- **largest** (recommended): Use the largest subject by area
- **average**: Average sharpness across all subjects
- **best**: Use the sharpest subject only
- **worst**: Use the blurriest subject (conservative approach)

## RAW File Support

### Supported Formats

- **Sony**: .ARW
- **Nikon**: .NEF
- **Canon**: .CR2, .CR3
- **Adobe**: .DNG
- **And more**: .RAF (Fuji), .ORF (Olympus), .RW2 (Panasonic)

### How RAW Processing Works

```python
# RAW files are automatically detected and processed
ARW file → rawpy.imread() → postprocess() → RGB array → BGR conversion → YOLO/OpenCV
```

**Performance Impact**: RAW processing adds ~200-500ms per image due to demosaicing, but provides analysis of original sensor data without JPEG compression artifacts.

## Output Structure

After processing, files are organized as:

```
output/
├── sharp/              # Sharp, in-focus images
├── blurry/             # Blurry or out-of-focus images
├── no_detection/       # No wildlife detected
├── sharpness_scores.csv  # Detailed scores for all images
└── processing_report.txt # Summary statistics
```

## Performance

### Expected Throughput

| Hardware | Speed | Notes |
|----------|-------|-------|
| RTX 5070 Ti | 5-8 imgs/sec | With PyTorch nightly + CUDA 12.8 |
| RTX 4090 | 8-12 imgs/sec | With standard PyTorch |
| RTX 3060 | 3-5 imgs/sec | Reduce batch_size if OOM |
| CPU (i7) | 1-2 imgs/sec | Fallback mode |

### Memory Usage

- **Batch size 16**: ~4-6GB VRAM
- **Batch size 32**: ~8-10GB VRAM
- **RAW files**: +1-2GB RAM for processing

**If you get "CUDA out of memory" errors**:
```yaml
processing:
  batch_size: 8  # Reduce from 16 to 8 or 4
```

## Troubleshooting

### "CUDA capability sm_120 is not compatible"

**Problem**: RTX 5070 Ti not supported by standard PyTorch

**Solution**:
```bash
pip uninstall torch torchvision
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### "Cannot read ARW file" / "rawpy not found"

**Problem**: RAW file library not installed

**Solution**:
```bash
pip install rawpy imageio
```

### "No GPU detected"

**Problem**: NVIDIA driver issue or PyTorch CPU-only version

**Solution**:
1. Check driver: `nvidia-smi` (should show driver 570+)
2. Verify PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall PyTorch with CUDA 12.8 (see installation steps)

### Low Detection Rate

**Problem**: Wildlife not being detected

**Solutions**:
- Decrease `yolo.confidence_threshold` (try 0.15 or 0.20)
- Check if subjects are in COCO classes (see config for list)
- Review `output/sharpness_scores.csv` to see detection results

### High Memory Usage

**Problem**: System running out of RAM

**Solutions**:
```yaml
processing:
  batch_size: 4        # Reduce batch size
  num_workers: 2       # Reduce worker threads
performance:
  half_precision: true # Enable FP16 (RTX 5070 Ti supports this)
```

## Advanced Usage

### Dry Run (Testing)

Test threshold values without moving files:

```bash
python -m src.main --dry-run
```

Review `output/sharpness_scores.csv` to see variance values, then adjust threshold in config.

### Save Visualizations

Save images with bounding boxes drawn:

```yaml
advanced:
  save_visualizations: true
  visualizations_dir: "output/visualizations"
```

### Multiple Configurations

Use different configs for different wildlife types:

```bash
# Birds in flight (stricter)
python -m src.main --config config_birds.yaml

# Mammals (more lenient)
python -m src.main --config config_mammals.yaml
```

## Development

### Project Structure

```
BlurImageDetection/
├── config/
│   ├── config.yaml           # User configuration
│   └── config.example.yaml   # Template
├── src/
│   ├── main.py               # Entry point
│   ├── config_loader.py      # Config handling
│   ├── detector.py           # YOLO detection
│   ├── sharpness.py          # Laplacian analysis
│   ├── file_manager.py       # File operations
│   ├── processor.py          # Main orchestrator
│   └── utils.py              # Helper functions
├── logs/                     # Generated logs
├── output/                   # Processing results
│   ├── sharp/
│   ├── blurry/
│   └── no_detection/
├── TestImages/               # Sample images
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

### Running from Source

```bash
# From project root
python -m src.main --config config/config.yaml

# Or directly
python src/main.py
```

## Credits

- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Laplacian Method**: OpenCV
- **RAW Processing**: [rawpy](https://github.com/letmaik/rawpy) (LibRaw wrapper)

## License

This project is provided as-is for wildlife photography organization. Modify and use as needed.

## Support

For issues, questions, or suggestions:
1. Check this README and configuration comments
2. Review `logs/` for error details
3. Check `output/sharpness_scores.csv` for detection/sharpness data

---

**Happy Shooting!** 📸🦅

Built for wildlife photographers who need to quickly sort through thousands of photos to find the sharp keepers.
>>>>>>> 2ef3059 (Initial commit: Add blur detection system with percentile-based sharpness analysis)
