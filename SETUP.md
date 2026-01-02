# LOLScan Setup Guide

A YOLO-based object detection system for League of Legends video analysis.

## Project Structure

```
LOLScan/
├── data/                    # Dataset directory
│   ├── raw/                # Raw video files
│   ├── processed/          # Processed frames
│   └── annotations/        # YOLO format annotations
├── models/                 # Model storage
│   ├── trained/           # Fine-tuned models
│   └── pretrained/        # Pre-trained weights
├── src/                    # Source code
│   ├── __init__.py
│   ├── config.py          # Configuration handler
│   ├── detector.py        # YOLO detection wrapper
│   └── video_processor.py # Video frame extraction
├── scripts/               # Executable scripts
│   ├── train.py          # Training script
│   └── detect.py         # Inference script
├── notebooks/            # Jupyter notebooks
├── config/               # Configuration files
│   └── config.yaml       # Main config
├── outputs/              # Output directory
│   ├── logs/            # Training logs
│   └── results/         # Detection results
├── requirements.txt      # Python dependencies
└── setup.py             # Package setup
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
cd C:\dev\LOLScan\LOLScan
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install Python Dependencies

```bash
# Install basic dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA 11.8 support
C:\dev\LOLScan\.venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from ultralytics import YOLO; print('YOLOv8 ready')"
```

### 4. Environment Configuration

Copy `.env.example` to `.env` and adjust settings as needed:

```bash
copy .env.example .env
```

## Usage

### Training

Prepare your dataset in YOLO format and run:

```bash
python scripts/train.py --data ./data/processed --config ./config/config.yaml
```

### Inference on Video

```bash
python scripts/detect.py path/to/video.mp4 --model ./models/trained/best.pt --output ./outputs/video_with_detections.mp4
```

## Dataset Preparation

For YOLO training, organize your dataset as follows:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Labels should be in YOLO format (one `.txt` file per image):
```
<class_id> <x_center> <y_center> <width> <height>
```

## Model Selection

- **yolov8n**: Nano - fastest, least accurate
- **yolov8s**: Small
- **yolov8m**: Medium - good balance (recommended)
- **yolov8l**: Large
- **yolov8x**: Extra large - slowest, most accurate

Change in `config/config.yaml`:
```yaml
model:
  name: "yolov8m"  # Change as needed
```

## Performance Optimization

### For GPU Memory Issues
Edit `config/config.yaml`:
```yaml
training:
  batch_size: 8  # Reduce batch size
  
model:
  imgsz: 416  # Reduce image size
```

### Frame Skipping
Process fewer frames for faster inference:
```yaml
video:
  frame_skip: 2  # Process every 2nd frame
```

## GPU Setup (CUDA 11.8)

The project is configured for CUDA 11.8. If you have a different CUDA version, adjust the PyTorch installation:

- **CUDA 12.1**: `https://download.pytorch.org/whl/cu121`
- **CUDA 11.7**: `https://download.pytorch.org/whl/cu117`
- **CPU Only**: `https://download.pytorch.org/whl/cpu`

## Troubleshooting

**CUDA out of memory**: Reduce batch size and image size in config
**Video codec issues**: Ensure FFmpeg is installed: `pip install imageio-ffmpeg`
**Missing dependencies**: Run `pip install -r requirements.txt` again

## Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [OpenCV Documentation](https://docs.opencv.org/)
