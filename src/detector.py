"""YOLO detector wrapper for object detection."""

import logging
from pathlib import Path
from typing import List, Dict, Any
import torch

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO-based object detector."""

    def __init__(self, model_name: str = "yolov8m", device: str = "cuda"):
        """Initialize YOLO detector.
        
        Args:
            model_name: YOLOv8 model size (n, s, m, l, x)
            device: Device to use (cuda or cpu)
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics package not installed. Install with: pip install ultralytics")
        
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading {model_name} on {self.device}")
        self.model = YOLO(f"{model_name}.pt")
        self.model.to(self.device)
        
        logger.info(f"Model loaded successfully")

    def detect(self, image: Any, conf: float = 0.5) -> List[Dict]:
        """Detect objects in image.
        
        Args:
            image: Input image (numpy array or tensor)
            conf: Confidence threshold
        
        Returns:
            List of detections with bounding boxes and class information
        """
        results = self.model.predict(image, conf=conf, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                detection = {
                    'bbox': box.xyxy[0].cpu().numpy(),  # [x1, y1, x2, y2]
                    'conf': float(box.conf[0].cpu().numpy()),
                    'class_id': int(box.cls[0].cpu().numpy()),
                    'class_name': result.names[int(box.cls[0])],
                }
                detections.append(detection)
        
        return detections

    def load_weights(self, weights_path: str):
        """Load custom trained weights.
        
        Args:
            weights_path: Path to .pt weights file
        """
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        
        logger.info(f"Loading weights from {weights_path}")
        self.model = self.model.__class__(weights_path)
        self.model.to(self.device)

    def __del__(self):
        """Cleanup GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
