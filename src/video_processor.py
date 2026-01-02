"""Video processing utilities for LOLScan."""

import cv2
import logging
from pathlib import Path
from typing import Generator, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handle video frame extraction and processing."""

    def __init__(self, video_path: str, frame_skip: int = 1, target_size: Tuple[int, int] = None):
        """Initialize video processor.
        
        Args:
            video_path: Path to video file
            frame_skip: Process every Nth frame
            target_size: Resize frames to (width, height). If None, keep original size.
        """
        self.video_path = Path(video_path)
        self.frame_skip = frame_skip
        self.target_size = target_size
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video: {self.video_path.name}")
        logger.info(f"Frames: {self.total_frames}, FPS: {self.fps}, Size: {self.width}x{self.height}")

    def get_frames(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Generate video frames.
        
        Yields:
            Tuple of (frame, frame_number)
        """
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            if frame_count % self.frame_skip == 0:
                if self.target_size:
                    frame = cv2.resize(frame, self.target_size)
                
                yield frame, frame_count
            
            frame_count += 1
        
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
