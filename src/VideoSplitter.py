import cv2
import logging
from pathlib import Path
from typing import Generator, Tuple
import numpy as np

# import config 
import config as cfg

logger = logging.getLogger(__name__)

class VideoSplitter:
    """
    VideoSplitter opens a video file and allows user to step through the video at
    a specified frame interval (incremented by w and decremented by s keys). It displays
    each frame in a window and allows the user to save frames as images by pressing the spacebar.

    Attributes:
        video_path (Path): Path to the video file.
        frame_skip (int): Number of frames to skip when stepping through the video. Default is 1.
        output_name: Defaults to name of the video file without extension. For example, "video.mp4" -> "video". video dir will have images and labels subdirs created.
    """
    def __init__(self, video_path: str, frame_skip: int = 1):
        self.video_path = Path(video_path)
        self.frame_skip = frame_skip
        self.output_name = self.video_path.stem
        self.current_frame_idx = 0
        self.last_read_position = -1  # Track the last frame position we read from the video

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

    def _reopen_capture(self) -> None:
        """Reopen the video capture to reset decoder state."""
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to reopen video: {self.video_path}")
        self.last_read_position = -1
        logger.debug("Video capture reopened")

    def get_frame(self, frame_idx: int) -> Tuple[bool, np.ndarray]:
        """
        Get a specific frame from the video with optimized reading.
        Uses sequential reading when possible, reopens capture when seeking backwards.
        
        Args:
            frame_idx: The frame index to retrieve.
            
        Returns:
            Tuple of (success, frame) where success is True if frame was read successfully.
        """
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return False, None
        
        # If seeking backwards or position is unknown, reopen to avoid decoder degradation
        if frame_idx < self.last_read_position:
            logger.debug(f"Seeking backwards from {self.last_read_position} to {frame_idx}, reopening capture")
            self._reopen_capture()
        
        # If we need to jump forward more than 100 frames, use seek
        # Otherwise, read sequentially to avoid seek overhead
        frames_to_skip = frame_idx - self.last_read_position - 1
        
        if frames_to_skip > 100 or self.last_read_position == -1:
            # Large jump or first read, use seek
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret:
                self.last_read_position = frame_idx
            return ret, frame
        else:
            # Sequential read - skip intermediate frames
            for _ in range(frames_to_skip):
                ret = self.cap.grab()  # Fast skip without decoding
                if not ret:
                    return False, None
                self.last_read_position += 1
            
            # Read the target frame
            ret, frame = self.cap.read()
            if ret:
                self.last_read_position = frame_idx
            return ret, frame

    def save_frame(self, current_frame: np.ndarray = None) -> None:
        """
        Save the current frame as an image file in the output directory.
        
        Args:
            current_frame: The frame to save. If None, will read the current frame.
        """
        output_image_dir = Path(cfg.Config().get("data.images_dir", "data/images")) / self.output_name
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir = Path(cfg.Config().get("data.labels_dir", "data/labels")) / self.output_name
        output_label_dir.mkdir(parents=True, exist_ok=True)

        if current_frame is None:
            ret, frame = self.get_frame(self.current_frame_idx)
            if not ret:
                logger.error(f"Failed to read frame at index {self.current_frame_idx}")
                return
        else:
            frame = current_frame

        frame_filename = output_image_dir / f"{self.output_name}_frame_{self.current_frame_idx:05d}.png"
        label_filename = output_label_dir / f"{self.output_name}_frame_{self.current_frame_idx:05d}.txt"
        
        cv2.imwrite(str(frame_filename), frame)
        with open(label_filename, 'w') as f:
            f.write("")  # Create an empty label file
            
        logger.info(f"Saved frame {self.current_frame_idx} to {frame_filename}")
        

    def handle_keypress(self, key: int, current_frame: np.ndarray, frame_number: int) -> bool:
        """
        Handle keypress events.

        Args:
            key (int): The key code of the pressed key.
            current_frame (np.ndarray): The current video frame.
            frame_number (int): The current frame number.

        Returns:
            bool: True to continue, False to exit.
        """
        if key == ord('q'):
            return False  # Quit
        elif key == ord('d'):
            self.current_frame_idx = min(self.current_frame_idx + self.frame_skip, self.total_frames - 1)
        elif key == ord('a'):
            self.current_frame_idx = max(self.current_frame_idx - self.frame_skip, 0)
        elif key == ord('w'):
            self.frame_skip = self.frame_skip + 1
        elif key == ord('s'):
            self.frame_skip = max(self.frame_skip - 1, 1)
        elif key == ord(' '):
            self.save_frame(current_frame)
        return True

if __name__ == "__main__":
    video_file = "C:\\dev\\LOLScan\\LOLScan\\data\\videos\\CaitVLux.mp4"  # Replace with your video file path
    splitter = VideoSplitter(video_file, frame_skip=5)

    # Read the initial frame
    ret, frame = splitter.get_frame(splitter.current_frame_idx)
    
    if not ret:
        logger.error("Failed to read initial frame.")
    else:
        previous_frame_idx = splitter.current_frame_idx
        
        while True:
            # Only read a new frame if the frame index has changed
            if splitter.current_frame_idx != previous_frame_idx:
                ret, frame = splitter.get_frame(splitter.current_frame_idx)

                if not ret:
                    logger.info("End of video reached.")
                    break
                
                previous_frame_idx = splitter.current_frame_idx

            display_frame = frame.copy()
            cv2.putText(display_frame, f"Frame: {splitter.current_frame_idx}/{splitter.total_frames}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Video Splitter", display_frame)
            key = cv2.waitKey(0) & 0xFF

            if not splitter.handle_keypress(key, frame, splitter.current_frame_idx):
                break