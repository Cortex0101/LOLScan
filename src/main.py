# import video_processor module
from video_processor import VideoProcessor
from YOLO_labeller import YOLOLabeller

"""
if __name__ == "__main__":
    video_path = "data/videos/KESHAEUW_VS_DRUTUTT.mp4"
    processor = VideoProcessor(video_path, frame_skip=100, target_size=(640, 480))
    processor.process_all_frames()
"""


if __name__ == "__main__":
    labeller = YOLOLabeller()
    labeller.run()
