import cv2
import os
import json
import logging
import numpy as np
from pathlib import Path

# import config 
import config as cfg
logger = logging.getLogger(__name__)

class YOLOLabeller:
    def __init__(self):
        self.classes = [
            "champion",  # Default class
            "enemy",
            "ally",
            "minion",
            "turret",
            "jungle_monster"
        ]

        self.class_colors = {
            0: (255, 0, 0),    # champion - blue
            1: (0, 0, 255),    # enemy - red
            2: (0, 255, 0),    # ally - green
            3: (255, 255, 0),  # minion - cyan
            4: (255, 0, 255),  # turret - magenta
            5: (0, 255, 255)   # jungle_monster - yellow
        }

        self.images_dir = Path(cfg.Config().get('data.images_dir', './data/images'))
        self.labels_dir = Path(cfg.Config().get('data.labels_dir', './data/labels'))
        
        # Labeling state
        self.current_frame_idx = 0
        self.frames = []
        self.image_files = []
        self.annotations = {}  # {image_name: [(class_id, x_center, y_center, width, height), ...]}
        
        # Drawing state
        self.drawing = False
        self.start_point = None
        self.current_box = None
        self.current_class = 0

        self.load_annotations()

        # open window
        cv2.namedWindow("YOLO Labeller")
        cv2.setMouseCallback("YOLO Labeller", self.mouse_events)

    def mouse_events(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box = (self.start_point, (x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            end_point = (x, y)
            self.add_annotation(self.start_point, end_point)
            self.current_box = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            # remove the annotation if clicked inside
            image_name = self.image_files[self.current_frame_idx]
            if image_name in self.annotations:
                to_remove = None
                for annot in self.annotations[image_name]:
                    class_id, x_center, y_center, width, height = annot
                    x1 = int((x_center - width / 2) * self.frames[self.current_frame_idx].shape[1])
                    y1 = int((y_center - height / 2) * self.frames[self.current_frame_idx].shape[0])
                    x2 = int((x_center + width / 2) * self.frames[self.current_frame_idx].shape[1])
                    y2 = int((y_center + height / 2) * self.frames[self.current_frame_idx].shape[0])
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        to_remove = annot
                        break
                if to_remove:
                    self.annotations[image_name].remove(to_remove)
                    logger.info(f"Removed annotation from {image_name}: {to_remove}")

    def key_events(self, key):
        if key == ord('q'):
            return 'quit'
        elif key == ord('d'):
            self.current_frame_idx = min(self.current_frame_idx + 1, len(self.frames) - 1)
        elif key == ord('a'):
            self.current_frame_idx = max(self.current_frame_idx - 1, 0)
        elif key >= ord('0') and key <= ord(str(len(self.classes) - 1)):
            self.current_class = key - ord('0')
            logger.info(f"Switched to class {self.current_class}: {self.classes[self.current_class]}")
        elif key == ord('z'):
            # Undo last annotation
            image_name = self.image_files[self.current_frame_idx]
            if image_name in self.annotations and self.annotations[image_name]:
                removed_annot = self.annotations[image_name].pop()
                logger.info(f"Undid annotation on {image_name}: {removed_annot}")

    def load_annotations(self):
        for label_file in self.labels_dir.glob("*.txt"):
            image_name = label_file.stem + ".jpg"
            with open(label_file, 'r') as f:
                annots = []
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annots.append((class_id, x_center, y_center, width, height))
                self.annotations[image_name] = annots
        logger.info(f"Loaded annotations for {len(self.annotations)} images.")

    def add_annotation(self, start_point, end_point):
        x1, y1 = start_point
        x2, y2 = end_point
        x_center = (x1 + x2) / 2 / self.frames[self.current_frame_idx].shape[1]
        y_center = (y1 + y2) / 2 / self.frames[self.current_frame_idx].shape[0]
        width = abs(x2 - x1) / self.frames[self.current_frame_idx].shape[1]
        height = abs(y2 - y1) / self.frames[self.current_frame_idx].shape[0]
        
        image_name = self.image_files[self.current_frame_idx]
        if image_name not in self.annotations:
            self.annotations[image_name] = []
        
        self.annotations[image_name].append((
            self.current_class,
            x_center,
            y_center,
            width,
            height
        ))
        logger.info(f"Added annotation to {image_name}: class {self.current_class}, box {start_point} to {end_point}")

    def save_annotations(self):
        for image_name, annots in self.annotations.items():
            label_file = self.labels_dir / f"{Path(image_name).stem}.txt"
            with open(label_file, 'w') as f:
                for annot in annots:
                    f.write(" ".join(map(str, annot)) + "\n")
            logger.info(f"Saved annotations to {label_file}")

    def load_images(self):
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.png'))])
        for img_file in self.image_files:
            img_path = self.images_dir / img_file
            img = cv2.imread(str(img_path))
            self.frames.append(img)
        logger.info(f"Loaded {len(self.frames)} images for labeling.")

    def run(self):
        self.load_images()
        
        while True:
            frame = self.frames[self.current_frame_idx].copy()
            
            # Draw existing annotations
            image_name = self.image_files[self.current_frame_idx]
            if image_name in self.annotations:
                for annot in self.annotations[image_name]:
                    class_id, x_center, y_center, width, height = annot
                    x1 = int((x_center - width / 2) * frame.shape[1])
                    y1 = int((y_center - height / 2) * frame.shape[0])
                    x2 = int((x_center + width / 2) * frame.shape[1])
                    y2 = int((y_center + height / 2) * frame.shape[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.class_colors[class_id], 2)
                    cv2.putText(frame, self.classes[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw current box
            if self.current_box:
                cv2.rectangle(frame, self.current_box[0], self.current_box[1], (255, 0, 0), 2)

            # draw current class info
            cv2.putText(frame, f"Class: {self.current_class} - {self.classes[self.current_class]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("YOLO Labeller", frame)
            key = cv2.waitKey(1) & 0xFF
            
            action = self.key_events(key)
            if action == 'quit':
                break
        
        cv2.destroyAllWindows()
        self.save_annotations()

    