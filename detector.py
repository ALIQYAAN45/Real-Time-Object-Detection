"""
detector.py
-----------
Modular YOLOv8 Object Detection module using Ultralytics.
Handles model loading, object inference, confidence filtering,
class selection filtering, and bounding box visualization.
"""

import os
import cv2
from ultralytics import YOLO
import config


class ObjectDetector:
    """
    Wraps Ultralytics YOLO to perform real-time object detection
    with customizable confidence threshold and object class filtering.
    """

    def __init__(self, model_name=None, conf_threshold=None, selected_classes=None):
        """
        Initializes the YOLO model and filtering parameters.
        
        Parameters:
            model_name (str): Name or path to the model weights (e.g., 'yolov8n.pt').
            conf_threshold (float): Minimum confidence threshold (0.0 to 1.0).
            selected_classes (list[str], optional): List of object class names to detect/log.
        """
        self.model_name = model_name or config.MODEL_NAME
        
        # 1. Configurable confidence threshold (default from config: 0.70)
        self.conf_threshold = (
            float(conf_threshold)
            if conf_threshold is not None
            else float(config.DEFAULT_CONFIDENCE_THRESHOLD)
        )

        # 2. Configurable selected object classes
        raw_classes = selected_classes if selected_classes is not None else config.DEFAULT_SELECTED_OBJECTS
        self.selected_classes = set(c.strip().lower() for c in raw_classes if c.strip())

        # Model file path lookup (check local models/ directory first)
        local_model_path = os.path.join("models", self.model_name)
        target_path = local_model_path if os.path.exists(local_model_path) else self.model_name

        print(f"[INFO] Loading YOLO model: '{target_path}'...")
        print(f"[INFO] Confidence Threshold : {self.conf_threshold * 100:.0f}% ({self.conf_threshold:.2f})")
        print(f"[INFO] Selected Classes     : {sorted(list(self.selected_classes))}")

        self.model = YOLO(target_path)
        print("[INFO] YOLO model loaded successfully.")

        # Mapping of class IDs to class names (e.g., {0: 'person', 67: 'cell phone', ...})
        self.class_names = self.model.names

    def get_all_classes(self):
        """
        Returns a sorted list of all 80 COCO class names supported by YOLOv8.
        Useful for Streamlit selection widgets in later stages.
        """
        return sorted(list(self.class_names.values()))

    def set_confidence_threshold(self, new_threshold):
        """Update the minimum confidence threshold dynamically."""
        self.conf_threshold = float(new_threshold)

    def set_selected_classes(self, new_classes):
        """Update the set of selected object classes to detect/log."""
        self.selected_classes = set(c.strip().lower() for c in new_classes if c.strip())

    def is_valid_detection(self, class_name, confidence):
        """
        Checks whether a detected object meets both:
          1. Confidence threshold condition.
          2. Selected object filter condition.
          
        Parameters:
            class_name (str): The detected class name.
            confidence (float): The confidence score (0.0 to 1.0).
            
        Returns:
            bool: True if detection passes both filters, False otherwise.
        """
        # Condition 1: Confidence threshold
        if confidence < self.conf_threshold:
            return False

        # Condition 2: Selected-object filter
        if self.selected_classes and class_name.lower() not in self.selected_classes:
            return False

        return True

    def detect(self, frame):
        """
        Runs object detection on a frame and returns only detections
        that satisfy both the confidence threshold and the selected-object filter.
        
        Parameters:
            frame: BGR image array from OpenCV.
            
        Returns:
            list[dict]: Filtered list of detected objects.
        """
        if frame is None:
            return []

        # Run inference using Ultralytics YOLO
        # We pass a base conf of 0.25 to the raw model so YOLO finds candidate objects,
        # and our detector applies the strict configurable threshold and class filter.
        results = self.model(frame, conf=0.25, verbose=False)

        detections = []
        if not results:
            return detections

        first_result = results[0]

        for box in first_result.boxes:
            conf = float(box.conf[0].item())
            class_id = int(box.cls[0].item())
            class_name = self.class_names.get(class_id, f"class_{class_id}")

            # Apply Stage 3 filtering logic
            if not self.is_valid_detection(class_name, conf):
                continue

            # Coordinates in (x1, y1, x2, y2) format
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Calculate bbox_x, bbox_y, bbox_w, bbox_h for MySQL logging
            bbox_x = x1
            bbox_y = y1
            bbox_w = max(0, x2 - x1)
            bbox_h = max(0, y2 - y1)

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(conf, 4),
                "box": (x1, y1, x2, y2),
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
            })

        return detections

    def draw_detections(self, frame, detections):
        """
        Draws bounding boxes and labels for all accepted detections on the frame.
        
        Parameters:
            frame: OpenCV BGR image frame.
            detections (list[dict]): Filtered detection records.
            
        Returns:
            frame: The annotated OpenCV frame.
        """
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            class_name = det["class_name"]
            conf = det["confidence"]

            # Label format: e.g., "person 94%"
            label = f"{class_name} {int(conf * 100)}%"

            # Distinct cyan/amber bounding box color
            box_color = (255, 178, 50)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Draw background box for readable text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            label_top = max(y1 - text_h - baseline - 4, 0)
            cv2.rectangle(
                frame,
                (x1, label_top),
                (x1 + text_w + 6, label_top + text_h + baseline + 4),
                box_color,
                -1
            )

            # Draw text label
            cv2.putText(
                frame,
                label,
                (x1 + 3, label_top + text_h + 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

        return frame
