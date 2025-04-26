# object_detector.py
from ultralytics import YOLO
import cv2
import torch

class ObjectDetector:
    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.25):
        """
        Initialize the object detector
        
        Args:
            model_name: YOLOv8 model to use
            conf_threshold: Confidence threshold for detection
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load the model
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        
    def detect(self, frame):
        """
        Detect objects in a frame
        
        Args:
            frame: Image frame to process
            
        Returns:
            List of detections with [x1, y1, x2, y2, confidence, class_id]
        """
        # Run inference
        results = self.model(frame, conf=self.conf_threshold)
        
        # Extract detections
        detections = []
        if results and len(results) > 0:
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    detections.append([x1, y1, x2, y2, conf, cls])
                    
        return detections