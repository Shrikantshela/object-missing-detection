# main.py
import cv2
import time
import argparse
import numpy as np
from object_detector import ObjectDetector
from object_tracker import ObjectTracker

# COCO class names for reference
COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
                'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
                'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush','remote','notebook']

def main():
    parser = argparse.ArgumentParser(description='Real-time detection of missing and new objects in video')
    parser.add_argument('--input', type=str, default='0', help='Path to input video file or camera index (default: 0)')
    parser.add_argument('--output', type=str, default='', help='Path to output video file (optional)')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model to use')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--missing-frames', type=int, default=5, help='Frames to wait before declaring object as missing')
    parser.add_argument('--new-frames', type=int, default=5, help='Frames to wait before declaring object as new')
    
    args = parser.parse_args()
    
    # Initialize detector and tracker
    detector = ObjectDetector(model_name=args.model, conf_threshold=args.conf)
    tracker = ObjectTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # Initialize video capture
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)
        
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.input}")
        return
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output is specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_avg = 0
    
    # Dictionary to store objects
    current_objects = {}
    previous_objects = {}
    missing_objects = {}
    new_objects = {}
    
    # Lists to store missing and new objects for visualization
    missing_visuals = []
    new_visuals = []
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Calculate FPS
        if frame_count % 30 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps_avg = 30 / elapsed_time if elapsed_time > 0 else 0
            start_time = time.time()
            print(f"FPS: {fps_avg:.2f}")
        
        # Detect objects
        detections = detector.detect(frame)
        
        # Update tracker
        tracks = tracker.update(detections)
        
        # Convert tracks to current_objects dictionary
        current_objects = {}
        for track_id, track in tracks.items():
            if track['hits'] >= tracker.min_hits:  # Only consider confirmed tracks
                current_objects[track_id] = {
                    'bbox': track['bbox'],
                    'class_id': track['class_id'],
                    'last_seen': frame_count
                }
        
        # Find missing objects
        if previous_objects:
            for obj_id in previous_objects:
                if obj_id not in current_objects:
                    # Object is missing
                    if obj_id not in missing_objects:
                        missing_objects[obj_id] = {
                            'bbox': previous_objects[obj_id]['bbox'],
                            'class_id': previous_objects[obj_id]['class_id'],
                            'first_missing': frame_count,
                            'confirmed': False
                        }
                else:
                    # Object is still present, remove from missing
                    if obj_id in missing_objects:
                        del missing_objects[obj_id]
        
        # Update missing objects status
        missing_visuals = []
        for obj_id in list(missing_objects.keys()):
            # Check if object has been missing long enough
            if frame_count - missing_objects[obj_id]['first_missing'] > args.missing_frames:
                missing_objects[obj_id]['confirmed'] = True
                missing_visuals.append(missing_objects[obj_id])
            
            # If object reappears, remove from missing
            if obj_id in current_objects:
                del missing_objects[obj_id]
        
        # Find new objects
        for obj_id in current_objects:
            if obj_id not in previous_objects:
                # New object detected
                if obj_id not in new_objects:
                    new_objects[obj_id] = {
                        'bbox': current_objects[obj_id]['bbox'],
                        'class_id': current_objects[obj_id]['class_id'],
                        'first_seen': frame_count,
                        'confirmed': False
                    }
            
        # Update new objects status
        new_visuals = []
        for obj_id in list(new_objects.keys()):
            # Check if object has been present long enough
            if frame_count - new_objects[obj_id]['first_seen'] > args.new_frames:
                new_objects[obj_id]['confirmed'] = True
                
                # Update with current position
                if obj_id in current_objects:
                    new_objects[obj_id]['bbox'] = current_objects[obj_id]['bbox']
                    
                new_visuals.append(new_objects[obj_id])
            
            # If object disappears or has been around too long, remove from new
            if obj_id not in current_objects or frame_count - new_objects[obj_id]['first_seen'] > 3 * args.new_frames:
                if obj_id in new_objects:
                    del new_objects[obj_id]
        
        # Draw results
        # Draw current objects
        for obj_id, obj in current_objects.items():
            x1, y1, x2, y2 = map(int, obj['bbox'])
            class_id = obj['class_id']
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"Class {class_id}"
            
            # Draw bounding box (blue)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw label
            label = f"{class_name} #{obj_id}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw missing objects (red)
        for obj in missing_visuals:
            if obj['confirmed']:
                x1, y1, x2, y2 = map(int, obj['bbox'])
                class_id = obj['class_id']
                class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"Class {class_id}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"MISSING: {class_name}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw new objects (green)
        for obj in new_visuals:
            if obj['confirmed']:
                x1, y1, x2, y2 = map(int, obj['bbox'])
                class_id = obj['class_id']
                class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"Class {class_id}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"NEW: {class_name}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add FPS text
        cv2.putText(frame, f"FPS: {fps_avg:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Real-time Object Detection', frame)
        
        # Write frame if output is specified
        if writer:
            writer.write(frame)
        
        # Update previous objects for next frame
        previous_objects = current_objects.copy()
        
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if writer:
        writer.write(frame)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()