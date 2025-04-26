# object_tracker.py
import numpy as np
from scipy.optimize import linear_sum_assignment

class ObjectTracker:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        """
        Simple object tracker
        
        Args:
            max_age: Maximum frames to keep a track alive without matching
            min_hits: Minimum hits to start tracking
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.next_id = 1
        self.tracks = {}  # Active tracks
        self.frame_count = 0
        
    def update(self, detections):
        """
        Update tracks with new detections
        
        Args:
            detections: List of detections [x1, y1, x2, y2, conf, class_id]
            
        Returns:
            Dict of active tracks with [track_id, bbox, class_id, age]
        """
        self.frame_count += 1
        
        # If no tracks yet, initialize with detections
        if not self.tracks:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                self.tracks[self.next_id] = {
                    'bbox': [x1, y1, x2, y2],
                    'class_id': cls,
                    'hits': 1,
                    'age': 0,
                    'last_seen': self.frame_count,
                    'time_since_update': 0
                }
                self.next_id += 1
            return self.tracks
            
        # Match detections to existing tracks
        if detections and self.tracks:
            # Compute IoU between existing tracks and new detections
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            track_indices = list(self.tracks.keys())
            
            for i, track_id in enumerate(track_indices):
                track = self.tracks[track_id]
                for j, det in enumerate(detections):
                    iou_matrix[i, j] = self._calculate_iou(track['bbox'], det[:4])
            
            # Hungarian algorithm for optimal assignment
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            
            # Update matched tracks
            matched_indices = []
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    track_id = track_indices[r]
                    x1, y1, x2, y2, conf, cls = detections[c]
                    
                    self.tracks[track_id]['bbox'] = [x1, y1, x2, y2]
                    self.tracks[track_id]['class_id'] = cls
                    self.tracks[track_id]['hits'] += 1
                    self.tracks[track_id]['time_since_update'] = 0
                    self.tracks[track_id]['last_seen'] = self.frame_count
                    
                    matched_indices.append((r, c))
            
            # Create new tracks for unmatched detections
            unmatched_detections = [i for i in range(len(detections))
                                    if not any(c == i for _, c in matched_indices)]
            
            for idx in unmatched_detections:
                x1, y1, x2, y2, conf, cls = detections[idx]
                self.tracks[self.next_id] = {
                    'bbox': [x1, y1, x2, y2],
                    'class_id': cls,
                    'hits': 1,
                    'age': 0,
                    'last_seen': self.frame_count,
                    'time_since_update': 0
                }
                self.next_id += 1
        
        # Update all tracks (increment age, mark missing)
        tracks_to_delete = []
        for track_id in list(self.tracks.keys()):
            # Increment time since update for unmatched tracks
            if self.tracks[track_id]['last_seen'] < self.frame_count:
                self.tracks[track_id]['time_since_update'] += 1
                
            # Delete old tracks
            if self.tracks[track_id]['time_since_update'] > self.max_age:
                tracks_to_delete.append(track_id)
                
            # Increment age for all tracks
            self.tracks[track_id]['age'] += 1
            
        # Remove dead tracks
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
            
        return self.tracks
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Determine intersection rectangle
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        # No intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # Calculate area of intersection rectangle
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate area of both bounding boxes
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Calculate IoU
        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        
        return max(0.0, min(1.0, iou))  # Ensure IoU is between 0 and 1