import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
from pathlib import Path
import pandas as pd

class MOTTracker:
    def __init__(self, max_lost=30, iou_threshold=0.3):
        self.max_lost = max_lost  # Maximum number of frames object can be lost before removing track
        self.iou_threshold = iou_threshold  # Minimum IoU for match
        self.tracks = []  # List to store active tracks
        self.track_count = 0  # Counter for assigning unique IDs
        
    class Track:
        def __init__(self, bbox, track_id):
            self.id = track_id
            self.bbox = bbox
            self.lost_frames = 0
            
    def calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection coordinates
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
        
    def create_similarity_matrix(self, detections):
        """Create IoU-based similarity matrix between tracks and detections"""
        if not self.tracks or not detections:
            return np.array([])
            
        similarity_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                similarity_matrix[i, j] = self.calculate_iou(track.bbox, detection)
                
        return similarity_matrix
        
    def update(self, detections):
        """Update tracks with new detections"""
        # Convert detections to list of bounding boxes if necessary
        if not isinstance(detections, list):
            detections = detections.tolist()
            
        # Handle first frame
        if not self.tracks:
            for det in detections:
                self.tracks.append(self.Track(det, self.track_count))
                self.track_count += 1
            return
            
        # Create similarity matrix
        similarity_matrix = self.create_similarity_matrix(detections)
        
        if len(similarity_matrix) > 0:
            # Use Hungarian algorithm for assignment
            track_indices, detection_indices = linear_sum_assignment(-similarity_matrix)
            
            # Filter matches based on IoU threshold
            matches = []
            for track_idx, det_idx in zip(track_indices, detection_indices):
                if similarity_matrix[track_idx, det_idx] >= self.iou_threshold:
                    matches.append((track_idx, det_idx))
                    
            matched_tracks, matched_detections = zip(*matches) if matches else ([], [])
        else:
            matched_tracks, matched_detections = [], []
            
        # Handle unmatched tracks and detections
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_detections]
        
        # Update matched tracks
        for track_idx, det_idx in zip(matched_tracks, matched_detections):
            self.tracks[track_idx].bbox = detections[det_idx]
            self.tracks[track_idx].lost_frames = 0
            
        # Update unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].lost_frames += 1
            
        # Remove tracks that have been lost for too long
        self.tracks = [track for track in self.tracks if track.lost_frames < self.max_lost]
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self.tracks.append(self.Track(detections[det_idx], self.track_count))
            self.track_count += 1
            
    def get_tracks(self):
        """Return current tracks"""
        return [(track.id, track.bbox) for track in self.tracks]


def load_detections(det_file):
    """Load detections from MOT Challenge format file"""
    df = pd.read_csv(det_file, header=None)
    return df.values


def process_sequence(sequence_path, det_path, output_path, visualize=True):
    """Process a complete sequence"""
    # Load all detections
    detections = load_detections(det_path)
    tracker = MOTTracker()
    
    # Create output video writer if visualization is enabled
    if visualize:
        first_frame = cv2.imread(str(Path(sequence_path) / '000001.jpg'))
        height, width = first_frame.shape[:2]
        out = cv2.VideoWriter('tracking_result_TP2.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            30, (width, height))
    
    # Process frame by frame
    current_frame = 1
    results = []
    
    while True:
        # Load image
        img_path = Path(sequence_path) / f'{current_frame:06d}.jpg'
        if not img_path.exists():
            break
            
        frame = cv2.imread(str(img_path))
        if frame is None:
            break
            
        # Get detections for current frame
        frame_detections = detections[detections[:, 0] == current_frame]
        if len(frame_detections) > 0:
            # Extract bounding boxes
            bboxes = frame_detections[:, 2:6]  # [x, y, w, h]
            # Update tracker
            tracker.update(bboxes)
            
        # Get current tracks
        tracks = tracker.get_tracks()
        
        # Save results
        for track_id, bbox in tracks:
            x, y, w, h = bbox
            results.append([current_frame, track_id, x, y, w, h, 1, -1, -1, -1])
            
        # Visualize if enabled
        if visualize:
            for track_id, bbox in tracks:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(track_id), (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            out.write(frame)
            
        current_frame += 1
        
    # Save results to file
    results = np.array(results)
    np.savetxt(output_path, results, delimiter=',', fmt='%d')
    
    if visualize:
        out.release()
        
    return results


process_sequence("ADL-Rundle-6/img1","ADL-Rundle-6/det/public-dataset/det.txt","ADL-Rundle-6/output.txt")