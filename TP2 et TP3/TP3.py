import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
from KalmanFilter import KalmanFilter
from pathlib import Path

class MOTTracker:
    def __init__(self, max_lost=30, iou_threshold=0.3):
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_count = 0
        
    class Track:
        def __init__(self, bbox, track_id):
            self.id = track_id
            self.bbox = bbox
            self.lost_frames = 0
            
            # Initialize Kalman filter with parameters
            dt = 1.0  # Time step of 1 frame
            u_x, u_y = 1, 1 
            std_acc = 1.0  
            x_std_meas = 0.1 # Standard deviation of position measurement
            y_std_meas = 0.1
            
            self.kf = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
            
            # Initialize state with centroid position
            x, y, w, h = bbox
            centroid_x = x + w/2
            centroid_y = y + h/2
            self.w = w  # Store width and height separately
            self.h = h
            
            self.kf.x = np.array([[centroid_x], [centroid_y], [0], [0]])  # Initial state with zero velocity
            
        def predict(self):
            """Predict next state using Kalman filter"""
            predicted_centroid = self.kf.predict()
            # Convert centroid back to bbox format
            x = predicted_centroid[0, 0] - self.w/2
            y = predicted_centroid[1, 0] - self.h/2
            return [x, y, self.w, self.h]
            
        def update(self, bbox):
            """Update Kalman filter with new measurement"""
            x, y, w, h = bbox
            self.w = w  # Update stored width and height
            self.h = h
            centroid_x = x + w/2
            centroid_y = y + h/2
            
            measurement = np.array([[centroid_x], [centroid_y]])
            self.kf.update(measurement)
            
            # Update bbox with new Kalman state
            new_centroid = self.kf.x[:2]
            self.bbox = [
                new_centroid[0, 0] - w/2,  # x
                new_centroid[1, 0] - h/2,  # y
                w,
                h
            ]
            
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
            predicted_bbox = track.predict()
            for j, detection in enumerate(detections):
                similarity_matrix[i, j] = self.calculate_iou(predicted_bbox, detection)
                
        return similarity_matrix
        
    def update(self, detections):
        """Update tracks with new detections"""
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
            self.tracks[track_idx].update(detections[det_idx])
            self.tracks[track_idx].lost_frames = 0
            
        # Update unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].lost_frames += 1
            # Keep the predicted state
            self.tracks[track_idx].bbox = self.tracks[track_idx].predict()
            
        # Remove lost tracks
        self.tracks = [track for track in self.tracks if track.lost_frames < self.max_lost]
        
        # Create new tracks
        for det_idx in unmatched_detections:
            self.tracks.append(self.Track(detections[det_idx], self.track_count))
            self.track_count += 1
            
    def get_tracks(self):
        """Return current tracks"""
        return [(track.id, track.bbox) for track in self.tracks]


def process_sequence(sequence_path, det_path, output_path, visualize=True):
    """Process a sequence using the tracker"""
    # Load detections
    detections = np.loadtxt(det_path, delimiter=',')
    tracker = MOTTracker()
    
    # Initialize video writer if visualization is enabled
    if visualize:
        first_frame = cv2.imread(str(sequence_path / '000001.jpg'))
        height, width = first_frame.shape[:2]
        out = cv2.VideoWriter('tracking_result.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            30, (width, height))
    
    current_frame = 1
    results = []
    
    while True:
        # Load image
        img_path = sequence_path / f'{current_frame:06d}.jpg'
        if not img_path.exists():
            break
            
        frame = cv2.imread(str(img_path))
        if frame is None:
            break
            
        # Get detections for current frame
        frame_detections = detections[detections[:, 0] == current_frame]
        if len(frame_detections) > 0:
            bboxes = frame_detections[:, 2:6]  # [x, y, w, h]
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
        
    # Save results
    results = np.array(results)
    np.savetxt(output_path, results, delimiter=',', fmt='%d')
    
    if visualize:
        out.release()
        
    return results

sequence_path = Path("../ADL-Rundle-6/img1")
det_path = "../ADL-Rundle-6/det/public-dataset/det.txt"
output_path = "../results_TP3.txt"

results = process_sequence(sequence_path, det_path, output_path)