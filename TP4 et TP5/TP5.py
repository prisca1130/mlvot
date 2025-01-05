from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

class PedestrianDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.3):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
    def detect(self, frame):
        """
        Detect pedestrians in a frame
        Returns: array of [x, y, w, h, conf]
        """
        results = self.model(frame, classes=0)  # class 0 is person in COCO
        detections = []
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                if box.conf[0] > self.conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0]
                    w = x2 - x1
                    h = y2 - y1
                    conf = box.conf[0]
                    detections.append([x1, y1, w, h, conf])
                    
        return np.array(detections)

def generate_detections_file(sequence_path, output_path):
    """
    Generate det.txt file for a sequence
    Format: <frame_id>,<track_id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
    """
    detector = PedestrianDetector()
    sequence_path = Path(sequence_path)
    
    # Get all frame paths
    frame_paths = sorted(sequence_path.glob('*.jpg'))
    detections_list = []
    
    print(frame_paths)
    # Process each frame
    for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc="Processing frames")):
        # Read frame
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
            
        # Get detections
        frame_dets = detector.detect(frame)
        
        # Add to detections list
        for det in frame_dets:
            x, y, w, h, conf = det
            # Format: frame_id, -1 (track_id), x, y, w, h, conf, -1, -1, -1
            detections_list.append([frame_idx + 1, -1, x, y, w, h, conf, -1, -1, -1])
    
    print(detections_list)

    detections_array = np.array(detections_list, dtype=object)
    
    # Save detections
    np.savetxt(output_path, detections_array, delimiter=',', fmt=['%d', '%d', '%.2f', '%.2f', 
                                                                  '%.2f', '%.2f', '%.2f', '%d', '%d', '%d'])

if __name__ == "__main__":
    # Example usage
    sequence_path = "../ADL-Rundle-6/img1"
    output_path = "results_TP5.txt"
    
    generate_detections_file(sequence_path, output_path)