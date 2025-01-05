import cv2
import numpy as np
from KalmanFilter import KalmanFilter
from Detector import detect

# Parameters
dt = 0.1
u_x, u_y = 1, 1
std_acc = 1
x_sdt_meas, y_sdt_meas = 0.1, 0.1

# Initialize Kalman Filter
kf = KalmanFilter(dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas)

# Video capture
cap = cv2.VideoCapture('randomball.avi')

# Get video properties (frame width, height, fps) to set up VideoWriter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Set up VideoWriter to save the video
fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # You can use other codecs like 'MP4V' for .mp4
out = cv2.VideoWriter('tracked_output.mp4', fourcc, fps, (frame_width, frame_height))

# Store trajectory
trajectory = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    centers = detect(frame)

    # Use the first detected object if available
    if centers:
        # Measurement (z): Use the first detected object's centroid
        z = centers[0]
        kf.update(z)
    else:
        z = None

    # Kalman Filter prediction
    predicted = kf.predict()

    # Visualization
    for center in centers:
        cX, cY = int(center[0][0]), int(center[1][0])
        cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)  # Green circle for detected object

    # Draw predicted position
    predicted_pos = (int(predicted[0, 0]), int(predicted[1, 0]))
    cv2.rectangle(frame, (predicted_pos[0]-5, predicted_pos[1]-5),
                  (predicted_pos[0]+5, predicted_pos[1]+5), (255, 0, 0), 2)  # Blue rectangle for prediction

    if z is not None:
        # Draw estimated position
        estimated_pos = (int(kf.x[0, 0]), int(kf.x[1, 0]))
        trajectory.append(estimated_pos)
        cv2.rectangle(frame, (estimated_pos[0]-10, estimated_pos[1]-10),
                      (estimated_pos[0]+10, estimated_pos[1]+10), (0, 0, 255), 2)  # Red rectangle for estimate

    # Draw trajectory
    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i-1], trajectory[i], (0, 255, 255), 2)  # Yellow trajectory line
        
    # Write the frame with the tracking results to the output video
    out.write(frame)

    # Display frame
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
