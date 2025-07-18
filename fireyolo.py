import cv2
import numpy as np
import os
import firebase_admin
from firebase_admin import credentials, firestore

# Load Firebase credentials
cred = credentials.Certificate(r"C:\Users\Jayesh\Desktop\yolo\yolo-cf82f-firebase-adminsdk-fbsvc-92d43984b9.json")
firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

# Paths
weights_path = r"C:\Users\Jayesh\Desktop\yolo\yolov3-tiny.weights"
config_path = r"C:\Users\Jayesh\Desktop\yolo\yolov3-tiny.cfg"
coco_path = r"C:\Users\Jayesh\Desktop\yolo\coco.names"
video_path = r"C:\Users\Jayesh\Desktop\yolo\Traffic jam\stock-footage-car-traffic-jam-on-the-highway.webm"
output_filename = r"C:\Users\Jayesh\Desktop\yolo\freetraffic.avi"

# Check if files exist
for path in [weights_path, config_path, coco_path, video_path]:
    if not os.path.exists(path):
        print(f" Error: File not found -> {path}")
        exit()

# Load YOLO model (without CUDA)
net = cv2.dnn.readNet(weights_path, config_path)

# Load class labels
classes = open(coco_path).read().strip().split("\n")
layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Open video
video_source = cv2.VideoCapture(video_path)
if not video_source.isOpened():
    print(" Error: Could not open video file")
    exit()

frame_width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_source.get(cv2.CAP_PROP_FPS))

output_video = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))

prev_positions = {}
slow_threshold = 2
frame_skip = 2  # Skip every 2nd frame
frame_count = 0

while True:
    frame_count += 1
    ret, frame = video_source.read()
    
    if not ret:
        break

    if frame_count % frame_skip != 0:
        continue  # Skip processing to improve speed

    # Reduce YOLO input size to 320x320 for faster processing
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(out_layers)

    boxes, confidences, class_ids = [], [], []
    confidence_threshold = 0.4

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold and classes[class_id] == "car":
                center_x, center_y, w, h = (detection[:4] * np.array([frame_width, frame_height, frame_width, frame_height])).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h), center_x, center_y])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes([b[:4] for b in boxes], confidences, confidence_threshold, 0.5)

    slow_count, total_count = 0, 0
    new_positions = {}

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h, cx, cy = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Calculate speed based on movement (basic tracking)
            if i in prev_positions:
                speed = np.linalg.norm(np.array(prev_positions[i]) - np.array([cx, cy]))
            else:
                speed = 0.0

            if speed < slow_threshold:
                slow_count += 1

            new_positions[i] = [cx, cy]
            total_count += 1

    prev_positions = new_positions
    traffic_status = "Traffic Jam" if total_count > 0 and (slow_count / total_count) > 0.4 else "Smooth Traffic"
    
    # Send Data to Firebase Firestore
    data = {
        "total_cars": total_count,
        "slow_cars": slow_count,
        "traffic_status": traffic_status
    }
    db.collection("traffic_status").document("latest").set(data)

    cv2.putText(frame, traffic_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    output_video.write(frame)

video_source.release()
output_video.release()
cv2.destroyAllWindows()

if os.path.exists(output_filename):
    print(f" Video processing complete. File saved at: {output_filename}")
    print(" Traffic status uploaded to Firebase Firestore")
else:
    print(" Error: Output video file not found.")
