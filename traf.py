import cv2
import numpy as np
import os

weights_path = r"C:\Users\Jayesh\Desktop\yolo\yolov3-tiny.weights"
config_path = r"C:\Users\Jayesh\Desktop\yolo\yolov3-tiny.cfg"
coco_path = r"C:\Users\Jayesh\Desktop\yolo\coco.names"
video_path = r"C:\Users\Jayesh\Desktop\yolo\Smooth Traffic\Untitled video - Made with Clipchamp (1).mp4"
output_filename = r"C:\Users\Jayesh\Desktop\yolo\output_video1.avi"

for path in [weights_path, config_path, coco_path, video_path]:
    if not os.path.exists(path):
        print(f"❌ Error: File not found -> {path}")
        exit()

net = cv2.dnn.readNet(weights_path, config_path)
classes = open(coco_path).read().strip().split("\n")

layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

video_source = cv2.VideoCapture(video_path)
if not video_source.isOpened():
    print("❌ Error: Could not open video file")
    exit()

frame_width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_source.get(cv2.CAP_PROP_FPS))

output_video = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))

prev_positions = {}
slow_threshold = 2

while True:
    ret, frame = video_source.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
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

            closest_obj_id = None
            min_distance = float("inf")

            for obj_id, prev_pos in prev_positions.items():
                distance = np.linalg.norm(np.array(prev_pos) - np.array([cx, cy]))
                if distance < min_distance:
                    min_distance = distance
                    closest_obj_id = obj_id

            speed = min_distance if closest_obj_id is not None else 0.0

            if speed < slow_threshold:
                slow_count += 1

            new_positions[i] = [cx, cy]
            total_count += 1

    prev_positions = new_positions
    traffic_status = "Traffic Jam" if total_count > 0 and (slow_count / total_count) > 0.4 else "Smooth Traffic"
    
    cv2.putText(frame, traffic_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    output_video.write(frame)

video_source.release()
output_video.release()
cv2.destroyAllWindows()

if os.path.exists(output_filename):
    print(f"✅ Video processing complete. File saved at: {output_filename}")
else:
    print("❌ Error: Output video file not found.")
