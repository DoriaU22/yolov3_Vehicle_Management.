import cv2
import numpy as np

# Load the pre-trained YOLO v3 network and the required files
# for cpu, it is recommended to use:
# net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
#  Less precision but more efficient in cases without gpu.
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()

# Adjust here to obtain the correct output layers
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Classes to detect
vehicle_classes = ["car", "bus", "motorbike", "truck", "bicycle"]

# Confidence threshold
confidence_threshold = 0.6  #  Increasing means being stricter with detections

# Open the video file
cap = cv2.VideoCapture("video_name.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width, channels = frame.shape

    # Processing the frame with YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Variables for storing results
    class_ids = []
    confidences = []
    boxes = []

    # Iterate on detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filtering only vehicle classes and with minimum confidence
            if confidence > confidence_threshold and classes[class_id] in vehicle_classes:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Coordinates of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply suppression of non-maximums to avoid overlapping charts
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Drawing bounding boxes and labels on the frame
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green color for all vehicles
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the processed frame
    cv2.imshow("Vehicle Detection", frame)

    # Press ‘q’ to exit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Freeing up resources
cap.release()
cv2.destroyAllWindows()
