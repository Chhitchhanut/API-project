from ultralytics import YOLO
import cv2
import collections

# Load pre-trained YOLO model (custom-trained on fruits or general object detection)
model = YOLO("path_to_your_fruit_model.pt")  # You need a trained model here

# Load image
image_path = "image6.png"
image = cv2.imread(image_path)

# Run detection
results = model(image)

# Get detections
boxes = results[0].boxes
names = model.names  # class index to label mapping

# Dictionary to hold fruit counts
fruit_counts = collections.Counter()

# Draw results
for box in boxes:
    cls_id = int(box.cls[0])
    label = names[cls_id]
    fruit_counts[label] += 1

    # Get bounding box coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = box.conf[0]

    # Draw box and label
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show total count
for fruit, count in fruit_counts.items():
    print(f"{fruit}: {count}")

# Show result image
cv2.imshow("Detected Fruits", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
