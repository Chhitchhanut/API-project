import cv2
from ultralytics import YOLO

# 1. Load a YOLOv8 pre-trained model
# By default, yolov8n.pt is trained on COCO (80 classes, includes apple, orange, banana)
model = YOLO("yolov8n.pt")  

# 2. Load image
image_path = "image6.png"  # replace with your fruit image
image = cv2.imread(image_path)

# 3. Run detection
results = model(image)

# 4. Initialize counter
fruit_counts = {}

# 5. Process results
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])       # class index
        conf = float(box.conf[0])     # confidence score
        label = model.names[cls_id]   # class name (e.g. apple, banana, orange)
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box

        # Count fruits
        fruit_counts[label] = fruit_counts.get(label, 0) + 1

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 6. Print results
print("Fruit Counts:")

for fruit, count in fruit_counts.items():
    print(f"{fruit}: {count}")

print(f"Total fruits: {sum(fruit_counts.values())}")

# 7. Show output
cv2.imshow("Fruit Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
