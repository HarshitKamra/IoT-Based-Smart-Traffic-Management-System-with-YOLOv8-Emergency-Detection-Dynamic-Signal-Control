
from ultralytics import YOLO
import cv2

# Load the YOLOv8 model (smallest version for speed)
model = YOLO("yolov8n.pt")

# Open webcam (0 = default webcam)
cap = cv2.VideoCapture(0)

# YOLO COCO vehicle class IDs
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on the current frame
    results = model(frame)
    boxes = results[0].boxes
    count = 0

    # Count vehicle-class detections only
    for cls in boxes.cls:
        if int(cls) in vehicle_classes:
            count += 1

    # Logic to estimate green time
    if count > 20:
        green_time = 40
    elif count > 10:
        green_time = 25
    else:
        green_time = 15

    # Plot results and overlay info
    annotated = results[0].plot()
    cv2.putText(annotated, f'Vehicle Count: {count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated, f'Green Time: {green_time}s', (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show result
    cv2.imshow("Live Traffic Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
