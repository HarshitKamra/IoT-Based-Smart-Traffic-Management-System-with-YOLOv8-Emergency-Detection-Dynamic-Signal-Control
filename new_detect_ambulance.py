from ultralytics import YOLO

# Use raw string to avoid Windows path issues (note the r before the string)
model = YOLO(r"C:\Users\HP\runs\detect\ambulance_yolov8n\weights\best.pt")

# Run on webcam
model.predict(source=0, show=True, conf=0.75)
