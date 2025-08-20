# IoT-Based-Smart-Traffic-Management-System-with-YOLOv8-Emergency-Detection-Dynamic-Signal-Control
Our project is a software-based AI system that detects emergency and normal vehicles from live video input using YOLOv8 and simulates smart traffic control logic based on real-time detection and vehicle counting. 
It is fully functional using a laptop and webcam, with no hardware dependencies.
Detects emergency vehicles and overrides red signals in real time
Counts vehicles per frame to estimate traffic density
Adjusts green signal duration (simulated)
Sends email alerts to hospitals/authorities using Python SMTP
Runs end-to-end on laptop – low cost, portable, and demo-ready
AI-only Detection – No GPS, RFID, or sensors; camera-based YOLOv8 detection.
Vehicle Counting + Density-Based Logic – Uses class-wise count for adaptive signal control.
Email Alerts – Sends alerts to hospitals upon emergency detection and tell their location where they are detected.
Software-Only Prototype – Works entirely via Python, OpenCV, webcam.
Dataset Merging via Roboflow – Combined public datasets for custom YOLOv8 training.
