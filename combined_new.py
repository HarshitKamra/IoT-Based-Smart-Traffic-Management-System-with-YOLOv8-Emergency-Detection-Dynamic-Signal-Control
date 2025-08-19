from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import smtplib
import winsound  # Sound alert
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from twilio.rest import Client

#  Twilio SMS Function
def send_sms_alert_twilio(message_body):
    account_sid = 'AC51a6d1a70a00289c8f1ccbd2db276640'
    auth_token = 'b899bc732deb43652087ba08f93f9302'
    from_number = '+18564741264'
    to_number = '+91787611200'

    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=message_body,
        from_=from_number,
        to=to_number
    )
    print(f" SMS sent: {message.sid}")

#  Email Function (Only for Ambulance)
def send_alert_to_hospital(location, image_path):
    sender_email = "smart.traffic.alertss@gmail.com"
    sender_password = "kzzdwqqiilyacnyi"
    recipient_email = "kamraharshit26@gmail.com"

    subject = "🚨 Emergency Alert: Ambulance Detected"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    body = f"""Ambulance detected near: {location}
Time: {timestamp}
Google Maps: https://www.google.com/maps/search/?api=1&query={location.replace(' ', '+')}"""

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    with open(image_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
        msg.attach(part)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print(" Email sent to hospital.")
        send_sms_alert_twilio("🚨 Ambulance detected at Thapar University. Check email for snapshot.")
    except Exception as e:
        print(" Email sending failed:", e)

#  Load YOLO Models
model_coco = YOLO("yolov8n.pt")
model_custom = YOLO(r"C:\\Users\\HP\\runs\\detect\\ambulance_yolov8n\\weights\\best.pt")
custom_class_names = model_custom.names
print("🔍 Custom class names:", custom_class_names)

#  COCO vehicle classes
coco_vehicle_classes = [2, 3, 5, 7]

#  Webcam Setup
cap = cv2.VideoCapture(0)
email_sent = False  # Flag to prevent multiple emails for ambulance

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results_coco = model_coco(frame, conf=0.5)  # Higher confidence for regular vehicles
    results_custom = model_custom(frame, conf=0.75)  # Updated: Use trained model with 0.75 conf

    annotated = frame.copy()
    vehicle_count = 0
    emergency_count = 0
    ambulance_detected = False
    firetruck_detected = False

    #  Count normal vehicles
    for box in results_coco[0].boxes:
        cls_id = int(box.cls.item())
        if cls_id in coco_vehicle_classes:
            vehicle_count += 1

    #  Emergency vehicle detection and handling
    for box in results_custom[0].boxes:
        if box.conf < 0.7:
            continue
        cls_id = int(box.cls.item())
        class_name = custom_class_names[cls_id].replace(" ", "").lower()

        if class_name == "ambulance":
            ambulance_detected = True
            emergency_count += 1
            print(" Ambulance Detected")
            winsound.Beep(1000, 1000)

            if not email_sent:
                snapshot_path = f"ambulance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(snapshot_path, frame)
                send_alert_to_hospital("Thapar University, Patiala", snapshot_path)
                email_sent = True

        elif class_name == "firetruck":
            firetruck_detected = True
            emergency_count += 1
            print(" Firetruck Detected (no hospital alert sent)")
            winsound.Beep(1000, 1000)

    if not ambulance_detected:
        email_sent = False

    if emergency_count > 0:
        green_time = 60
    elif vehicle_count > 20:
        green_time = 40
    elif vehicle_count > 10:
        green_time = 25
    else:
        green_time = 15

    annotated = results_coco[0].plot(img=annotated)
    annotated = results_custom[0].plot(img=annotated)

    cv2.putText(annotated, f'Vehicles: {vehicle_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(annotated, f'Emergencies: {emergency_count}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(annotated, f'Green Time: {green_time}s', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if ambulance_detected:
        cv2.putText(annotated, 'AMBULANCE ALERT SENT', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if firetruck_detected:
        cv2.putText(annotated, 'FIRETRUCK DETECTED', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Traffic + Emergency Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
