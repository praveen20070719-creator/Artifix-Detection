import cv2
import requests
import time
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# Config - Use your specific credentials
API_KEY = "8Xvcz3aZlYhB56qAFxt7s04RbGVUEgoJWINkOywPHTuCSMD1pLErO3a4D6R10Zx8cPYiSdIuLfwsmkXy"
MY_NUMBER = "9342827687,9361656061"
MIN_CONFIDENCE = 0.8
SMS_COOLDOWN = 300 
last_sms_time = 0

model = YOLO('yolov8n.pt')

def send_alert_sms(conf_score):
    global last_sms_time
    url = "https://www.fast2sms.com/dev/bulkV2"
    message = f"ARTIFIX: Elephant detected ({conf_score:.1%}). Alert Officer Abishek: 9361656061."
    payload = {"route": "q", "message": message, "language": "english", "numbers": MY_NUMBER}
    headers = {"authorization": API_KEY}
    try:
        requests.post(url, data=payload, headers=headers)
    except Exception as e:
        print(f"SMS Error: {e}")

def generate_frames():
    global last_sms_time
    # Use 0 for local webcam or a URL for an IP camera
    cap = cv2.VideoCapture(0) 
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        results = model(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 20: # 20 is elephant in COCO
                    conf = float(box.conf[0])
                    if conf >= MIN_CONFIDENCE:
                        # Draw boxes
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        
                        # SMS Logic
                        curr = time.time()
                        if (curr - last_sms_time) > SMS_COOLDOWN:
                            send_alert_sms(conf)
                            last_sms_time = curr

        # Encode the frame so the browser can read it
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)