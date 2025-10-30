from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
import mediapipe as mp
import threading

app = Flask(__name__)

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

latest_frame = None
frame_lock = threading.Lock()

# ==== Phân loại cử chỉ ====
def classify_gesture(hand_landmarks):
    lm = [(p.x, p.y) for p in hand_landmarks.landmark]
    def is_finger_open(tip, pip): return lm[tip][1] < lm[pip][1]
    fingers = [
        lm[4][0] < lm[3][0],                # thumb
        is_finger_open(8, 6),               # index
        is_finger_open(12, 10),             # middle
        is_finger_open(16, 14),             # ring
        is_finger_open(20, 18)              # pinky
    ]
    up_count = fingers.count(True)
    if up_count == 0:  return "Fist"
    if up_count == 5:  return "Open Hand"
    if up_count == 2 and fingers[1] and fingers[2]: return "Peace"
    if up_count == 1 and fingers[0]: return "Thumbs Up"
    return f"Partial ({up_count} fingers)"

# ==== Route chính ====
@app.route('/')
def home():
    return "✅ ESP32-CAM Gesture Server (with live camera view)"

@app.route('/upload', methods=['POST'])
def upload():
    global latest_frame
    try:
        img_array = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb)

        gesture = "No hand"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            gesture = classify_gesture(results.multi_hand_landmarks[0])

        # Lưu frame mới nhất để hiển thị
        with frame_lock:
            latest_frame = frame.copy()

        return jsonify({"gesture": gesture})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==== Stream MJPEG ====
def generate_stream():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', latest_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
