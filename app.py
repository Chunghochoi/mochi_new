from flask import Flask, request, jsonify, Response
import cv2, numpy as np, mediapipe as mp, threading, time

app = Flask(__name__)

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,          # ✅ dùng pipeline streaming nhanh hơn
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

latest_frame = None
latest_gesture = "Waiting..."
frame_lock = threading.Lock()

def classify_gesture(hand_landmarks):
    lm = [(p.x, p.y) for p in hand_landmarks.landmark]
    def open_(tip,pip): return lm[tip][1] < lm[pip][1]
    f = [
        lm[4][0] < lm[3][0],
        open_(8,6),
        open_(12,10),
        open_(16,14),
        open_(20,18)
    ]
    n = f.count(True)
    if n==0: return "Fist"
    if n==5: return "Open Hand"
    if n==2 and f[1] and f[2]: return "Peace"
    if n==1 and f[0]: return "Thumbs Up"
    return f"Partial ({n})"

@app.route('/')
def home():
    return "⚡️ ESP32 Gesture Fast Server Running!"

@app.route('/upload', methods=['POST'])
def upload():
    global latest_frame, latest_gesture
    try:
        img = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error":"Invalid image"}),400

        # ✅ resize nhỏ để xử lý nhanh
        frame_small = cv2.resize(frame, (224, 224))

        # ✅ chuyển sang RGB và nhận diện tay
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb)

        gesture = "No hand"
        if results.multi_hand_landmarks:
            gesture = classify_gesture(results.multi_hand_landmarks[0])

        # ✅ Cập nhật frame & kết quả
        with frame_lock:
            latest_frame = frame_small.copy()
            latest_gesture = gesture

        # ✅ Phản hồi ngay (dưới 150 ms)
        return jsonify({"gesture": gesture})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==== Stream nhẹ (MJPEG) ====
def gen():
    global latest_frame, latest_gesture
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()
            cv2.putText(frame, latest_gesture, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ok, buf = cv2.imencode('.jpg', frame)
        if not ok: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buf.tobytes() + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # ✅ Gunicorn sẽ spawn nhiều worker, nhưng đây chỉ là local run
    app.run(host="0.0.0.0", port=5000, threaded=True)
