from flask import Flask, request, jsonify, Response
import cv2, numpy as np, mediapipe as mp, threading, time

app = Flask(__name__)

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

latest_frame = None
latest_gesture = "Waiting..."
frame_lock = threading.Lock()

# ======== PHÂN LOẠI CỬ CHỈ ========
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


# ======== XỬ LÝ ẢNH GỬI TỪ ESP32 ========
@app.route('/upload', methods=['POST'])
def upload():
    global latest_frame, latest_gesture
    try:
        img = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error":"Invalid image"}),400

        # resize nhỏ để xử lý nhanh
        frame_small = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb)

        gesture = "No hand"
        if results.multi_hand_landmarks:
            gesture = classify_gesture(results.multi_hand_landmarks[0])

        # vẽ text nhỏ trên khung hình
        cv2.putText(frame_small, f"Gesture: {gesture}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        with frame_lock:
            latest_frame = frame_small.copy()
            latest_gesture = gesture

        return jsonify({"gesture": gesture})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ======== STREAM VIDEO KHÔNG BLOCK ========
def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for ESP32...", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            else:
                frame = latest_frame.copy()

            ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

        time.sleep(0.2)  # ✅ Giới hạn ~5 FPS để không nghẽn Flask thread

@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def home():
    return "✅ ESP32-CAM Gesture Server — Fast & Non-blocking Stream!"


if __name__ == '__main__':
    # ✅ threaded=True cho phép upload + stream song song
    app.run(host="0.0.0.0", port=5000, threaded=True)
