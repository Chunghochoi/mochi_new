from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)

# Sử dụng MediaPipe Hands (qua mediapipe-runtime)
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

@app.route('/')
def home():
    return "✅ ESP32-CAM Gesture Server (mediapipe-runtime) is running!"

@app.route('/upload', methods=['POST'])
def upload():
    try:
        img_array = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb)

        gesture = "No hand"
        if results.multi_hand_landmarks:
            gesture = "Hand detected"

        return jsonify({"gesture": gesture})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.
