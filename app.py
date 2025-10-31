[file: chunghochoi/mochi_new/mochi_new-9d3c4e3bf50c9319c5eed24367589aa29e0f2b4c/app.py]
from flask import Flask, request, jsonify, Response
import cv2, numpy as np, mediapipe as mp, threading, time
import queue # ✅ Thêm thư viện hàng đợi

app = Flask(__name__)

# ======== CẤU HÌNH MEDIAPIPE ========
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ======== BIẾN TOÀN CỤC (THREAD-SAFE) ========
latest_frame = None # Frame (ảnh) đã xử lý để hiển thị
latest_gesture = "Waiting..."
frame_lock = threading.Lock()
# ✅ Hàng đợi để chứa các khung hình thô từ ESP32 chờ xử lý
frame_queue = queue.Queue(maxsize=5)

# ======== PHÂN LOẠI CỬ CHỈ (Không đổi) ========
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

# ======== ✅ LUỒNG XỬ LÝ ẢNH (CHẠY NỀN) ========
# Luồng này liên tục lấy ảnh từ ESP32 (trong queue) để xử lý
def process_esp32_frames():
    global latest_frame, latest_gesture
    while True:
        try:
            # Lấy ảnh thô (bytes) từ hàng đợi
            img_data = frame_queue.get(block=True, timeout=10) 
            
            # Giải mã ảnh
            img = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
            if frame is None:
                continue 

            # Xử lý MediaPipe
            frame_small = cv2.resize(frame, (224, 224))
            rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb)

            gesture = "No hand"
            if results.multi_hand_landmarks:
                gesture = classify_gesture(results.multi_hand_landmarks[0])

            # Vẽ kết quả lên ảnh
            cv2.putText(frame_small, f"Gesture: {gesture}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Cập nhật biến toàn cục một cách an toàn
            with frame_lock:
                latest_frame = frame_small.copy()
                latest_gesture = gesture
        
        except queue.Empty:
            pass # Không có ảnh mới, tiếp tục chờ
        except Exception as e:
            print(f"Lỗi trong luồng xử lý: {e}")
            time.sleep(0.1)

# ======== ✅ UPLOAD ROUTE (SIÊU NHANH) ========
# Route này chỉ nhận ảnh từ ESP32 và ném vào hàng đợi
@app.route('/upload', methods=['POST'])
def upload():
    try:
        frame_queue.put_nowait(request.data)
        # Trả về 202 Accepted ngay lập tức, ESP32 không phải chờ
        return jsonify({"status": "received"}), 202
    except queue.Full:
        return jsonify({"error": "Queue full, frame dropped"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======== ✅ STREAM VIDEO TỪ ESP32 ========
# Luồng này stream ảnh đã xử lý (latest_frame) ra trình duyệt
def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            # Nếu chưa có ảnh, gửi ảnh đen
            if latest_frame is None:
                frame = np.zeros((224, 224, 3), dtype=np.uint8) # Kích thước 224x224
                cv2.putText(frame, "Waiting for ESP32...", (20, 112),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            else:
                # Lấy ảnh mới nhất đã được xử lý
                frame = latest_frame.copy()

            # Mã hóa ảnh thành JPEG để gửi đi
            ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Gửi ảnh dưới dạng multipart
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

        # Stream ở 10 FPS
        time.sleep(0.1)  

@app.route('/video')
def video_feed():
    # Trả về stream từ hàm generate_frames
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def home():
    return "✅ ESP32-CAM Gesture Server — Fast & Non-blocking Stream!"

# ======== KHỞI ĐỘNG LUỒNG XỬ LÝ NỀN ========
processing_thread = threading.Thread(target=process_esp32_frames, daemon=True)
processing_thread.start()

if __name__ == '__main__':
    # threaded=True cho phép upload + stream song song
    app.run(host="0.0.0.0", port=5000, threaded=True)
