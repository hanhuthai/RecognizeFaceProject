import cv2
from RegisterFace import register_faceByFrame
rtsp_url = 'rtsp://192.168.1.49:8551/profile2/media.smp'
cap = cv2.VideoCapture(0)
frame_skip = 5
frame_count = 0
latest_frame = None

def generate_frames():
    global frame_count,latest_frame
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            latest_frame = frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')