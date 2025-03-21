# video_stream.py
import cv2
from RegisterFace import register_faceByFrame
import utils.utils as utils


cap = cv2.VideoCapture(0)
frame_skip = 5  # Chỉ xử lý 1 frame trong mỗi 5 frame
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
            # Register face
            # print("generate_frames-Register flag:", utils.register_flag)
            # if utils.register_flag:
            #     register_faceByFrame(frame)

            #cv2.putText(frame, "Hello World", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # convert JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # return  MJPEG thread
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')