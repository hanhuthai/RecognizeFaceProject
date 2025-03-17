import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis

# Load FAISS index
index = faiss.read_index("face_db.index")

# Khởi tạo InsightFace
app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Kiểm tra khuôn mặt đã đăng ký
def is_face_registered(embedding, threshold=1):
    D, I = index.search(np.array([embedding], dtype=np.float32), 1)
    return D[0][0] < threshold

# Mở video thay vì camera
video_path = "C:\\Users\\lehuuchinh\\Downloads\\6414966157105.mp4"  # Đổi đường dẫn file video tại đây
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không thể mở video!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Thoát nếu hết video

    faces = app.get(frame)

    for face in faces:
        embedding = face.normed_embedding  # Vector 512D
        label = "Registered Face" if is_face_registered(embedding) else "Unknown Face"

        # Vẽ khung + hiển thị nhãn
        x, y, w, h = face.bbox.astype(int)
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Nhấn 'q' để thoát

cap.release()
cv2.destroyAllWindows()
