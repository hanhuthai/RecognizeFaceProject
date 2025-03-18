import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis

index = faiss.read_index("face_db.index")

app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def is_face_registered(embedding, threshold=1):
    D, I = index.search(np.array([embedding], dtype=np.float32), 1)
    return D[0][0] < threshold

video_path = "C:\\Users\\lehuuchinh\\Downloads\\6414966157105.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không thể mở video!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    faces = app.get(frame)

    for face in faces:
        embedding = face.normed_embedding
        label = "Registered Face" if is_face_registered(embedding) else "Unknown Face"

        x, y, w, h = face.bbox.astype(int)
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
