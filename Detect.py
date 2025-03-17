import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis

# Load the saved FAISS index
index = faiss.read_index("face_db.index")

# Initialize InsightFace with GPU
app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


# Function to check if the detected face matches any saved embeddings
# 0.6-1  càng nhỏ càng nghiêm ngặt, há miệng sẽ ko bắt được
def is_face_registered(embedding, threshold=1):
    D, I = index.search(np.array([embedding], dtype=np.float32), 1)
    return D[0][0] < threshold


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở camera!")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        embedding = face.normed_embedding  # Vector 512D
        if is_face_registered(embedding):
            label = "Registered Face"
        else:
            label = "Unknown Face"

        # Draw bounding box and label
        x, y, w, h = face.bbox.astype(int)
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()