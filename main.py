import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis

app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

index = faiss.IndexFlatL2(512)

face_angles = {
    "front": False,
    "left": False,
    "right": False,
    "up": False,
    "down": False
}
face_embeddings = {}

def display_registered_angles(frame, face_angles):
    text = "Registered angles: " + ", ".join([angle for angle, registered in face_angles.items() if registered])
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cap = cv2.VideoCapture(0)

while not all(face_angles.values()):
    ret, frame = cap.read()
    if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
        break

    faces = app.get(frame)

    for face in faces:
        embedding = face.normed_embedding
        yaw, pitch, roll = face.pose
        print(f"Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}")

        if -10 < yaw < 10 and -10 < pitch < 10 and not face_angles["front"]:
            label = "Front"
            face_angles["front"] = True
            face_embeddings["front"] = embedding
        elif yaw < -20 and -10 < pitch < 10 and not face_angles["down"]:
            label = "Down"
            face_angles["down"] = True
            face_embeddings["down"] = embedding
        elif yaw > 20 and -10 < pitch < 10 and not face_angles["up"]:
            label = "Up"
            face_angles["up"] = True
            face_embeddings["up"] = embedding
        elif -10 < yaw < 10 and pitch > 15 and not face_angles["left"]:
            label = "Left"
            face_angles["left"] = True
            face_embeddings["left"] = embedding
        elif -10 < yaw < 10 and pitch < -15 and not face_angles["right"]:
            label = "Right"
            face_angles["right"] = True
            face_embeddings["right"] = embedding
        else:
            label = "Adjust Position"

        x, y, w, h = face.bbox.astype(int)
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    display_registered_angles(frame, face_angles)
    cv2.imshow("Face Registration", frame)

cap.release()
cv2.destroyAllWindows()

# Lưu embeddings vào FAISS
for key, emb in face_embeddings.items():
    index.add(np.array([emb], dtype=np.float32))

faiss.write_index(index, "face_db.index")
print("✅ Saved 5 face angles to FAISS!")
