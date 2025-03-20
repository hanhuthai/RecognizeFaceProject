import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis
from utils.utils import register_flag, set_register_flag

# List of face angles to captutre
face_angles = {
    "front": False,
    "left": False,
    "right": False,
    "up": False,
    "down": False
}
face_embeddings = {}

# Initialize FAISS with 512 dimensions
index = faiss.IndexFlatL2(512)
def display_registered_angles(frame, face_angles):
    text = "Registered angles: " + ", ".join([angle for angle, registered in face_angles.items() if registered])
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def register_faceByFrame(frame):
    app = FaceAnalysis(providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    faces = app.get(frame)

    for face in faces:
        embedding = face.normed_embedding  # Vector 512D
        yaw, pitch, roll = face.pose  # Face pose angles
        print(f"Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}")

        # Yaw (Rotation around Y-axis - left/right)
        # Pitch (Rotation around X-axis - up/down)
        # Roll (Rotation around Z-axis - tilt left/right)
        # Determine face angle
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
        # Draw bounding box and label
        x, y, w, h = face.bbox.astype(int)
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    display_registered_angles(frame, face_angles)
    if all(face_angles.values()):
        print("✅ Captured all face angles!")
        set_register_flag(False)
        # Add to FAISS
        for key, emb in face_embeddings.items():
            index.add(np.array([emb], dtype=np.float32))

        # Save FAISS to file
        faiss.write_index(index, "face_db.index")
        print("✅ Saved 5 face angles to FAISS!")
   # cv2.imshow("Face Registration", frame)


def register_face():
    """ Thu thập 5 góc mặt, lưu FAISS và trả về embedding """

    # Initialize InsightFace with GPU
    app = FaceAnalysis(providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))


    index = faiss.IndexFlatL2(512)

    cap = cv2.VideoCapture(0)

    while not all(face_angles.values()):  # Run until all 5 angles are captured
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)

        for face in faces:
            embedding = face.normed_embedding  # Vector 512D
            yaw, pitch, roll = face.pose  # Face pose angles
            print(f"Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}")

            # Yaw (Rotation around Y-axis - left/right)
            # Pitch (Rotation around X-axis - up/down)
            # Roll (Rotation around Z-axis - tilt left/right)
            # Determine face angle
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
            # Draw bounding box and label
            x, y, w, h = face.bbox.astype(int)
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        display_registered_angles(frame, face_angles)
        cv2.imshow("Face Registration", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Add to FAISS
    for key, emb in face_embeddings.items():
        index.add(np.array([emb], dtype=np.float32))

    # Save FAISS to file
    faiss.write_index(index, "face_db.index")
    print("✅ Saved 5 face angles to FAISS!")

    return face_embeddings  # Trả về embedding để server xử lý tiếp
