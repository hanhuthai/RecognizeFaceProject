import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis
face_angles = {
    "front": False,
    "left": False,
    "right": False,
    "up": False,
    "down": False
}
face_embeddings = {}
captured_images = {}  # Lưu ảnh theo góc mặt

# Initialize FAISS with 512 dimensions
index = faiss.IndexFlatL2(512)
def display_registered_angles(frame, face_angles):
    text = "Registered angles: " + ", ".join([angle for angle, registered in face_angles.items() if registered])
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

async def register_faceByFrame(frame, is_registering=True, app=None):
    if not is_registering:
        return  # Nếu đã dừng thì không xử lý tiếp

    # Ensure the FaceAnalysis object is provided
    if app is None:
        raise ValueError("FaceAnalysis object must be provided")

    # Check if frame is None before processing
    if frame is None:
        print("Warning: Frame is None, skipping processing")
        return
        
    faces = app.get(frame)

    for face in faces:
        if not is_registering:  # Kiểm tra liên tục để thoát ngay lập tức
            return
        embedding = face.normed_embedding 
        yaw, pitch, roll = face.pose  
        angle_detected = None
        if -10 < yaw < 10 and -10 < pitch < 10 and not face_angles["front"]:
            label = "Front"
            face_angles["front"] = True
            face_embeddings["front"] = embedding
            angle_detected = "front"
        elif yaw < -20 and -10 < pitch < 10 and not face_angles["down"]:
            label = "Down"
            face_angles["down"] = True
            face_embeddings["down"] = embedding
            angle_detected = "down"
        elif yaw > 20 and -10 < pitch < 10 and not face_angles["up"]:
            label = "Up"
            face_angles["up"] = True
            face_embeddings["up"] = embedding
            angle_detected = "up"
        elif -10 < yaw < 10 and pitch > 15 and not face_angles["left"]:
            label = "Left"
            face_angles["left"] = True
            face_embeddings["left"] = embedding
            angle_detected = "left"
        elif -10 < yaw < 10 and pitch < -15 and not face_angles["right"]:
            label = "Right"
            face_angles["right"] = True
            face_embeddings["right"] = embedding
            angle_detected = "right"
        if angle_detected:
            face_angles[angle_detected] = True
            face_embeddings[angle_detected] = embedding

            # Lưu ảnh vào biến tạm
            captured_images[angle_detected] = frame.copy()
            print(f"✅ Captured {angle_detected} face angle")

    registered_angles = [angle for angle, registered in face_angles.items() if registered]
    print("Registered angles so far:", registered_angles)

    # if all(face_angles.values()):
    #     print("✅ Captured all face angles!")
    #     for key, emb in face_embeddings.items():
    #         index.add(np.array([emb], dtype=np.float32))

    #     faiss.write_index(index, "face_db.index")
    #     print("✅ Saved 5 face angles to FAISS!")
