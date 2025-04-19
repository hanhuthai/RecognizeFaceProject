import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis
from alg.ExtractEbedding import extract_feature, init_model

init_model("D:\\workspaces\\AI_portal\\RecognizeFaceProject\\ArcFace.onnx")

face_angles = {
    "front": False,
    "left": False,
    "right": False,
    "up": False,
    "down": False
}
face_embeddings = {}
captured_images = {}

index = faiss.IndexFlatL2(512)
def display_registered_angles(frame, face_angles):
    text = "Registered angles: " + ", ".join([angle for angle, registered in face_angles.items() if registered])
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def register_faceByFrame(frame):
    app = FaceAnalysis(providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    faces = app.get(frame)

    for face in faces:
        yaw, pitch, roll = face.pose
        angle_detected = None

        x1, y1, x2, y2 = map(int, face.bbox)

        if -10 < yaw < 10 and -10 < pitch < 10 and not face_angles["front"]:
            face_angles["front"] = True
            angle_detected = "front"
        elif yaw < -20 and -10 < pitch < 10 and not face_angles["down"]:
            face_angles["down"] = True
            angle_detected = "down"
        elif yaw > 20 and -10 < pitch < 10 and not face_angles["up"]:
            face_angles["up"] = True
            angle_detected = "up"
        elif -10 < yaw < 10 and pitch > 15 and not face_angles["left"]:
            face_angles["left"] = True
            angle_detected = "left"
        elif -10 < yaw < 10 and pitch < -15 and not face_angles["right"]:
            face_angles["right"] = True
            angle_detected = "right"

        if angle_detected:
            cropped_face = frame[y1:y2, x1:x2]
            face_embeddings[angle_detected] = extract_feature(cropped_face)
            print(face_embeddings[angle_detected])
            face_angles[angle_detected] = True

            captured_images[angle_detected] = cropped_face
            print(f"✅ Captured {angle_detected} face angle")

    registered_angles = [angle for angle, registered in face_angles.items() if registered]
    print("Registered angles so far:", registered_angles)
