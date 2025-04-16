import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis
from alg.ExtractEbedding import extract_feature, init_model  # Import the extract_feature and init_model functions

# Initialize the ONNX model globally
init_model("D:\\workspaces\\AI_portal\\RecognizeFaceProject\\ArcFace.onnx")  # Replace "path_to_model.onnx" with the actual model path

# List of face angles to captutre
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

def register_faceByFrame(frame):
    app = FaceAnalysis(providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    faces = app.get(frame)

    for face in faces:
        #embedding = face.normed_embedding
        yaw, pitch, roll = face.pose
        angle_detected = None

        # Crop the face using the bounding box
        x1, y1, x2, y2 = map(int, face.bbox)

        if -10 < yaw < 10 and -10 < pitch < 10 and not face_angles["front"]:
            face_angles["front"] = True
            #face_embeddings["front"] = embedding
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
            face_embeddings[angle_detected] = extract_feature(cropped_face)  # Extract feature from the cropped face
            print(face_embeddings[angle_detected])
            face_angles[angle_detected] = True
            #face_embeddings[angle_detected] = embedding

            # Save the cropped face instead of the entire frame
            captured_images[angle_detected] = cropped_face
            print(f"✅ Captured {angle_detected} face angle")

    registered_angles = [angle for angle, registered in face_angles.items() if registered]
    print("Registered angles so far:", registered_angles)

    # if all(face_angles.values()):
    #     print("✅ Captured all face angles!")
    #     for key, emb in face_embeddings.items():
    #         index.add(np.array([emb], dtype=np.float32))

    #     faiss.write_index(index, "face_db.index")
    #     print("✅ Saved 5 face angles to FAISS!")
