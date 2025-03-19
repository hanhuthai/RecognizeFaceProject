import cv2
import numpy as np
import uuid
from insightface.app import FaceAnalysis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
from models.face_model import FaceIndex

async def save_face_to_db(session: AsyncSession, user_id: str, angle: str, name: str, age: int, address: str,
                          phone: str, embedding: bytes):
    stmt = insert(FaceIndex).values(
        user_id=user_id,
        angle=angle,
        name=name,
        age=age,
        address=address,
        phonenumber=phone,
        embedding=embedding
    )
    await session.execute(stmt)
    await session.commit()

async def capture_face(name: str, age: int, address: str, phonenumber: str, db: AsyncSession):
    user_id = str(uuid.uuid4())  # ✅ Tự động sinh UUID
    print(f"🔹 User ID mới: {user_id}")

    face_analyzer = FaceAnalysis(providers=['CUDAExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(0)  # Mở webcam
    if not cap.isOpened():
        return {"error": "Không thể mở webcam!"}

    face_angles = {"front": False, "left": False, "right": False, "up": False, "down": False}
    face_embeddings = {}

    def display_registered_angles(frame, face_angles):
        """Hiển thị các góc đã đăng ký trên màn hình"""
        detected = [angle for angle, registered in face_angles.items() if registered]
        text = "Đã nhận diện: " + ", ".join(detected) if detected else "Hãy xoay mặt theo hướng khác!"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    while not all(face_angles.values()):
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_analyzer.get(frame)
        for face in faces:
            if face.normed_embedding is None:
                continue

            embedding = np.array(face.normed_embedding, dtype=np.float32)
            if embedding.shape[0] != 512:
                continue

            yaw, pitch, roll = face.pose
            detected_angle = None

            if -10 < yaw < 10 and -10 < pitch < 10 and not face_angles["front"]:
                detected_angle = "front"
            elif yaw < -20 and -10 < pitch < 10 and not face_angles["left"]:
                detected_angle = "left"
            elif yaw > 20 and -10 < pitch < 10 and not face_angles["right"]:
                detected_angle = "right"
            elif -10 < yaw < 10 and pitch > 15 and not face_angles["up"]:
                detected_angle = "up"
            elif -10 < yaw < 10 and pitch < -15 and not face_angles["down"]:
                detected_angle = "down"

            if detected_angle:
                face_angles[detected_angle] = True
                face_embeddings[detected_angle] = embedding
                print(f"✅ Nhận diện góc {detected_angle}")

            # ✅ Vẽ bounding box
            x, y, w, h = face.bbox.astype(int)
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, detected_angle or "Adjust Position", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        display_registered_angles(frame, face_angles)  # Hiển thị trạng thái
        cv2.imshow("Face Registration", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
            break

    cap.release()
    cv2.destroyAllWindows()

    async def save_all():
        for angle, emb in face_embeddings.items():
            emb_bytes = emb.tobytes()
            await save_face_to_db(db, user_id, angle, name, age, address, phonenumber, emb_bytes)

    await save_all()

    return {
        "message": "✅ Đăng ký khuôn mặt thành công!",
        "user_id": user_id,
        "angles_detected": {key: bool(value) for key, value in face_angles.items()}
    }
