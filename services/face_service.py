# import cv2
# import numpy as np
# import asyncio
# from insightface.app import FaceAnalysis
# from sqlalchemy.ext.asyncio import AsyncSession
# from models.face_model import FaceIndex
# from sqlalchemy import insert
#
# async def save_face_to_db(session: AsyncSession, user_id: int, angle: str, name: str, age: int, address: str,
#                           phone: str, embedding: bytes):
#     stmt = insert(FaceIndex).values(
#         user_id=user_id,
#         angle=angle,
#         name=name,
#         age=age,
#         address=address,
#         phonenumber=phone,
#         embedding=embedding
#     )
#     await session.execute(stmt)
#     await session.commit()
#
# async def capture_face(user_id: int, name: str, age: int, address: str, phonenumber: str, db: AsyncSession):
#     face_analyzer = FaceAnalysis(providers=['CUDAExecutionProvider'])
#     face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
#
#     cap = cv2.VideoCapture(0)  # Mở webcam
#     if not cap.isOpened():
#         return {"error": "Không thể mở webcam!"}
#
#     face_angles = {"front": False, "left": False, "right": False, "up": False, "down": False}
#     face_embeddings = {}
#
#     while not all(face_angles.values()):
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         faces = face_analyzer.get(frame)
#         for face in faces:
#             if face.normed_embedding is None:
#                 continue
#
#             embedding = np.array(face.normed_embedding, dtype=np.float32)
#             if embedding.shape[0] != 512:
#                 continue
#
#             # Lấy tọa độ bounding box
#             bbox = face.bbox.astype(int)
#             x1, y1, x2, y2 = bbox
#
#             # Vẽ bounding box quanh khuôn mặt
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#             yaw, pitch, roll = face.pose
#             detected_angle = None
#
#             if -10 < yaw < 10 and -10 < pitch < 10 and not face_angles["front"]:
#                 detected_angle = "front"
#             elif yaw < -20 and -10 < pitch < 10 and not face_angles["left"]:
#                 detected_angle = "left"
#             elif yaw > 20 and -10 < pitch < 10 and not face_angles["right"]:
#                 detected_angle = "right"
#             elif -10 < yaw < 10 and pitch > 15 and not face_angles["up"]:
#                 detected_angle = "up"
#             elif -10 < yaw < 10 and pitch < -15 and not face_angles["down"]:
#                 detected_angle = "down"
#
#             if detected_angle:
#                 face_angles[detected_angle] = True
#                 face_embeddings[detected_angle] = embedding
#
#                 # ✅ Hiển thị text trên màn hình khi góc này đã được detect
#                 cv2.putText(frame, f"{detected_angle.upper()} DETECTED", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#
#         # ✅ Hiển thị khung hình webcam
#         cv2.imshow("Webcam", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # Bấm 'q' để thoát
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()  # Đóng cửa sổ webcam sau khi hoàn tất
#
#     async def save_all():
#         for angle, emb in face_embeddings.items():
#             emb_bytes = emb.tobytes()
#             await save_face_to_db(db, user_id, angle, name, age, address, phonenumber, emb_bytes)
#
#     await save_all()
#
#     return {
#         "message": "✅ Webcam processed successfully",
#         "angles_detected": {key: bool(value) for key, value in face_angles.items()}
#     }
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
from models.face_model import FaceIndex

async def save_face_to_db(session: AsyncSession, user_id: int, angle: str, name: str, age: int, address: str,
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

async def capture_face(user_id: int, name: str, age: int, address: str, phonenumber: str, db: AsyncSession):
    face_analyzer = FaceAnalysis(providers=['CUDAExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(0)  # Mở webcam
    if not cap.isOpened():
        return {"error": "Không thể mở webcam!"}

    face_angles = {"front": False, "left": False, "right": False, "up": False, "down": False}
    face_embeddings = {}

    def display_registered_angles(frame, face_angles):
        """Hiển thị các góc đã đăng ký trên màn hình"""
        text = "Đã nhận diện: " + ", ".join([angle for angle, registered in face_angles.items() if registered])
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
            label = "Adjust Position"

            if -10 < yaw < 10 and -10 < pitch < 10 and not face_angles["front"]:
                label = "Front"
                face_angles["front"] = True
                face_embeddings["front"] = embedding
            elif yaw < -20 and -10 < pitch < 10 and not face_angles["left"]:
                label = "Left"
                face_angles["left"] = True
                face_embeddings["left"] = embedding
            elif yaw > 20 and -10 < pitch < 10 and not face_angles["right"]:
                label = "Right"
                face_angles["right"] = True
                face_embeddings["right"] = embedding
            elif -10 < yaw < 10 and pitch > 15 and not face_angles["up"]:
                label = "Up"
                face_angles["up"] = True
                face_embeddings["up"] = embedding
            elif -10 < yaw < 10 and pitch < -15 and not face_angles["down"]:
                label = "Down"
                face_angles["down"] = True
                face_embeddings["down"] = embedding

            # ✅ Vẽ bounding box
            x, y, w, h = face.bbox.astype(int)
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        display_registered_angles(frame, face_angles)  # Hiển thị thông tin các góc đã nhận diện
        cv2.imshow("Face Registration", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
            break

    cap.release()
    cv2.destroyAllWindows()  # Đóng cửa sổ webcam sau khi hoàn tất

    async def save_all():
        for angle, emb in face_embeddings.items():
            emb_bytes = emb.tobytes()
            await save_face_to_db(db, user_id, angle, name, age, address, phonenumber, emb_bytes)

    await save_all()

    return {
        "message": "✅ Đăng ký khuôn mặt thành công!",
        "angles_detected": {key: bool(value) for key, value in face_angles.items()}
    }
