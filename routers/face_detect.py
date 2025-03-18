import cv2
import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from insightface.app import FaceAnalysis
from database import AsyncSessionLocal
from sqlalchemy import text


router = APIRouter(prefix="/detect", tags=["Face Detection"])

# Khởi tạo model nhận diện khuôn mặt
face_analyzer = FaceAnalysis(providers=['CUDAExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@router.get("")
async def detect_faces_webcam(db: AsyncSession = Depends(get_db)):
    cap = cv2.VideoCapture(0)  # Mở webcam

    if not cap.isOpened():
        return {"error": "Không thể mở webcam!"}

    detected_faces = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_analyzer.get(frame)  # Phát hiện khuôn mặt

        for face in faces:
            if face.normed_embedding is None:
                continue

            embedding = np.array(face.normed_embedding, dtype=np.float32)

            # Truy vấn database để tìm vector gần nhất
            result = await db.execute(text("SELECT id, user_id, name, embedding FROM face_index"))
            face_data = result.fetchall()

            min_distance = float("inf")
            best_match = None

            for row in face_data:
                db_embedding = np.frombuffer(row[3], dtype=np.float32)
                distance = np.linalg.norm(embedding - db_embedding)

                if distance < min_distance:
                    min_distance = distance
                    best_match = row

            if best_match and min_distance < 1.0:  # Ngưỡng nhận diện
                label = f"User: {best_match[2]}"  # Tên người dùng
            else:
                label = "Unknown"

            # Vẽ khung nhận diện
            x, y, w, h = map(int, face.bbox)
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return {"message": "✅ Face detection completed!", "detected_faces": detected_faces}
