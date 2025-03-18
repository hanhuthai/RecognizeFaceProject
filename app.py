import cv2
import numpy as np
import faiss
import uvicorn
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.params import Depends
from insightface.app import FaceAnalysis
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.cors import CORSMiddleware

from database import get_db

app = FastAPI()
origins = [
    "http://localhost:3000",  # FE đang chạy trên localhost:3000 (React, Vite,...)
    "http://127.0.0.1:3000",
    "*"  # Cho phép tất cả (KHÔNG NÊN DÙNG TRÊN PRODUCTION)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Chỉ định domain cho phép
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức (GET, POST, PUT, DELETE, ...)
    allow_headers=["*"],  # Cho phép tất cả các headers
)
# Khởi tạo InsightFace với GPU
face_analyzer = FaceAnalysis(providers=['CUDAExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# Khởi tạo FAISS với 512 dimensions
faiss_index = faiss.IndexFlatL2(512)

# Danh sách góc mặt cần đăng ký
face_angles = {
    "front": False,
    "left": False,
    "right": False,
    "up": False,
    "down": False
}
face_embeddings = {}


@app.get("/status")
def get_status():
    return {"status": "API is running"}


@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    global face_angles, face_embeddings

    # Reset trạng thái nhận diện
    face_angles = {key: False for key in face_angles}
    face_embeddings = {}

    # Lưu video tạm thời
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as buffer:
        buffer.write(file.file.read())

    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Không thể mở video!"}

    while not all(face_angles.values()):  # Chạy đến khi đủ 5 góc mặt
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_analyzer.get(frame)

        for face in faces:
            if face.normed_embedding is None:
                continue  # Bỏ qua nếu không có embedding

            embedding = np.array(face.normed_embedding, dtype=np.float32)  # Vector 512D
            if embedding.shape[0] != 512:
                continue  # Đảm bảo đúng số chiều

            yaw, pitch, roll = face.pose  # Góc quay khuôn mặt

            # Xác định góc khuôn mặt
            if -10 < yaw < 10 and -10 < pitch < 10 and not face_angles["front"]:
                face_angles["front"] = True
                face_embeddings["front"] = embedding
            elif yaw < -20 and -10 < pitch < 10 and not face_angles["left"]:
                face_angles["left"] = True
                face_embeddings["left"] = embedding
            elif yaw > 20 and -10 < pitch < 10 and not face_angles["right"]:
                face_angles["right"] = True
                face_embeddings["right"] = embedding
            elif -10 < yaw < 10 and pitch > 15 and not face_angles["up"]:
                face_angles["up"] = True
                face_embeddings["up"] = embedding
            elif -10 < yaw < 10 and pitch < -15 and not face_angles["down"]:
                face_angles["down"] = True
                face_embeddings["down"] = embedding

    cap.release()

    # Lưu embeddings vào FAISS
    for key, emb in face_embeddings.items():
        faiss_index.add(np.expand_dims(emb, axis=0))  # Đảm bảo đúng shape (1, 512)

    # Ghi dữ liệu FAISS ra file
    faiss.write_index(faiss_index, "face_db.index")

    return {
        "message": "✅ Video processed successfully",
        "angles_detected": {key: bool(value) for key, value in face_angles.items()}
    }


@app.get("/face-embeddings")
def get_face_embeddings():
    return {"face_embeddings": {key: emb.tolist() for key, emb in face_embeddings.items()}}


# API kiểm tra khuôn mặt đã đăng ký
@app.post("/detect-video")
async def detect_faces(file: UploadFile = File(...)):
    """API nhận video và kiểm tra khuôn mặt"""

    # Lưu video tạm thời
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        video_path = temp_video.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Không thể mở video!"}

    detected_faces = []  # Lưu danh sách khuôn mặt phát hiện

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Thoát nếu hết video

        faces = face_analyzer.get(frame)

        for face in faces:
            if face.normed_embedding is None:
                continue  # Bỏ qua nếu không có embedding

            embedding = np.array(face.normed_embedding, dtype=np.float32)  # Vector 512D
            if embedding.shape[0] != 512:
                continue  # Đảm bảo đúng số chiều

            # Tìm kiếm trong FAISS
            D, I = faiss_index.search(np.expand_dims(embedding, axis=0), 1)
            is_registered = D[0][0] < 1  # Ngưỡng nhận diện

            label = "Registered" if is_registered else "Unknown"
            x, y, w, h = map(int, face.bbox)

            detected_faces.append({
                "x": x, "y": y, "w": w, "h": h,
                "label": label
            })

    cap.release()

    return {
        "message": "✅ Face detection completed!",
        "detected_faces": detected_faces
    }

@app.get("/test-db")
async def test_db(session: AsyncSession = Depends(get_db)):
    return {"message": "✅ Database connection is working!"}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

