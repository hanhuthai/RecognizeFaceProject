import cv2
import numpy as np
import faiss
import uvicorn
import tempfile
from fastapi import FastAPI, UploadFile, File
from insightface.app import FaceAnalysis
from starlette.middleware.cors import CORSMiddleware
from routers.face_controller import router as face_router

from database import init_db

app = FastAPI()
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
face_analyzer = FaceAnalysis(providers=['CUDAExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

faiss_index = faiss.IndexFlatL2(512)

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


# Khởi tạo DB
async def startup_event():
    await init_db()


app.add_event_handler("startup", startup_event)


app.include_router(face_router)

@app.get("/face-embeddings")
def get_face_embeddings():
    return {"face_embeddings": {key: emb.tolist() for key, emb in face_embeddings.items()}}


@app.post("/detect-video")
async def detect_faces(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        video_path = temp_video.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Không thể mở video!"}

    detected_faces = []

    while True:
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

            D, I = faiss_index.search(np.expand_dims(embedding, axis=0), 1)
            is_registered = D[0][0] < 1

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

