import cv2
import numpy as np
import faiss
import tempfile
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from insightface.app import FaceAnalysis
from models.face_model import FaceIndex
import pickle


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

    result = await session.execute(stmt)
    await session.commit()

    return result


face_analyzer = FaceAnalysis(providers=['CUDAExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

try:
    faiss_index = faiss.read_index("face_db.index")
except:
    faiss_index = faiss.IndexFlatL2(512)

face_angles = {
    "front": False,
    "left": False,
    "right": False,
    "up": False,
    "down": False
}
face_embeddings = {}

async def process_video(file, user_id: int, name: str, age: int, address: str, phonenumber: str, db: AsyncSession):
    global face_angles, face_embeddings

    face_angles = {key: False for key in face_angles}
    face_embeddings = {}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        video_path = temp_video.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Không thể mở video!"}

    while not all(face_angles.values()):
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_analyzer.get(frame)

        for face in faces:
            if face.normed_embedding is None:
                continue

            embedding = np.array(face.normed_embedding, dtype=np.float32)  # Vector 512D
            if embedding.shape[0] != 512:
                continue

            yaw, pitch, roll = face.pose

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

    for angle, emb in face_embeddings.items():
        emb_bytes = pickle.dumps(emb)
        await save_face_to_db(db, user_id, angle, name, age, address, phonenumber, emb_bytes)

    return {
        "message": "✅ Video processed successfully",
        "angles_detected": {key: bool(value) for key, value in face_angles.items()}
    }
