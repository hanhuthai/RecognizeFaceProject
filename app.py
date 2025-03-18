import numpy as np
import uvicorn
from fastapi import FastAPI, Depends
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.cors import CORSMiddleware
from database import get_db
from models.face_model import FaceIndex
from routers.face_controller import router as face_router
from routers.face_detect import router as detect_router
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

@app.get("/get-embedding/{face_id}")
async def get_embedding(face_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(FaceIndex).filter(FaceIndex.id == face_id))
    face_data = result.scalar_one_or_none()

    if not face_data:
        return {"error": "Không tìm thấy dữ liệu!"}

    embedding_blob = face_data.embedding

    if len(embedding_blob) % 4 != 0:
        return {"error": "Dữ liệu embedding không hợp lệ!"}

    embedding_array = np.frombuffer(embedding_blob, dtype=np.float32)

    return {"embedding": embedding_array.tolist()}

async def startup_event():
    await init_db()

app.add_event_handler("startup", startup_event)

app.include_router(face_router)
app.include_router(detect_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

