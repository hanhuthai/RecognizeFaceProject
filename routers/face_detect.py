from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from database import AsyncSessionLocal
from services.face_detect_service import detect_faces_webcam

router = APIRouter(prefix="/detect", tags=["Face Detection"])

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@router.get("")
async def detect_faces(db: AsyncSession = Depends(get_db)):
    return await detect_faces_webcam(db)
