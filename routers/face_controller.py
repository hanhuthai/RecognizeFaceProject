from fastapi import APIRouter, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session
from services.face_service import process_video
from database import AsyncSessionLocal

router = APIRouter(prefix="/register-face", tags=["Face Registration"])

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@router.post("/")
async def register_face(
    user_id: int = Form(...),
    name: str = Form(...),
    age: int = Form(...),
    address: str = Form(...),
    phonenumber: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSessionLocal = Depends(get_db)
):
    print(f"user_id: {user_id}, name: {name}, age: {age}, address: {address}, phonenumber: {phonenumber}, file: {file.filename}")
    result = await process_video(file, user_id, name, age, address, phonenumber, db)
    return result


