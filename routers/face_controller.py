from fastapi import APIRouter, Form, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from services.face_service import capture_face
from database import AsyncSessionLocal
import uuid
router = APIRouter(prefix="/register-face", tags=["Face Registration"])


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


@router.post("/")
async def register_face(
        name: str = Form(...),
        age: int = Form(...),
        address: str = Form(...),
        phonenumber: str = Form(...),
        db: AsyncSession = Depends(get_db)
):
    user_id = str(uuid.uuid4())  # ✅ Tạo UUID mới cho user_id

    print(f"user_id: {user_id}, name: {name}, age: {age}, address: {address}, phonenumber: {phonenumber}")

    result = await capture_face( name, age, address, phonenumber, db)
    return result

