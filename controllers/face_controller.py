from fastapi import APIRouter, Depends, BackgroundTasks, WebSocket, Response, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.websockets import WebSocketDisconnect
import asyncio

from database import get_db
from schemas.FaceInfo import FaceInfo
from services.face_service import (
    register_face_async,
    reset_face_angles,
    save_face_info,
    get_face_embedding,
    get_face_embedding_by_direction
)
from video_stream import generate_frames
import RegisterFace as rgf

router = APIRouter(tags=["face"])

stop_event = asyncio.Event()

@router.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.post("/register-face")
async def register_face(background_tasks: BackgroundTasks):
    stop_event.clear()
    print("Registering face...")
    background_tasks.add_task(register_face_async, stop_event)
    return {"status": "processing", "message": "Face registration started"}

@router.post("/stop-register")
async def stop_register_face():
    stop_event.set()
    await reset_face_angles()
    print("Stopped face registration and reset face angles.")
    return {"status": "stopped"}

@router.post("/save-face")
async def save_face(face_info: FaceInfo, db: AsyncSession = Depends(get_db)):
    return await save_face_info(face_info, db)

@router.get("/embedding/{face_id}")
async def get_embedding(face_id: int, db: AsyncSession = Depends(get_db)):
    return await get_face_embedding(face_id, db)

@router.get("/embedding/{face_id}/{direction}")
async def get_embedding_by_direction(face_id: int, direction: str, db: AsyncSession = Depends(get_db)):
    return await get_face_embedding_by_direction(face_id, direction, db)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(rgf.face_angles)
            await asyncio.sleep(1)  
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Unexpected error: {e}")

@router.get("/check-connection")
async def check_connection():
    return {"status": "connected"}