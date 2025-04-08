import json

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, BackgroundTasks, Depends, Response, status
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from RegisterFace import register_faceByFrame  # Import từ file registerFace.py
from video_stream import generate_frames  # Import từ file video_stream.py
import RegisterFace as rgf
import video_stream as vs
import asyncio  # Import asyncio
import logging
from schemas.FaceInfo import FaceInfo
from database import get_db
from models.face_information import FaceInformation
from sqlalchemy.future import select
from sqlalchemy import text
from models.face_groupinfo import FaceGroupInfo
import os
import cv2
from fastapi.middleware.cors import CORSMiddleware
from RegisterFace import face_angles, face_embeddings, captured_images
import concurrent.futures
import RegisterFace
from starlette.websockets import WebSocketDisconnect
import uvicorn
import unicodedata  # Add this import for normalization
import re  # Add this import for regex
load_dotenv()
# Set the logging level for the websockets library to WARNING
logging.getLogger("websockets").setLevel(logging.WARNING)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả nguồn gốc (thay "*" bằng domain cụ thể nếu cần)
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả phương thức (GET, POST, OPTIONS, ...)
    allow_headers=["*"],  # Cho phép tất cả headers
)
stop_event = asyncio.Event()
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("ui/RegisterFace.html", "r") as f:
        return f.read()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/register-face")
async def api_register_face(background_tasks: BackgroundTasks):
    """ Gọi register_face() and return to client """
    stop_event.clear()
    print("Registering face...")
    background_tasks.add_task(register_face_async)
    return {"status": "processing"}

def register_face_async():
    """Hàm xử lý đăng ký khuôn mặt"""
    while not stop_event.is_set() and not all(face_angles.values()):
        register_faceByFrame(vs.latest_frame)
        print("Registering...")

async def reset_face_angles():
    """Đặt tất cả các phần tử của face_angles về False"""
    RegisterFace.face_angles = {key: False for key in face_angles}
    RegisterFace.face_embeddings = {key: None for key in face_embeddings}
    RegisterFace.captured_images = {key: None for key in captured_images}
    print("Reset face angles and captured images.", face_angles)
    
@app.post("/stop-register-face")
async def stop_register_face():
    """API để dừng quá trình đăng ký và reset face_angles"""
    stop_event.set()
    await reset_face_angles()
      # Dừng quá trình đăng ký
    print("Stopped face registration and reset face angles.")
    return {"status": "stopped"}
def run_in_executor():
    """Chạy đăng ký trong Thread Pool Executor"""
    future = executor.submit(register_face_async)
    return future.result()
@app.post("/save-face")
async def save_face(face_info: FaceInfo, db: AsyncSession = Depends(get_db)):
    # Validate email format
    if not face_info.email.endswith("@gmail.com"):
        return Response(
            content=json.dumps({"status": "error", "message": "Email must be in the format '@gmail.com'."}),
            status_code=status.HTTP_400_BAD_REQUEST,
            media_type="application/json"
        )

    # Validate name is a string
    if not face_info.firstName.isalpha() or not face_info.lastName.isalpha():
        return Response(
            json.dumps({"status": "error", "message": "Name must contain only alphabetic characters."}),
            status_code=status.HTTP_400_BAD_REQUEST,
            media_type="application/json"
        )

    # Validate empId is an integer
    if not isinstance(face_info.empId, str):
        return Response(
            json.dumps({"status": "error", "message": "Employee ID must be an integer."}),
            status_code=status.HTTP_400_BAD_REQUEST,
            media_type="application/json"
        )

    # Validate phone is an integer and has 10 digits
    if not isinstance(face_info.phone, str) or len(face_info.phone) != 10:
        return Response(
            json.dumps({"status": "error", "message": "Phone number must be a 10-digit integer."}),
            status_code=status.HTTP_400_BAD_REQUEST,
            media_type="application/json"
        )

    result = await db.execute(select(FaceInformation.faceInfoId).order_by(FaceInformation.faceInfoId.desc()).limit(1))
    last_face = result.scalar()
    new_id = (last_face + 1) if last_face else 1

    group_result = await db.execute(select(FaceGroupInfo.id).where(FaceGroupInfo.groupName == face_info.groupName))
    group_id = group_result.scalar() or 0  

    insert_sql = text(
        """
        CALL sp_insert_target(:id_, :empId, :firstName, :lastName, :groupId, :dob, 
                              :gender, :phone, :email, :avatar, @output);
        """
    )
    
    await db.execute(
        insert_sql,
        {
            "id_": new_id,
            "empId": face_info.empId,
            "firstName": face_info.firstName,
            "lastName": face_info.lastName,
            "groupId": group_id,  
            "dob": face_info.dob,
            "gender": face_info.gender,
            "phone": face_info.phone,
            "email": face_info.email,
            "avatar": face_info.avatar
        }
    )

    output_sql = text("SELECT @output AS refValue;")
    output_result = await db.execute(output_sql)
    ref_value = output_result.scalar()

    await db.commit()
    face_database_path = os.getenv('FACE_DATABASE_PATH')
    print(face_database_path)
    if ref_value == 1:
        if len(captured_images) < 4:
            return Response(
                content=json.dumps({"status": "error", "message": "Not enough face angles captured. Please complete all 5 angles before saving."}),
                status_code=status.HTTP_400_BAD_REQUEST,
                media_type="application/json"
            )
        else:
            normalized_last_name = normalize_filename(face_info.lastName)
            normalized_first_name = normalize_filename(face_info.firstName)
            normalized_group_name = normalize_filename(face_info.groupName)
            normalized_group_name_ClusterA = "ClusterA"
            normalized_dob = re.sub(r'\s+', '', face_info.dob)
            concatenated_name_dob = f"{normalized_last_name}-{normalized_dob}"
            save_dir = f"{face_database_path}/{normalized_group_name_ClusterA}/{new_id}"
            os.makedirs(save_dir, exist_ok=True)
            save_dir_1 = f"{face_database_path}/{normalized_group_name}/{concatenated_name_dob}"
            os.makedirs(save_dir_1, exist_ok=True)

        for angle, img in captured_images.items():
            img_path = os.path.normpath(os.path.join(save_dir, f"{angle}.jpg"))
            img_path1 = os.path.normpath(os.path.join(save_dir_1, f"{angle}.jpg"))

            try:
                if cv2.imwrite(img_path, img) and cv2.imwrite(img_path1, img):
                    print(f"✅ Saved image: {img_path}")
                else:
                    print(f"❌ Failed to save image: {img_path}")
            except cv2.error as e:
                print(f"❌ Error saving image {img_path}: {e}")

        # Reset face angles and captured images after successful save
        await reset_face_angles()

        return Response(
            content=json.dumps({
                "status": "success",
                "message": "Face data and images saved successfully",
                "data": {
                    "faceInfoId": new_id,
                    "empId": face_info.empId,
                    "firstName": face_info.firstName,
                    "lastName": face_info.lastName,
                    "groupId": group_id,
                    "groupName": face_info.groupName,
                    "dob": face_info.dob,
                    "gender": face_info.gender,
                    "phone": face_info.phone,
                    "email": face_info.email,
                    "avatar": face_info.avatar
                }
            }),
            status_code=status.HTTP_201_CREATED,
            media_type="application/json"
        )
    else:
        return Response(
            json.dumps({"status": "error", "message": "Failed to save face data"}),
            status_code=status.HTTP_400_BAD_REQUEST,
            media_type="application/json"
        )

def normalize_filename(name):
    """Normalize a string to remove diacritics, spaces, and invalid characters."""
    name = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode('utf-8')  # Remove diacritics
    name = re.sub(r'\s+', '', name)  # Remove spaces
    name = re.sub(r'[^\w.-]', '', name)  # Remove invalid characters
    return name

@app.websocket("/ws")
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)