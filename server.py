# server.py
from fastapi import FastAPI, WebSocket, BackgroundTasks, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from RegisterFace import register_face, register_faceByFrame  # Import từ file registerFace.py
from video_stream import generate_frames  # Import từ file video_stream.py
import RegisterFace as rgf
import video_stream as vs
import asyncio  # Import asyncio
import logging
from schemas.FaceInfo import FaceInfo
from database import get_db
from models.face_information import FaceInformation
from sqlalchemy.future import select
# Set the logging level for the websockets library to WARNING
logging.getLogger("websockets").setLevel(logging.WARNING)
app = FastAPI()


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
    print("Registering face...")
    background_tasks.add_task(register_face_async)
    return {"status": "processing"}

def register_face_async():
    while not all(rgf.face_angles.values()):
        register_faceByFrame(vs.latest_frame)
        print("Registering")  # Or save to DB

@app.post("/save-face")
async def save_face(face_info: FaceInfo, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(FaceInformation.faceInfoId).order_by(FaceInformation.faceInfoId.desc()).limit(1))
    last_face = result.scalar()

    new_id = (last_face + 1) if last_face else 1

    newFaceInfo = FaceInformation(
        faceInfoId=new_id,  
        empId=face_info.empId,
        name=face_info.name,
        firstName=face_info.firstName,
        lastName=face_info.lastName,
        groupId=face_info.groupId,
        groupName=face_info.groupName,
        dob=face_info.dob,
        gender=face_info.gender,
        phone=face_info.phone,
        email=face_info.email,
        avatar=face_info.avatar,
    )
    
    db.add(newFaceInfo)
    await db.commit()
    await db.refresh(newFaceInfo)

    return {
        "status": "success",
        "message": "Face data saved successfully",
        "data": newFaceInfo
    }

    
   
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        await websocket.send_json(rgf.face_angles)
        await asyncio.sleep(1)  # Adjust the frequency as needed
