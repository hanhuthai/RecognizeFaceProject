# server.py
from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from RegisterFace import register_face, register_faceByFrame  # Import từ file registerFace.py
from video_stream import generate_frames  # Import từ file video_stream.py
import RegisterFace as rgf
import video_stream as vs
import asyncio  # Import asyncio
import logging
from pydantic import BaseModel
from SearchFace import search_similar_faces  # Import the search function

# Set the logging level for the websockets library to WARNING
logging.getLogger("websockets").setLevel(logging.WARNING)
app = FastAPI()

class FaceData(BaseModel):
    name: str
    gender: str
    phone: str
    email: str

class SearchRequest(BaseModel):
    image_path: str
    k: int

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
async def save_face(data: FaceData):
    # Save the face data and embeddings
    print(f"Name: {data.name}")
    print(f"Gender: {data.gender}")
    print(f"Phone: {data.phone}")
    print(f"Email: {data.email}")
    #print(f"Embeddings: {rgf.face_embeddings}")
    # Add your save logic here (e.g., save to database)
    return {"status": "success", "message": "Face data saved successfully"}

@app.post("/search-face")
async def api_search_face(request: SearchRequest):
    """ Search for similar faces given an image path and number of top similar faces to return """
    similar_faces = search_similar_faces(request.image_path, request.k)
    return {"similar_faces": similar_faces}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        await websocket.send_json(rgf.face_angles)
        await asyncio.sleep(1)  # Adjust the frequency as needed

