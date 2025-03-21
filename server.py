# server.py
from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from RegisterFace import register_face, register_faceByFrame  # Import từ file registerFace.py
from video_stream import generate_frames  # Import từ file video_stream.py
import utils.utils as utils
import RegisterFace as rgf
import video_stream as vs
import asyncio  # Import asyncio

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r") as f:
        return f.read()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/register-face")
async def api_register_face(background_tasks: BackgroundTasks):
    """ Gọi register_face() and return to client """
    print("Registering face...")
    utils.set_register_flag(True)
    background_tasks.add_task(register_face_async)
    return {"status": "processing"}

def register_face_async():
    while not all(rgf.face_angles.values()):
        register_faceByFrame(vs.latest_frame)
        print("Registering")  # Or save to DB

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        await websocket.send_json(rgf.face_angles)
        await asyncio.sleep(1)  # Adjust the frequency as needed

# @app.post("/register-face")
# async def api_register_face(background_tasks: BackgroundTasks):
#     """ Gọi register_face() và trả về kết quả cho client """
#     #embeddings = register_face()
#     background_tasks.add_task(register_face_async)
#     return {"status": "processing"}
# # dùng background_tasks để chạy hàm register_face_async() mà không chờ kết quả,  trả về ngay lập tức "status": "processing"
# def register_face_async():
#     embeddings = register_face()
#     print("Embeddings:", embeddings)  # Hoặc lưu vào DB
# #    return {"status": "success", "embeddings": {k: v.tolist() for k, v in embeddings.items()}}