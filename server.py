# server.py
import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from RegisterFace import register_face  # Import từ file registerFace.py
from video_stream import generate_frames  # Import từ file video_stream.py
from fastapi import BackgroundTasks
import utils.utils as utils

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r") as f:
        return f.read()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/register-face")
async def api_register_face():
    """ Gọi register_face() và trả về kết quả cho client """
    print("Registering face...")
    utils.set_register_flag(True)
    #embeddings = register_face()
    #return {"status": "success", "embeddings": {k: v.tolist() for k, v in embeddings.items()}}
    return "success"

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