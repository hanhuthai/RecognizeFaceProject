import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# Lấy biến môi trường
face_database_path = os.getenv('FACE_DATABASE_PATH')
print(face_database_path)
