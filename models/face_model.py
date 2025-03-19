import uuid
from sqlalchemy import Column, Integer, String, Enum, Text, LargeBinary
from database import Base

class FaceIndex(Base):
    __tablename__ = "face_index"

    id = Column(String(255), primary_key=True,unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(255), nullable=False, default=lambda: str(uuid.uuid4()))  # Cho phép trùng lặp
    name = Column(String(255), nullable=False)
    age = Column(Integer, nullable=False)
    address = Column(Text, nullable=False)
    phonenumber = Column(String(20), nullable=False)
    angle = Column(Enum("front", "left", "right", "up", "down"), nullable=False)
    embedding = Column(LargeBinary(2048), nullable=False)  # Lưu trực tiếp embedding
