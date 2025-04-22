from sqlalchemy import Column, Integer, String, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship
from models.face_information import Base  # Import Base from face_information module

class FaceRegistered(Base):
    __tablename__ = 'face_registered'

    id = Column(Integer, primary_key=True, autoincrement=True)
    userName = Column(String(100), nullable=True)  # Thêm cột name với độ dài 100 ký tự
    direction = Column(String(50), nullable=True)
    faceInfoId = Column(Integer, ForeignKey('face_information.faceInfoId'), nullable=True)
    embedding = Column(LargeBinary(2048), nullable=True)  # Specified size of 2048 bytes

    face_information = relationship("FaceInformation", back_populates="faces")


