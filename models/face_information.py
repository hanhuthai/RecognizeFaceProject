from sqlalchemy import Column, Integer, String, TIMESTAMP, JSON, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

Base = declarative_base()

class FaceInformation(Base):
    __tablename__ = "face_information"

    faceInfoId = Column(Integer, primary_key=True, index=True)
    empId = Column(String(45), nullable=False)
    name = Column(String(100), nullable=False)
    firstName = Column(String(45), nullable=True, default="")
    lastName = Column(String(45), nullable=True, default="0")  # Giá trị mặc định '0'
    groupId = Column(Integer, nullable=True)
    groupName = Column(String(100), nullable=True, default="")
    dob = Column(String(45), nullable=True, default="")
    gender = Column(String(10), nullable=True, default="")
    phone = Column(String(45), nullable=True, default="")
    email = Column(String(90), nullable=True, default="")
    avatar = Column(String(2000), nullable=True, default="")
    createdAt = Column(TIMESTAMP, server_default=func.current_timestamp())  # Lấy timestamp hiện tại
    updatedAt = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())
    embedding = Column(LargeBinary, nullable=True, default=None)
    faces = relationship("Face", back_populates="face_information")

