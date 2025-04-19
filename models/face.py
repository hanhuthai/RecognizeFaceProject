from sqlalchemy import Column, Integer, String, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship
from models.face_information import Base  # Import Base from face_information module

class Face(Base):
    __tablename__ = 'face'

    id = Column(Integer, primary_key=True, autoincrement=True)
    direction = Column(String(50), nullable=True)
    faceInfoId = Column(Integer, ForeignKey('face_information.faceInfoId'), nullable=True)
    embedding = Column(LargeBinary(2048), nullable=True)  # Specified size of 2048 bytes

    face_information = relationship("FaceInformation", back_populates="faces")


