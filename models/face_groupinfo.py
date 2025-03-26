from sqlalchemy import Column, Integer, String, TIMESTAMP, text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class FaceGroupInfo(Base):
    __tablename__ = "face_group_info"

    id = Column(Integer, primary_key=True, autoincrement=True)
    groupName = Column(String(100), nullable=False)
    description = Column(String(250), default="")
    numTarget = Column(Integer, default=0)
    createdAt = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    updatedAt = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))
