from pydantic import BaseModel
from typing import Optional

class FaceInfo(BaseModel):
    faceInfoId: Optional[int] = None
    empId: str
    name: str
    firstName: Optional[str] = ""
    lastName: Optional[str] = "0"  # Giá trị mặc định
    groupId: Optional[int] = None
    groupName: Optional[str] = ""
    dob: Optional[str] = ""
    gender: Optional[str] = ""
    phone: Optional[str] = ""
    email: Optional[str] = ""
    avatar: Optional[str] = ""
    embedding: Optional[dict] = None

    class Config:
        orm_mode = True  # Hỗ trợ convert từ SQLAlchemy Model
