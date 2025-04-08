from pydantic import BaseModel
from typing import Optional

class FaceInfo(BaseModel):
    faceInfoId: Optional[int] = None
    empId: str
    name: Optional[str] = ""
    firstName: Optional[str] = ""
    lastName: Optional[str] = "0"
    groupId: Optional[int] = None
    groupName: Optional[str] = ""
    dob: Optional[str] = ""
    gender: Optional[str] = ""
    phone: Optional[str] = ""
    email: Optional[str] = ""
    avatar: Optional[str] = ""
    embedding: Optional[bytes] = None  # Dùng bytes thay vì dict

    class Config:
        orm_mode = True
