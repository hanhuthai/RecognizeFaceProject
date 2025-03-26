from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class FaceGroupInfoBase(BaseModel):
    groupName: str = Field(..., max_length=100)
    description: Optional[str] = Field("", max_length=250)
    numTarget: Optional[int] = 0

class FaceGroupInfoCreate(FaceGroupInfoBase):
    pass

class FaceGroupInfoUpdate(BaseModel):
    description: Optional[str] = Field(None, max_length=250)
    numTarget: Optional[int] = None

class FaceGroupInfoResponse(FaceGroupInfoBase):
    id: int
    createdAt: datetime
    updatedAt: datetime

    class Config:
        from_attributes = True  # Dùng khi chuyển từ SQLAlchemy model sang Pydantic schema
