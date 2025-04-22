1. Web using: FastAPI + WebSocket
 - run with debugmode : uvicorn server:app --host 127.0.0.1 --port 8000 --reload --log-level debug
 - run with production mode : uvicorn server:app --host 0.0.0.0 --port 8000 --reload
2. kill port 8000
netstat -ano | findstr :8000

3. Database Initialization
 - Create database tables before starting the application:
```python
from database import engine, Base
from models.face_information import FaceInformation
from models.face_groupinfo import FaceGroupInfo
from models.face import Face

# Import all models here to ensure they're registered with SQLAlchemy

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Run this function once to initialize the database
```
