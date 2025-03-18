from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import asyncio

DATABASE_URL = "mysql+aiomysql://root:1234@localhost:3306/ai_portal"

engine = create_async_engine(DATABASE_URL, echo=True)

AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# async def get_db():
#     async with AsyncSessionLocal() as session:
#         yield session
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session