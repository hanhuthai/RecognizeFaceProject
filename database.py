from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

# Thay đổi thông tin kết nối MySQL
DATABASE_URL = "mysql+asyncmy://user:password@localhost:3307/ai_portal"

# Tạo engine kết nối
engine = create_async_engine(DATABASE_URL, echo=True)

# Tạo session factory
async_session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Base Model cho ORM
Base = declarative_base()

# Dependency để lấy session
async def get_db():
    async with async_session_factory() as session:
        yield session

async def check_database_connection():
    try:
        async with async_session_factory() as session:
            result = await session.execute(text("SELECT 1"))
            print("✅ Kết nối MySQL thành công:", result.scalar())
    except Exception as e:
        print("❌ Lỗi kết nối MySQL:", str(e))