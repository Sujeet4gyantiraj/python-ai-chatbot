import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# IMPORTANT: import models so Base knows all tables
from models import Base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://adarsh1927:Adarsh%40123@localhost:5432/rhysleybot"
)

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ⚠️ Dev only — comment in production
# Base.metadata.create_all(bind=engine)

