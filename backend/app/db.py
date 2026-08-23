from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

try:
    from app.models import Base
except ImportError:
    from models import Base

# Load environment variables from .env file
load_dotenv()
if not os.environ.get("DATABASE_URL"):
    # Try parent directory relative to this file
    parent_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(dotenv_path=parent_env)

# Get this from your Neon dashboard - looks like:
# postgresql://user:password@ep-xxxx.region.aws.neon.tech/dbname?sslmode=require
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    # Fallback to avoid breaking imports during local build/test if environment is not set up
    DATABASE_URL = "sqlite:///./test.db"

# Fix for Heroku/Neon style postgres URLs
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def init_db():
    Base.metadata.create_all(bind=engine)

