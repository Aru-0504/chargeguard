from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

try:
    from app.models import Base
except ImportError:
    try:
        from models import Base
    except ImportError:
        from backend.app.models import Base

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
    DATABASE_URL = "sqlite:///./chargeguard.db"

# Fix for Heroku/Neon style postgres URLs
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

try:
    if "postgresql" in DATABASE_URL:
        # 3 second connect timeout so we don't hang if remote DB is unreachable
        engine = create_engine(DATABASE_URL, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            pass
    else:
        engine = create_engine(DATABASE_URL)
except Exception as e:
    print(f"[ChargeGuard DB] Primary database connection failed ({e}). Falling back to local sqlite:///./chargeguard.db")
    DATABASE_URL = "sqlite:///./chargeguard.db"
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def init_db():
    Base.metadata.create_all(bind=engine)

