"""环境变量与路径配置。"""
import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

# backend/.env
BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")

# config.py -> app -> backend -> ore_recognition_web -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "ore_kids")

DATABASE_URL = os.getenv("DATABASE_URL") or (
    f"mysql+pymysql://{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
)

SECRET_KEY = os.getenv(
    "SECRET_KEY", "ore-park-dev-secret-key-change-me-in-production"
)
TOKEN_EXPIRE_DAYS = int(os.getenv("TOKEN_EXPIRE_DAYS", "7"))

MODEL_PATH = Path(os.getenv(
    "MODEL_PATH",
    str(PROJECT_ROOT / "models" / "ore_classification_model.pth"),
))

UPLOAD_DIR = BACKEND_DIR / "media" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
USE_TTA = os.getenv("USE_TTA", "true").lower() in {"1", "true", "yes", "on"}

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

DEFAULT_CLASS_NAMES = ["方解石", "石英", "赤铁矿", "长石", "黄铁矿"]
