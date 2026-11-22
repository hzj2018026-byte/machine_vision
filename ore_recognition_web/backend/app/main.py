"""识矿乐园 FastAPI 入口。"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import CORS_ORIGINS, DEFAULT_CLASS_NAMES, UPLOAD_DIR
from .database import Base, SessionLocal, engine
from .routers import admin, auth, ores, predict, records
from .seed import seed_admin, seed_ore_cards
from .services.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        seed_ore_cards(db)
        seed_admin(db)
    yield


app = FastAPI(title="识矿乐园 API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /media/uploads/xxx.jpg -> backend/media/uploads/xxx.jpg
app.mount("/media", StaticFiles(directory=UPLOAD_DIR.parent), name="media")

app.include_router(auth.router)
app.include_router(ores.router)
app.include_router(predict.router)
app.include_router(records.router)
app.include_router(admin.router)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_ready": model_service.ready,
        "class_names": model_service.class_names or DEFAULT_CLASS_NAMES,
        "model_error": model_service.error,
        "img_size": model_service.img_size,
        "tta": model_service.use_tta,
    }
