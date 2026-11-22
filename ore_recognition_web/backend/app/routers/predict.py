"""图片识别主流程。"""
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image
from sqlalchemy.orm import Session

from ..config import ALLOWED_TYPES, MAX_UPLOAD_MB, UPLOAD_DIR
from ..database import get_db
from ..deps import get_optional_user
from ..models import OreCard, RecognitionRecord, User, UserCollection
from ..schemas import OreCardOut, PredictOut
from ..services.model_service import model_service

router = APIRouter(prefix="/api", tags=["识别"])


@router.post("/predict", response_model=PredictOut)
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),   # 游客也可以识别
):
    if not model_service.ready:
        raise HTTPException(status_code=503, detail=model_service.error or "模型未加载")

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="只支持 JPG / PNG / WebP / BMP 图片")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="图片内容为空")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"图片不能超过 {MAX_UPLOAD_MB}MB")

    # UUID 重命名，避免中文文件名和重名覆盖
    suffix = {
        "image/png": ".png",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
    }.get(file.content_type, ".jpg")
    filename = f"{uuid.uuid4().hex}{suffix}"
    filepath = UPLOAD_DIR / filename
    filepath.write_bytes(raw)

    try:
        image = Image.open(filepath)
        image.load()
    except Exception:
        filepath.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="图片无法解析，请换一张试试")

    try:
        result = model_service.predict(image)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    label = result["label"]
    confidence = result["confidence"]
    uncertain = result["uncertain"]

    if uncertain:
        message = "看不太清楚，再靠近一点、光线亮一些，一次只放一块矿石试试～"
    else:
        message = f"这是{label}！"

    card = db.query(OreCard).filter(OreCard.key == label).first()

    record = RecognitionRecord(
        user_id=user.id if user else None,   # 游客记录 user_id 为空
        image_path=f"/media/uploads/{filename}",
        label=label,
        confidence=confidence,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    # 登录用户识别成功后自动解锁图鉴
    if user is not None:
        exists = (
            db.query(UserCollection)
            .filter(
                UserCollection.user_id == user.id,
                UserCollection.ore_key == label,
            )
            .first()
        )
        if exists is None:
            db.add(UserCollection(user_id=user.id, ore_key=label))
            db.commit()

    return PredictOut(
        label=label,
        confidence=confidence,
        uncertain=uncertain,
        message=message,
        top3=result["top3"],
        card=OreCardOut.model_validate(card) if card else None,
        record_id=record.id,
    )
