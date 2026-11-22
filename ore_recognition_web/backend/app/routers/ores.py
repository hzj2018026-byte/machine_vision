"""矿石知识卡读取。"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import OreCard
from ..schemas import OreCardOut

router = APIRouter(prefix="/api/ores", tags=["知识卡"])


@router.get("", response_model=list[OreCardOut])
def list_ores(db: Session = Depends(get_db)):
    return db.query(OreCard).order_by(OreCard.id.asc()).all()


@router.get("/{key}", response_model=OreCardOut)
def get_ore(key: str, db: Session = Depends(get_db)):
    card = db.query(OreCard).filter(OreCard.key == key).first()
    if card is None:
        raise HTTPException(status_code=404, detail="知识卡不存在")
    return card
