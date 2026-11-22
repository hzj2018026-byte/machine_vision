"""个人识别历史、图鉴、统计与纠错。"""
from datetime import date, datetime, time, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..config import DEFAULT_CLASS_NAMES
from ..database import get_db
from ..deps import get_current_user
from ..models import (
    Badcase,
    OreCard,
    RecognitionRecord,
    User,
    UserCollection,
)
from ..schemas import (
    BadcaseIn,
    BadcaseOut,
    CollectionOut,
    DayCount,
    LabelCount,
    RecordOut,
    StatsOut,
)
from ..services.model_service import model_service

router = APIRouter(prefix="/api", tags=["个人数据"])


@router.get("/records", response_model=list[RecordOut])
def list_records(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return (
        db.query(RecognitionRecord)
        .filter(RecognitionRecord.user_id == user.id)
        .order_by(RecognitionRecord.created_at.desc(), RecognitionRecord.id.desc())
        .limit(limit)
        .all()
    )


@router.get("/collection", response_model=CollectionOut)
def get_collection(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    total = db.query(OreCard).count()
    unlocked = (
        db.query(UserCollection)
        .filter(UserCollection.user_id == user.id)
        .all()
    )
    keys = [row.ore_key for row in unlocked]
    percent = int(len(keys) / total * 100) if total else 0
    return CollectionOut(total=total, unlocked=keys, percent=percent)


@router.get("/stats", response_model=StatsOut)
def get_stats(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    rows = (
        db.query(RecognitionRecord)
        .filter(RecognitionRecord.user_id == user.id)
        .all()
    )

    today = date.today()
    start = today - timedelta(days=6)
    total = len(rows)
    today_count = sum(1 for row in rows if row.created_at.date() == today)
    week_total = sum(1 for row in rows if start <= row.created_at.date() <= today)
    avg_confidence = (
        round(sum(row.confidence for row in rows) / total, 3) if total else 0.0
    )

    buckets = {start + timedelta(days=i): 0 for i in range(7)}
    label_counts: dict[str, int] = {}
    for row in rows:
        day = row.created_at.date()
        if day in buckets:
            buckets[day] += 1
        label_counts[row.label] = label_counts.get(row.label, 0) + 1

    ore_total = db.query(OreCard).count()
    unlocked = (
        db.query(UserCollection)
        .filter(UserCollection.user_id == user.id)
        .count()
    )
    percent = int(unlocked / ore_total * 100) if ore_total else 0

    return StatsOut(
        total=total,
        today=today_count,
        week_total=week_total,
        avg_confidence=avg_confidence,
        days=[
            DayCount(date=day.strftime("%m-%d"), count=count)
            for day, count in buckets.items()
        ],
        labels=[
            LabelCount(label=label, count=count)
            for label, count in sorted(
                label_counts.items(), key=lambda item: item[1], reverse=True
            )
        ],
        ore_total=ore_total,
        unlocked=unlocked,
        percent=percent,
    )


@router.post("/badcase", response_model=BadcaseOut, status_code=201)
def create_badcase(
    payload: BadcaseIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    record = db.get(RecognitionRecord, payload.record_id)
    # 只能纠正自己的记录，防止越权
    if record is None or record.user_id != user.id:
        raise HTTPException(status_code=404, detail="识别记录不存在")

    correct = payload.correct_label.strip()
    supported = model_service.class_names or DEFAULT_CLASS_NAMES
    if correct not in supported:
        raise HTTPException(status_code=400, detail="请选择模型支持的类别")
    if correct == record.label:
        raise HTTPException(status_code=400, detail="纠正的类别和识别结果相同")

    row = db.query(Badcase).filter(Badcase.record_id == record.id).first()
    if row is None:
        row = Badcase(
            user_id=user.id,
            record_id=record.id,
            image_path=record.image_path,
            pred_label=record.label,
            correct_label=correct,
        )
        db.add(row)
    else:
        # 同一条记录重复提交，覆盖上次纠正
        row.pred_label = record.label
        row.correct_label = correct
    db.commit()
    db.refresh(row)
    return row
