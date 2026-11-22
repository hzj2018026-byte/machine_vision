"""管理端接口：概览、用户、识别记录、纠错、知识卡。"""
from datetime import date, datetime, time, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_admin_user
from ..models import Badcase, OreCard, RecognitionRecord, User
from ..schemas import (
    AdminBadcaseOut,
    AdminBadcasePageOut,
    AdminOverviewOut,
    AdminRecordOut,
    AdminRecordPageOut,
    AdminUserOut,
    AdminUserPageOut,
    BadcaseHandleIn,
    DayCount,
    LabelCount,
    OreCardOut,
    OreCardUpdate,
)

router = APIRouter(prefix="/api/admin", tags=["管理端"])


def _paging(page: int, page_size: int) -> tuple[int, int]:
    # 防止前端传入负数或一次拉取过多数据
    return max(1, page), min(max(1, page_size), 50)


@router.get("/overview", response_model=AdminOverviewOut)
def overview(db: Session = Depends(get_db), _: User = Depends(get_admin_user)):
    today = date.today()
    start = today - timedelta(days=6)

    user_total = db.query(func.count(User.id)).scalar() or 0
    record_total = db.query(func.count(RecognitionRecord.id)).scalar() or 0
    today_records = (
        db.query(func.count(RecognitionRecord.id))
        .filter(RecognitionRecord.created_at >= datetime.combine(today, time.min))
        .scalar()
        or 0
    )
    badcase_total = db.query(func.count(Badcase.id)).scalar() or 0
    pending_badcases = (
        db.query(func.count(Badcase.id))
        .filter(Badcase.handled.is_(False))
        .scalar()
        or 0
    )
    avg_confidence = db.query(func.avg(RecognitionRecord.confidence)).scalar() or 0

    day_rows = (
        db.query(func.date(RecognitionRecord.created_at), func.count(RecognitionRecord.id))
        .filter(RecognitionRecord.created_at >= datetime.combine(start, time.min))
        .group_by(func.date(RecognitionRecord.created_at))
        .all()
    )
    day_counts = {str(day): count for day, count in day_rows}
    days = [
        DayCount(
            date=(start + timedelta(days=i)).strftime("%m-%d"),
            count=day_counts.get(str(start + timedelta(days=i)), 0),
        )
        for i in range(7)
    ]

    label_rows = (
        db.query(RecognitionRecord.label, func.count(RecognitionRecord.id))
        .group_by(RecognitionRecord.label)
        .order_by(func.count(RecognitionRecord.id).desc())
        .all()
    )

    return AdminOverviewOut(
        user_total=user_total,
        record_total=record_total,
        today_records=today_records,
        badcase_total=badcase_total,
        pending_badcases=pending_badcases,
        avg_confidence=round(float(avg_confidence), 3),
        ore_total=db.query(func.count(OreCard.id)).scalar() or 0,
        days=days,
        labels=[LabelCount(label=label, count=count) for label, count in label_rows],
    )


@router.get("/users", response_model=AdminUserPageOut)
def list_users(
    query: str = "",
    page: int = 1,
    page_size: int = 10,
    db: Session = Depends(get_db),
    _: User = Depends(get_admin_user),
):
    page, page_size = _paging(page, page_size)
    base = (
        db.query(User, func.count(RecognitionRecord.id))
        .outerjoin(RecognitionRecord, RecognitionRecord.user_id == User.id)
        .group_by(User.id)
    )

    keyword = query.strip()
    if keyword:
        like = f"%{keyword}%"
        base = base.filter(or_(User.username.like(like), User.nickname.like(like)))

    total = base.count()
    rows = (
        base.order_by(User.created_at.desc(), User.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return AdminUserPageOut(
        total=total,
        items=[
            AdminUserOut(
                id=user.id,
                username=user.username,
                nickname=user.nickname,
                is_admin=user.is_admin,
                record_count=count,
                created_at=user.created_at,
            )
            for user, count in rows
        ],
    )


@router.get("/records", response_model=AdminRecordPageOut)
def list_records(
    label: str = "",
    page: int = 1,
    page_size: int = 10,
    db: Session = Depends(get_db),
    _: User = Depends(get_admin_user),
):
    page, page_size = _paging(page, page_size)
    base = db.query(RecognitionRecord, User).outerjoin(
        User, RecognitionRecord.user_id == User.id
    )
    if label.strip():
        base = base.filter(RecognitionRecord.label == label.strip())

    total = base.count()
    rows = (
        base.order_by(
            RecognitionRecord.created_at.desc(), RecognitionRecord.id.desc()
        )
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return AdminRecordPageOut(
        total=total,
        items=[
            AdminRecordOut(
                id=record.id,
                username=user.username if user else None,
                image_path=record.image_path,
                label=record.label,
                confidence=record.confidence,
                created_at=record.created_at,
            )
            for record, user in rows
        ],
    )


@router.get("/badcases", response_model=AdminBadcasePageOut)
def list_badcases(
    handled: bool | None = None,
    page: int = 1,
    page_size: int = 10,
    db: Session = Depends(get_db),
    _: User = Depends(get_admin_user),
):
    page, page_size = _paging(page, page_size)
    base = db.query(Badcase, User).outerjoin(User, Badcase.user_id == User.id)
    if handled is not None:
        base = base.filter(Badcase.handled.is_(handled))

    total = base.count()
    rows = (
        base.order_by(Badcase.created_at.desc(), Badcase.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return AdminBadcasePageOut(
        total=total,
        items=[
            AdminBadcaseOut(
                id=row.id,
                record_id=row.record_id,
                username=user.username if user else None,
                image_path=row.image_path,
                pred_label=row.pred_label,
                correct_label=row.correct_label,
                handled=row.handled,
                created_at=row.created_at,
            )
            for row, user in rows
        ],
    )


@router.patch("/badcases/{badcase_id}", response_model=AdminBadcaseOut)
def update_badcase(
    badcase_id: int,
    payload: BadcaseHandleIn,
    db: Session = Depends(get_db),
    _: User = Depends(get_admin_user),
):
    row = db.get(Badcase, badcase_id)
    if row is None:
        raise HTTPException(status_code=404, detail="纠错记录不存在")
    row.handled = payload.handled
    db.commit()
    db.refresh(row)
    user = db.get(User, row.user_id)
    return AdminBadcaseOut(
        id=row.id,
        record_id=row.record_id,
        username=user.username if user else None,
        image_path=row.image_path,
        pred_label=row.pred_label,
        correct_label=row.correct_label,
        handled=row.handled,
        created_at=row.created_at,
    )


@router.put("/ores/{key}", response_model=OreCardOut)
def update_ore(
    key: str,
    payload: OreCardUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(get_admin_user),
):
    card = db.query(OreCard).filter(OreCard.key == key).first()
    if card is None:
        raise HTTPException(status_code=404, detail="知识卡不存在")

    # exclude_none：只更新前端真正提交的字段
    for field, value in payload.model_dump(exclude_none=True).items():
        if isinstance(value, str):
            value = value.strip()
            if not value and field in {"name", "emoji"}:
                raise HTTPException(status_code=400, detail="名称和图标不能为空")
        setattr(card, field, value)

    db.commit()
    db.refresh(card)
    return card
