"""启动种子数据：知识卡 + 默认管理员（幂等）。"""
from sqlalchemy.orm import Session

from .models import OreCard, User
from .security import hash_password
from .seed_data import ORE_CARDS

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "123456"


def seed_ore_cards(db: Session) -> None:
    existing = {card.key for card in db.query(OreCard).all()}
    for item in ORE_CARDS:
        if item["key"] in existing:
            continue
        db.add(OreCard(**item))
    db.commit()


def seed_admin(db: Session) -> None:
    if db.query(User).filter(User.username == ADMIN_USERNAME).first():
        return
    db.add(
        User(
            username=ADMIN_USERNAME,
            password_hash=hash_password(ADMIN_PASSWORD),
            nickname="管理员",
            is_admin=True,
        )
    )
    db.commit()
