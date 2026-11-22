"""ORM 模型：用户、矿石知识卡、识别记录、纠错、图鉴解锁。"""
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    nickname: Mapped[str] = mapped_column(String(50), default="小小矿物学家")
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    records: Mapped[list["RecognitionRecord"]] = relationship(back_populates="user")
    collection: Mapped[list["UserCollection"]] = relationship(back_populates="user")


class OreCard(Base):
    __tablename__ = "ore_cards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 数据库列名用 ore_key，避免 MySQL 保留字 key；Python/接口里仍叫 key
    key: Mapped[str] = mapped_column("ore_key", String(20), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(20), nullable=False)
    pinyin: Mapped[str] = mapped_column(String(50), default="")
    emoji: Mapped[str] = mapped_column(String(10), default="💎")
    color: Mapped[str] = mapped_column(String(100), default="#8d6e63")
    hardness: Mapped[str] = mapped_column(String(50), default="")
    luster: Mapped[str] = mapped_column(String(100), default="")
    streak: Mapped[str] = mapped_column(String(100), default="")
    fun_fact: Mapped[str] = mapped_column(Text, default="")


class RecognitionRecord(Base):
    __tablename__ = "recognition_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    image_path: Mapped[str] = mapped_column(String(255), default="")
    label: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)

    user: Mapped[User | None] = relationship(back_populates="records")


class Badcase(Base):
    __tablename__ = "badcases"
    __table_args__ = (UniqueConstraint("record_id", name="uq_badcase_record"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    record_id: Mapped[int] = mapped_column(
        ForeignKey("recognition_records.id"), nullable=False, index=True
    )
    image_path: Mapped[str] = mapped_column(String(255), default="")
    pred_label: Mapped[str] = mapped_column(String(20), nullable=False)
    correct_label: Mapped[str] = mapped_column(String(20), nullable=False)
    handled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class UserCollection(Base):
    __tablename__ = "user_collection"
    __table_args__ = (UniqueConstraint("user_id", "ore_key", name="uq_user_ore"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    ore_key: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    user: Mapped[User] = relationship(back_populates="collection")
