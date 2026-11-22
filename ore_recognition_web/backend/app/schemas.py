"""Pydantic 请求/响应模型。"""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ---------- 认证 ----------
class RegisterIn(BaseModel):
    username: str = Field(min_length=2, max_length=50)
    password: str = Field(min_length=6, max_length=100)
    nickname: str = Field(default="小小矿物学家", max_length=50)


class LoginIn(BaseModel):
    username: str
    password: str


class AuthOut(BaseModel):
    token: str
    username: str
    nickname: str
    is_admin: bool


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    nickname: str
    is_admin: bool


# ---------- 知识卡 ----------
class OreCardOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    key: str
    name: str
    pinyin: str
    emoji: str
    color: str
    hardness: str
    luster: str
    streak: str
    fun_fact: str


class OreCardUpdate(BaseModel):
    name: str | None = None
    pinyin: str | None = None
    emoji: str | None = None
    color: str | None = None
    hardness: str | None = None
    luster: str | None = None
    streak: str | None = None
    fun_fact: str | None = None


# ---------- 识别 ----------
class TopItem(BaseModel):
    label: str
    confidence: float


class PredictOut(BaseModel):
    label: str
    confidence: float
    uncertain: bool
    message: str
    top3: list[TopItem]
    card: OreCardOut | None = None
    record_id: int


# ---------- 个人数据 ----------
class RecordOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    image_path: str
    label: str
    confidence: float
    created_at: datetime


class CollectionOut(BaseModel):
    total: int
    unlocked: list[str]
    percent: int


class DayCount(BaseModel):
    date: str
    count: int


class LabelCount(BaseModel):
    label: str
    count: int


class StatsOut(BaseModel):
    total: int
    today: int
    week_total: int
    avg_confidence: float
    days: list[DayCount]
    labels: list[LabelCount]
    ore_total: int
    unlocked: int
    percent: int


class BadcaseIn(BaseModel):
    record_id: int
    correct_label: str


class BadcaseOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    record_id: int
    image_path: str
    pred_label: str
    correct_label: str
    handled: bool
    created_at: datetime


# ---------- 管理端 ----------
class AdminOverviewOut(BaseModel):
    user_total: int
    record_total: int
    today_records: int
    badcase_total: int
    pending_badcases: int
    avg_confidence: float
    ore_total: int
    days: list[DayCount]
    labels: list[LabelCount]


class AdminUserOut(BaseModel):
    id: int
    username: str
    nickname: str
    is_admin: bool
    record_count: int
    created_at: datetime


class AdminUserPageOut(BaseModel):
    total: int
    items: list[AdminUserOut]


class AdminRecordOut(BaseModel):
    id: int
    username: str | None
    image_path: str
    label: str
    confidence: float
    created_at: datetime


class AdminRecordPageOut(BaseModel):
    total: int
    items: list[AdminRecordOut]


class AdminBadcaseOut(BaseModel):
    id: int
    record_id: int
    username: str | None
    image_path: str
    pred_label: str
    correct_label: str
    handled: bool
    created_at: datetime


class AdminBadcasePageOut(BaseModel):
    total: int
    items: list[AdminBadcaseOut]


class BadcaseHandleIn(BaseModel):
    handled: bool
