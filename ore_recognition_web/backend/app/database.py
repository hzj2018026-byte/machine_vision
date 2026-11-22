"""SQLAlchemy 引擎、会话与 Base。

如果连接的是 MySQL，启动时会自动执行
`CREATE DATABASE IF NOT EXISTS ...`，避免手动建库。
"""
import re

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import DATABASE_URL


def ensure_database() -> None:
    """MySQL 模式下自动创建数据库；SQLite 模式直接跳过。"""
    url = make_url(DATABASE_URL)
    if url.get_backend_name() != "mysql":
        return

    import pymysql  # 只有 MySQL 模式才需要

    db_name = url.database
    if not db_name:
        return
    if not re.fullmatch(r"[A-Za-z0-9_]+", db_name):
        raise ValueError(f"数据库名只能包含字母、数字和下划线：{db_name}")

    try:
        conn = pymysql.connect(
            host=url.host or "127.0.0.1",
            port=url.port or 3306,
            user=url.username or "root",
            password=url.password or "",
            charset="utf8mb4",
            autocommit=True,
            connect_timeout=5,
        )
    except pymysql.MySQLError as exc:
        raise RuntimeError(
            "无法连接 MySQL。请检查：\n"
            "1. MySQL 服务是否已启动；\n"
            "2. backend/.env 里的 DB_USER / DB_PASSWORD 是否正确；\n"
            "3. 如果不想装 MySQL，直接双击 start_backend_sqlite.bat 使用 SQLite。\n"
            f"原始错误：{exc}"
        ) from exc

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        print(f"[db] MySQL 数据库已就绪：{db_name}")
    except pymysql.MySQLError as exc:
        raise RuntimeError(
            f"自动创建数据库 `{db_name}` 失败，请确认 MySQL 用户有 CREATE 权限。\n"
            f"原始错误：{exc}"
        ) from exc
    finally:
        conn.close()


ensure_database()

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    connect_args=connect_args,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
