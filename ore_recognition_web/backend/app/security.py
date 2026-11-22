"""密码哈希与签名 Token。"""
import base64
import hashlib
import hmac
import json
import time

from .config import SECRET_KEY, TOKEN_EXPIRE_DAYS

_ITERATIONS = 200_000


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def hash_password(password: str) -> str:
    salt = hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:16]
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), _ITERATIONS
    ).hex()
    return f"pbkdf2${_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt, digest = stored.split("$")
        calc = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), int(iterations)
        ).hex()
        return hmac.compare_digest(calc, digest)
    except (ValueError, TypeError):
        return False


def create_token(user_id: int) -> str:
    payload = {
        "uid": user_id,
        "exp": int(time.time()) + TOKEN_EXPIRE_DAYS * 86400,
    }
    body = _b64encode(json.dumps(payload, separators=(",", ":")).encode())
    signature = _b64encode(
        hmac.new(SECRET_KEY.encode(), body.encode(), hashlib.sha256).digest()
    )
    return f"{body}.{signature}"


def decode_token(token: str) -> int | None:
    try:
        body, signature = token.split(".")
        expected = _b64encode(
            hmac.new(SECRET_KEY.encode(), body.encode(), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(signature, expected):
            return None
        payload = json.loads(_b64decode(body))
        if payload.get("exp", 0) < time.time():
            return None
        return int(payload["uid"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return None
