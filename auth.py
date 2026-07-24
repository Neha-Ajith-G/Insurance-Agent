"""
auth: password hashing, JWT issuing/validation, and the
get_current_user dependency other endpoints protect themselves with.

This is the standard FastAPI OAuth2PasswordBearer + JWT pattern from
FastAPI's own security documentation -- deliberately not reinvented,
since the point of this layer is to learn that exact, widely-used shape.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import bcrypt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from authdb import get_user_by_username
from authmodels import UserOut

# Loads a local .env file if one exists (harmless no-op if it doesn't --
# e.g. in Docker, where real environment variables are set directly via
# docker-compose.yml's `environment:` block instead of a .env file).
load_dotenv()

# REQUIRED, no fallback default -- a hardcoded secret key in source would
# mean anyone who ever sees this code (a git history, a shared screen)
# could forge valid tokens for any user, forever. Generate one with:
#   python -c "import secrets; print(secrets.token_hex(32))"
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "AUTH_SECRET_KEY environment variable is not set. Generate one with: "
        "python -c \"import secrets; print(secrets.token_hex(32))\" -- "
        "and set it before starting the app."
    )

ALGORITHM = os.environ.get("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# bcrypt handles salting internally -- gensalt() generates a fresh
# random salt per call, embedded in the returned hash itself, so no
# separate salt storage is needed.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(username: str, expires_delta: timedelta | None = None) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def authenticate_user(username: str, password: str) -> dict | None:
    """Returns the user record on success, None on any failure (unknown
    username or wrong password) -- deliberately not distinguishing which,
    in either the return value or the eventual HTTP response, so a login
    attempt can't be used to enumerate valid usernames."""
    user = get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserOut:
    """
    The dependency any protected endpoint declares to require a logged-in
    user, e.g.:

        @app.get("/something")
        def something(current_user: UserOut = Depends(get_current_user)):
            ...

    FastAPI extracts the "Authorization: Bearer <token>" header via
    oauth2_scheme automatically; this function only has to validate it.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError as e:
        raise credentials_exception from e

    user = get_user_by_username(username)
    if user is None:
        raise credentials_exception

    return UserOut(id=user["id"], username=user["username"], created_at=user["created_at"])