"""
auth_db: SQLAlchemy-backed storage for users.

Kept as its own module, same layering convention as the rest of the
project -- this one owns "where user records live," nothing else.
auth.py and auth_router.py call create_user()/get_user_by_username()
and never touch SQLAlchemy directly, so this file is the only place
that would need to change if the backing store ever moved again (e.g.
SQLite -> Postgres is just a DATABASE_URL change, nothing more).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker

# Configurable via env var, same reasoning as docclient.py's model/
# base_url -- lets Docker (or a future Postgres upgrade) point this
# somewhere else without a code change. Note the SQLAlchemy URL scheme
# ("sqlite:///./auth.db") rather than a bare file path.
DATABASE_URL = os.environ.get("AUTH_DATABASE_URL", "sqlite:///./auth.db")

# check_same_thread=False is SQLite-specific: FastAPI's sync route
# handlers run in a threadpool, and SQLite's default driver forbids a
# connection created on one thread from being used on another. Gated on
# the URL scheme rather than applied unconditionally, since this
# wouldn't apply (or be needed) for e.g. a Postgres URL later.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserExistsError(Exception):
    """Raised when registering a username that's already taken."""


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(String, nullable=False)


def init_db() -> None:
    """Creates the users table if it doesn't exist yet. Safe to call
    repeatedly (e.g. on every app startup)."""
    Base.metadata.create_all(bind=engine)


def _to_dict(user: User) -> dict:
    # auth.py/auth_router.py expect plain dict-style access (user["id"],
    # user["hashed_password"], etc.) -- converting here means neither of
    # those files needs to know or care that this is a SQLAlchemy model
    # underneath, same as before when it was a sqlite3.Row.
    return {
        "id": user.id,
        "username": user.username,
        "hashed_password": user.hashed_password,
        "created_at": user.created_at,
    }


def create_user(username: str, hashed_password: str) -> dict:
    """Inserts a new user. Raises UserExistsError on a duplicate
    username rather than letting a raw IntegrityError escape --
    callers shouldn't need to know this is backed by SQLAlchemy at all."""
    db = SessionLocal()
    try:
        user = User(
            username=username,
            hashed_password=hashed_password,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        db.add(user)
        try:
            db.commit()
        except IntegrityError as e:
            db.rollback()
            raise UserExistsError(f"Username {username!r} is already taken") from e
        db.refresh(user)
        return _to_dict(user)
    finally:
        db.close()


def get_user_by_username(username: str) -> dict | None:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        return _to_dict(user) if user else None
    finally:
        db.close()