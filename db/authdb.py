"""
auth_db: SQLAlchemy-backed storage for users.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("AUTH_DATABASE_URL", "sqlite:///./auth.db")

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
    Base.metadata.create_all(bind=engine)


def _to_dict(user: User) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "hashed_password": user.hashed_password,
        "created_at": user.created_at,
    }


def create_user(username: str, hashed_password: str) -> dict:
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
