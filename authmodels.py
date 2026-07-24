"""Pydantic models for authentication layer"""
from __future__ import annotations

from pydantic import BaseModel, Field

class UserCreate(BaseModel):
    """Request body for POST /register"""
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=72, description="Plaintext password.",)#not stored as is , 72 len is bcrypt's hard limit.

class UserOut(BaseModel):
    """Public-facing useer shape --password hash isnot included"""
    id: int
    username: str
    created_at: str

class Token(BaseModel):
    """Response body for POST /token."""
 
    access_token: str
    token_type: str = "bearer" 
 
 
class TokenData(BaseModel):
    """Decoded JWT payload shape, used internally by get_current_user."""
 
    username: str | None = None