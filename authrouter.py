"""
auth_router: /register, /token, and a demonstration protected endpoint.

Kept as an APIRouter (not defined directly in app.py) so app.py just does
`app.include_router(auth_router)` -- the same "each file owns one
concern" pattern as the rest of the project.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from auth import authenticate_user, create_access_token, get_current_user, hash_password
from authdb import UserExistsError, create_user, init_db
from authmodels import Token, UserCreate, UserOut

router = APIRouter(tags=["auth"])

# Ensures the users table exists as soon as this module is imported --
# app.py imports this router at startup, so this runs before any
# request can arrive.
init_db()


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(user: UserCreate) -> UserOut:
    try:
        created = create_user(user.username, hash_password(user.password))
    except UserExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    return UserOut(id=created["id"], username=created["username"], created_at=created["created_at"])


@router.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()) -> Token:
    # OAuth2PasswordRequestForm expects username/password as form fields,
    # not JSON -- that's the OAuth2 spec's convention (and what FastAPI's
    # auto-generated /docs "Authorize" button sends), not a personal choice.
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(username=user["username"])
    return Token(access_token=access_token, token_type="bearer")


@router.get("/users/me", response_model=UserOut)
def read_current_user(current_user: UserOut = Depends(get_current_user)) -> UserOut:
    """Demonstration protected endpoint -- proves the whole chain works:
    register -> login -> use the token -> get back exactly who you are.
    Any future endpoint (document listing, chat) protects itself the
    same way: just add `current_user: UserOut = Depends(get_current_user)`
    as a parameter."""
    return current_user