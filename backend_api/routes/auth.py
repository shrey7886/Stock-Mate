from __future__ import annotations

import hashlib
import secrets
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends

from backend_api.core.security import create_jwt_token, decode_jwt_token, get_current_user
from backend_api.database.token_store import create_user, get_user_by_email, update_user_password
from backend_api.utils.email import send_password_reset_email
from backend_api.models.schemas import LoginRequest, LoginResponse, RegisterRequest, RegisterResponse, ForgotPasswordRequest, ResetPasswordRequest, ActionResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Hash password with PBKDF2-HMAC-SHA256. Returns (hash_hex, salt_hex)."""
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 260_000)
    return dk.hex(), salt


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored 'salt:hash' format."""
    parts = stored_hash.split(":")
    if len(parts) != 2:
        return False
    salt, expected = parts
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 260_000)
    return secrets.compare_digest(dk.hex(), expected)


@router.post("/register", response_model=RegisterResponse)
def register(payload: RegisterRequest) -> RegisterResponse:
    if not payload.email or not payload.password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    if len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    user_id = str(uuid4())
    hash_hex, salt = _hash_password(payload.password)
    stored_hash = f"{salt}:{hash_hex}"

    success = create_user(
        user_id=user_id,
        email=payload.email.strip().lower(),
        password_hash=stored_hash,
        display_name=payload.display_name,
    )
    if not success:
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    token = create_jwt_token(user_id=user_id, email=payload.email.strip().lower())
    return RegisterResponse(
        access_token=token,
        token_type="bearer",
        user_id=user_id,
        message="Account created successfully",
    )


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    if not payload.email or not payload.password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    user = get_user_by_email(payload.email.strip().lower())
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not _verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_jwt_token(user_id=user["user_id"], email=user["email"])
    return LoginResponse(access_token=token, token_type="bearer", user_id=user["user_id"])


@router.post("/forgot-password", response_model=ActionResponse)
def forgot_password(payload: ForgotPasswordRequest, background_tasks: BackgroundTasks) -> ActionResponse:
    user = get_user_by_email(payload.email.strip().lower())
    if user:
        # Create a 15-minute token
        token = create_jwt_token(user_id=user["user_id"], email=user["email"], exp_minutes=15)
        background_tasks.add_task(send_password_reset_email, user["email"], token)
    
    # Always return success to prevent email enumeration
    return ActionResponse(success=True, message="If that email is registered, a password reset link has been sent.")


@router.post("/reset-password", response_model=ActionResponse)
def reset_password(payload: ResetPasswordRequest) -> ActionResponse:
    if len(payload.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
        
    try:
        token_data = decode_jwt_token(payload.token)
    except HTTPException:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
        
    email = token_data.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Invalid token payload")
        
    # Verify user exists
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Hash new password
    hash_hex, salt = _hash_password(payload.new_password)
    stored_hash = f"{salt}:{hash_hex}"
    
    # Update DB
    success = update_user_password(email=email, new_password_hash=stored_hash)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update password")
        
    return ActionResponse(success=True, message="Password has been reset successfully")

@router.get("/me")
def get_me(current_user: dict = Depends(get_current_user)):
    user_id = current_user.get("sub")
    email = current_user.get("email")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid auth payload")
    
    # Return user details resolving the Promise successfully
    return {"user_id": user_id, "email": email}
