"""
Authentication module for admin panel.

Provides JWT token-based authentication with role verification.
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import JWTError, jwt
import bcrypt
import logging

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours

# Security scheme
security = HTTPBearer()

router = APIRouter(prefix="/auth", tags=["authentication"])


# Pydantic models
class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: dict


class User(BaseModel):
    username: str
    role: str
    full_name: Optional[str] = None


class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None


# In-memory user database (replace with actual database in production)
# Pre-computed bcrypt hash for "admin123"
USERS_DB = {
    "admin": {
        "username": "admin",
        "password_hash": "$2b$12$shwQQ8ba.6YDbOh9IFdrg.JxzwUAy.dkucdJqW0MkhwhADCk2UyQG",  # admin123
        "role": "admin",
        "full_name": "Administrator"
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Authenticate a user by username and password."""
    user = USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> User:
    """
    Dependency to get the current authenticated user from JWT token.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: User = Depends(get_current_user)):
            return {"user": user.username}
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        
        if username is None:
            raise credentials_exception
            
        token_data = TokenData(username=username, role=role)
        
    except JWTError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise credentials_exception
    
    user = USERS_DB.get(token_data.username)
    if user is None:
        raise credentials_exception
    
    return User(
        username=user["username"],
        role=user["role"],
        full_name=user.get("full_name")
    )


async def get_current_admin_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """
    Dependency to verify user has admin role.
    
    Usage:
        @app.delete("/admin/resource")
        async def admin_only(user: User = Depends(get_current_admin_user)):
            return {"admin": user.username}
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Authenticate user and return JWT access token.
    
    Default credentials:
    - Username: admin
    - Password: admin123
    
    Example:
    ```
    POST /api/auth/login
    {
        "username": "admin",
        "password": "admin123"
    }
    ```
    
    Returns:
    ```
    {
        "access_token": "eyJhbGc...",
        "token_type": "bearer",
        "expires_in": 28800,
        "user": {
            "username": "admin",
            "role": "admin",
            "full_name": "Administrator"
        }
    }
    ```
    """
    user = authenticate_user(request.username, request.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires
    )
    
    logger.info(f"User {user['username']} logged in successfully")
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # seconds
        user={
            "username": user["username"],
            "role": user["role"],
            "full_name": user.get("full_name")
        }
    )


@router.get("/me")
async def get_me(current_user: Annotated[User, Depends(get_current_user)]):
    """
    Get current authenticated user information.
    
    Requires: Authorization: Bearer <token>
    """
    return {
        "username": current_user.username,
        "role": current_user.role,
        "full_name": current_user.full_name
    }


@router.post("/logout")
async def logout(current_user: Annotated[User, Depends(get_current_user)]):
    """
    Logout endpoint (client should discard token).
    
    Note: JWT tokens are stateless, so logout is handled client-side
    by removing the token from storage.
    """
    logger.info(f"User {current_user.username} logged out")
    return {"message": "Successfully logged out"}
