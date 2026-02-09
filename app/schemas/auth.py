import re
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, SecretStr, field_validator

# Authentication schemas
class UserCreate(BaseModel):
    """
    Schema for user registration inputs.
    """
    email: EmailStr = Field(..., description="User's email address")

    # SecretStr prevents the password from being logged in tracebacks
    password: SecretStr = Field(..., description="User's password", min_length=8, max_length=18)
    @field_validator("password")
    @classmethod
    def validate_password(cls, v: SecretStr) -> SecretStr:
        """
        Enforce strong password policies.
        """
        password = v.get_secret_value()

        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r"[A-Z]", password):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[0-9]", password):
            raise ValueError("Password must contain at least one number")
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            raise ValueError("Password must contain at least one special character")
            
        return v
    

class Token(BaseModel):
    """
    Schema for the JWT Access Token Response.
    """
    access_token: str = Field(..., description="The JWT Access Token")
    token_type: str = Field(default="bearer", description="The type of token")
    expires_at: datetime = Field(..., description="The token expiration timestamp")


class UserResponse(BaseModel):
    """
    Public user profile schema (safe return to frontend).
    Notice we exclude the password here.     
    """
    id: int
    email: str
    token: Token