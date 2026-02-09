from typing import TYPE_CHECKING, List
from sqlmodel import Field, Relationship
from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.user import User

# Session Model
class Session(BaseModel, table=True):
    """
    Represents a specific chat conversation/thread.
    Links AI's memory to a specific context.
    """
    
    # String IDs (uuid)
    id: str = Field(primary_key=True)

    # Foreign Key -> linking session to a specific user
    user_id: int = Field(foreign_key="user.id")

    # chat name
    name: str = Field(default="New Chat")

    # Relationship link back to the user
    user: "User" = Relationship(back_populates="sessions")

# This creates a session model that links to the user model via a foreign key.
# Each session represents a distinct conversation context for the AI.