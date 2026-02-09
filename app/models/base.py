from datetime import datetime, UTC
from typing import List, Optional
from sqlmodel import Field, SQLModel, Relationship

# Base Database Model
class BaseModel(SQLModel):
    """
    Abstract base model that adds common fields to all tables.
    Using an abstract class ensures consistency across our schema.
    """
    # always use UTC in production to avoid timezone headaches
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))