from typing import TYPE_CHECKING, List
import bcrypt
from sqlmodel import Field, Relationship
from app.models.base import BaseModel

# Prevent circular imports for type hinting
if TYPE_CHECKING:
    from app.models.session import Session

#
# User Model
#
class User(BaseModel, table=True):
    """
    Represents a registered user in the system
    """

    # Primary key
    id: int = Field(default=None, primary_key=True)

    # Email must be unique and indexed for fast lookups during login
    email: str = Field(unique=True, index=True)

    # Never store plain text passwords. we store the Bcrypt hash.
    hashed_password: str

    # Relationship: One user has many chat sessions
    sessions: List["Session"] = Relationship(back_populates="user")
    def verify_passwords(self, password: str) -> bool:
        """
        Verifies a raw password against the stored hash.
        """
        return bcrypt.checkpw(password.encode("utf-8"), self.hashed_password.encode("utf-8"))
    
    # here we have embedded the password hashing logic directly into the model.
    # Basically 'encapsulation', logic for handling user data lives with the user data, preventing 
    # security mistakes elsewhere in the app
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Generates a secure Bcrypt hash/salt for a new password.
        """
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")