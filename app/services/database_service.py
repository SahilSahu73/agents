from typing import List, Optional
from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool
from sqlmodel import Session, SQLModel, create_engine, select, col

from app.core.config import Environment, settings
from app.core.logging import logger
from app.models.session import Session as ChatSession
from app.models.user import User


# Database Service
class DatabaseService:
    """
    Singleton service handling all database interactions.
    Manages the connection pool and provides clean CRUD interfaces.
    Uses SQLModel for ORM Operations and maintains a connection pool.
    """
    def __init__(self):
        """
        Initialize the engine with robust pooling settings.
        """
        try:
            # pool_size: no. of connections to keep open permanently
            # max_overflow: no. of temporary connections to allow during spikes
            pool_size = settings.POSTGRES_POOL_SIZE
            max_overflow = settings.POSTGRES_MAX_OVERFLOW

            # Create the connection URL from settings
            connection_url = (
                f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
                f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
            )

            # Configuring the QueuePool
            self.engine = create_engine(
                connection_url,
                pool_pre_ping=True,  # check if connection is alive before using it
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=30,     # Fail if no connection available after 30s
                pool_recycle=1800,   # Recycle connections every 30 mins to prevent stale sockets
            )
            # Create tables if they don't exist (code-first migration)
            SQLModel.metadata.create_all(self.engine)

            logger.info(
                "Database_initialized",
                environment=settings.ENVIRONMENT.value, 
                pool_size=pool_size,
                max_overflow=max_overflow,
            )

        except SQLAlchemyError as e:
            logger.error("Database_initialization_error", error=str(e), environment=settings.ENVIRONMENT.value)
            # In Dev, we might want to crash. In prod, maybe we want to retry.
            if settings.ENVIRONMENT != Environment.PRODUCTION:
                raise

    # User Management
    async def create_user(self, email: str, password_hash: str) -> User:
        """
        Create a new user with hashed password
        
        Args:
            email: User's email
            password_hash: hashed password

        Returns:
            User: The created user
        """
        with Session(self.engine) as session:
            user = User(email=email, hashed_password=password_hash)
            session.add(user)
            session.commit()
            session.refresh(user)
            logger.info("user_created", email=email)
            return user

    async def get_user(self, user_id: int) -> Optional[User]:
        """Get a User by User ID.
        
        Args:
            user_id: The ID of the user to retrieve

        Returns:
            Optional[User]: The user if found, None otherwise
        """
        with Session(self.engine) as session:
            user = session.get(User, user_id)
            return user

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Fetch user by email for login flow."""
        with Session(self.engine) as session:
            statement = select(User).where(User.email == email)
            return session.exec(statement).first()
        
    async def delete_user_by_email(self, email: str) -> bool:
        """Delete a user by email.
        
        Args:
            email: The email of the user to delete

        Returns:
            bool: True if deletion was successful, False if user not found
        """
        with Session(self.engine) as session:
            user = session.exec(select(User).where(User.email == email)).first()
            if not user:
                return False
            
            session.delete(user)
            session.commit()
            logger.info("User_deleted", email=email)
            return True

    # Session Management
    async def create_session(self, session_id: str, user_id: int, name: str = "") -> ChatSession:
        """Create a new chat session linked to a user.
        
        Args:
            session_id: The ID for the new session
            user_id: The ID of the user who owns the session
            name: Optional name for the session (defaults to an empty string)

        Returns:
            ChatSession: The created session
        """
        with Session(self.engine) as session:
            chat_session = ChatSession(id=session_id, user_id=user_id, name=name)
            session.add(chat_session)
            session.commit()
            session.refresh(chat_session)
            logger.info("session_created", session_id=session_id, user_id=user_id, name=name)
            return chat_session
        
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID.
        
        Args:
            session_id: The ID of the session to delete

        Returns:
            bool: True if deletion was successful, False if session not found
        """
        with Session(self.engine) as session:
            chat_session = session.get(ChatSession, session_id)
            if not chat_session:
                 return False
            
            session.delete(chat_session)
            session.commit()
            logger.info("session_deleted", session_id=session_id)
            return True
        
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a session by ID.
        
        Args:
            session_id: The ID of the session to retrieve

        Returns:
            Optional[ChatSession]: The session if found, None otherwise
        """
        with Session(self.engine) as session:
            chat_session = session.get(ChatSession, session_id)
            return chat_session
   
    async def get_user_sessions(self, user_id: int) -> List[ChatSession]:
        """List all chat history for a specific user.
        
        Args:
            user_id: The ID of the user

        Returns:
            List[ChatSession]: List of user's sessions
        """
        with Session(self.engine) as session:
            statement = select(ChatSession).where(ChatSession.user_id == user_id).order_by(col(ChatSession.created_at).desc())
            return list(session.exec(statement).all())
        
    async def update_session_name(self, session_id: str, name: str) -> ChatSession:
        """Update a session's name.

        Args:
            session_id: The ID of the session to update
            name: The new name for the session

        Returns:
            ChatSession: The updated session

        Raises:
            HTTPException: If session is not found
        """
        with Session(self.engine) as session:
            chat_session = session.get(ChatSession, session_id)
            if not chat_session:
                raise HTTPException(status_code=404, detail="Session not found")

            chat_session.name = name
            session.add(chat_session)
            session.commit()
            session.refresh(chat_session)
            logger.info("session_name_updated", session_id=session_id, name=name)
            return chat_session

    def get_session_maker(self):
        """Get a session maker for creating database sessions.

        Returns:
            Session: A SQLModel session maker
        """
        return Session(self.engine)

    async def health_check(self) -> bool:
        """Check database connection health.

        Returns:
            bool: True if database is healthy, False otherwise
        """
        try:
            with Session(self.engine) as session:
                # Execute a simple query to check connection
                session.exec(select(1)).first()
                return True
        except Exception as e:
            logger.error("database_health_check_failed", error=str(e))
            return False

# Create a global singleton instance
database_service = DatabaseService()