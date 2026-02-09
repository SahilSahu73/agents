# We need a model for langgraph persistence. Langgraph is stateful, if the server restarts 
# we don't want the AI to forget what step it was on.

# we need a Thread model that acts as an anchor for these checkpoints.
from datetime import UTC, datetime
from sqlmodel import Field, SQLModel

# Thread model (Langgraph state)
class Thread(SQLModel, table=True):
    """
    Acts as a lightweight anchor for LangGraph checkpoints.
    The actual state blob is stored by the AsyncPostgresSaver,
    but we need this table to validate thread existence.
    """
    id: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))