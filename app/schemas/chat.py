import re
from typing import List, Literal
from typing_extensions import Self
from pydantic import BaseModel, Field, field_validator

# chat schemas
class Message(BaseModel):
    """
    Represents a single message in the conversation history.
    """
    role: Literal["user", "assistant", "system"] = Field(..., description="Who sent the message")
    content: str = Field(..., description="The message content", min_length=1, max_length=3000)
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """
        Sanitization: Prevent basic XSS or injection attacks.
        """
        if re.search(r"<script.*?>.*?</script>", v, re.IGNORECASE | re.DOTALL):
            raise ValueError("Content contains potentially harmful script tags")
        return v
    

class ChatRequest(BaseModel):
    """
    Payload sent to the /chat endpoint
    """
    messages: List[Message] = Field(..., min_length=1)

class ChatResponse(BaseModel):
    """
    Standard response from the /chat endpoint.
    """
    messages: List[Message]

class StreamResponse(BaseModel):
    """
    Chunk Format for Server-Sent Events (SSE) streaming
    """
    content: str = Field(default="")
    done: bool = Field(default=False)