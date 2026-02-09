from typing import Annotated
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# LangGraph State Schemas
# Langgraph works by passing a state object between nodes (Agents, Tools, Memory)
# We need to explicitly define what that state looks like

class GraphState(BaseModel):
    """
    The central state object passed between graph nodes.
    """

    # add_messages() is a reducer. It tells Langgraph: "When new message
    # comes in, append it to the list rather than overwriting it."
    messages: Annotated[list, add_messages] = Field(
        default_factory=list,
        description="The conversation history"
    )

    # mem0ai will be used to retrieve context from LTM
    long_term_memory: str = Field(
        default="",
        description="Relevant context extracted from vector store"
    )