"""Pydantic models and TypedDicts for API requests/responses and internal state."""

from typing import List, Dict, Any, Optional, Sequence, TypedDict, Annotated
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class MessageRequest(BaseModel):
    """Schema for incoming chat message requests."""

    content: str
    thread_id: Optional[str] = None
    include_reasoning: bool = False


class ChatHistoryResponse(BaseModel):
    """Schema for chat history API response."""

    thread_id: str
    messages: List[Dict[str, Any]]
    reasoning_history: Optional[List[Dict[str, str]]] = None


def reduce_reasoning_history(
    left: Optional[List[Dict[str, str]]], right: Optional[Sequence[Dict[str, str]]]
) -> List[Dict[str, str]]:
    """Reducer function to append new reasoning steps to the history."""
    if not left:
        left = []
    if not right:
        return left

    if isinstance(right, dict):
        right_list = [right]
    elif isinstance(right, list):
        right_list = list(right)
    else:
        return left

    return left + right_list


class AgentState(TypedDict, total=False):
    """Represents the state of the LangGraph agent."""

    messages: Annotated[List[BaseMessage], add_messages]
    input_content: Optional[str]
    reasoning_history: Annotated[
        Optional[List[Dict[str, str]]], reduce_reasoning_history
    ]
    remaining_steps: Optional[int]
