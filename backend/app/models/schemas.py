from typing import List, Dict, Any, Optional, Sequence
from pydantic import BaseModel
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class MessageRequest(BaseModel):
    content: str
    thread_id: Optional[str] = None
    include_reasoning: bool = False


class ChatHistoryResponse(BaseModel):
    thread_id: str
    messages: List[Dict[str, Any]]
    reasoning_history: Optional[List[Dict[str, str]]] = None


def reduce_reasoning_history(
    left: Optional[List[Dict[str, str]]], right: Optional[Sequence[Dict[str, str]]]
) -> List[Dict[str, str]]:
    if not left:
        left = []
    if not right:
        return left

    if isinstance(right, dict):
        right = [right]
    elif not isinstance(right, list):
        return left

    return left + list(right)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    input_content: Optional[str] = None
    reasoning_history: Annotated[
        Optional[List[Dict[str, str]]], reduce_reasoning_history
    ] = None

    remaining_steps: Optional[int] = None
