from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class MessageRequest(BaseModel):
    content: str
    thread_id: Optional[str] = None


class MessageResponse(BaseModel):
    thread_id: str
    message: str


class ChatHistoryResponse(BaseModel):
    thread_id: str
    messages: List[Dict[str, Any]]


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    input_content: Optional[str] = None
