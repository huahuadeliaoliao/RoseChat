import logging
import json
from typing import Dict, List, Any, Optional, AsyncGenerator

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AIMessageChunk
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.base import BaseCheckpointSaver
from psycopg_pool import AsyncConnectionPool

from app.config import settings
from app.models.schemas import AgentState

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, checkpointer: BaseCheckpointSaver, db_pool: AsyncConnectionPool):
        self.checkpointer = checkpointer
        self.db_pool = db_pool
        if not isinstance(checkpointer, BaseCheckpointSaver):
            raise TypeError(
                "ChatService requires a valid BaseCheckpointSaver instance."
            )
        if not isinstance(db_pool, AsyncConnectionPool):
            raise TypeError(
                "ChatService requires a valid AsyncConnectionPool instance."
            )
        self.chat_agent = self._init_chat_agent()
        if self.chat_agent:
            logger.info(
                "ChatService initialized with provided checkpointer and compiled chat agent."
            )
        else:
            logger.error("ChatService initialized but chat agent compilation failed.")

    def _init_chat_agent(self) -> Optional[Any]:
        try:
            logger.info("Initializing chat model...")
            chat_model = ChatOpenAI(
                model=settings.model_name,
                temperature=settings.temperature,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )

            logger.info("Initializing StateGraph...")
            workflow = StateGraph(AgentState)

            def agent_node(state: AgentState):
                history = state.get("messages", [])
                new_content = state.get("input_content")

                logger.info("--- Agent Node Start ---")
                logger.info(
                    f"Loaded 'input_content' present: {new_content is not None}"
                )
                logger.info(f"Loaded history message count: {len(history)}")
                for i, msg in enumerate(history):
                    role = msg.type if hasattr(msg, "type") else "unknown"
                    content_preview = str(getattr(msg, "content", ""))[:150]
                    logger.debug(
                        f"  History[{i}] Role: {role}, Content Preview: '{content_preview}...', ID: {getattr(msg, 'id', 'N/A')}"
                    )
                logger.debug("--- End Loaded History Log ---")

                if new_content is None and not history:
                    logger.warning(
                        "Agent node called with no history and no new input."
                    )
                    return {}
                elif new_content is None:
                    messages_to_llm = history
                    logger.info(
                        "Agent node processing based on history only (no new input_content)."
                    )
                else:
                    messages_to_llm = history + [HumanMessage(content=new_content)]
                    logger.info("Agent node combining history with new input_content.")

                logger.info(f"Messages sent to LLM count: {len(messages_to_llm)}")
                logger.debug("--- Agent Node Input to LLM ---")
                for i, msg in enumerate(messages_to_llm):
                    role = "Unknown"
                    if isinstance(msg, HumanMessage):
                        role = "User"
                    elif isinstance(msg, AIMessage):
                        role = "Assistant"
                    content_preview = str(getattr(msg, "content", ""))[:150]
                    logger.debug(
                        f"  [{i}] Role: {role}, Content Preview: '{content_preview}...'"
                    )
                logger.debug("--- End Agent Node Input to LLM ---")

                if not messages_to_llm:
                    logger.error("No messages to send to LLM after processing.")
                    return {"input_content": None}

                try:
                    response = chat_model.invoke(messages_to_llm)
                    logger.debug(f"Agent node got raw response from LLM: {response}")

                    messages_to_add = []
                    if new_content is not None:
                        messages_to_add.append(HumanMessage(content=new_content))
                    messages_to_add.append(response)

                    update = {
                        "messages": messages_to_add,
                        "input_content": None,
                    }

                    logger.info("--- Agent Node Finish ---")
                    return update
                except Exception as node_exc:
                    logger.error(
                        f"Error invoking chat model in agent node: {node_exc}",
                        exc_info=True,
                    )
                    return {"input_content": None}

            workflow.add_node("agent", agent_node)
            workflow.set_entry_point("agent")
            workflow.add_edge(START, "agent")
            workflow.set_finish_point("agent")

            logger.info("Compiling workflow with checkpointer...")
            compiled_graph = workflow.compile(checkpointer=self.checkpointer)
            logger.info("Workflow compiled successfully.")
            return compiled_graph
        except Exception as e:
            logger.error(f"Error initializing chat agent: {str(e)}", exc_info=True)
            return None

    async def stream_message(
        self, content: str, thread_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        if not self.chat_agent:
            logger.error("Chat agent is not initialized for streaming.")
            raise RuntimeError("ChatService's chat_agent is not initialized.")
        if not self.checkpointer:
            logger.error("Checkpointer is not available in ChatService for streaming.")
            raise RuntimeError("ChatService's checkpointer is not available.")

        if thread_id and not isinstance(thread_id, str):
            logger.error(
                f"Invalid thread_id type received for streaming: {type(thread_id)}",
                extra={"thread_id": thread_id},
            )
            raise TypeError("thread_id must be a string or None")

        config = {"configurable": {"thread_id": thread_id}}
        input_payload = {"input_content": content}
        current_thread_id = thread_id

        logger.info(
            f"Streaming chat agent response for thread_id: {thread_id}",
            extra={"thread_id": thread_id},
        )

        try:
            async for event in self.chat_agent.astream(
                input_payload, config, stream_mode="messages"
            ):
                if isinstance(event, tuple) and len(event) > 0:
                    chunk = event[0]
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        yield f"data: {json.dumps({'type': 'content', 'content': chunk.content}, ensure_ascii=False)}\n\n"

            final_state = await self.chat_agent.aget_state(config)
            final_thread_id = "unknown"  # Default value
            if final_state and final_state.config:
                determined_thread_id = final_state.config.get("configurable", {}).get(
                    "thread_id"
                )
                if determined_thread_id:
                    final_thread_id = determined_thread_id
                else:
                    logger.warning(
                        f"Could not determine final thread_id from state for initial id {current_thread_id}, using initial.",
                        extra={"initial_thread_id": current_thread_id},
                    )
                    final_thread_id = current_thread_id or "unknown_fallback"
            else:
                logger.warning(
                    f"Could not get final state/config for initial id {current_thread_id}, using initial.",
                    extra={"initial_thread_id": current_thread_id},
                )
                final_thread_id = current_thread_id or "unknown_fallback"

            logger.info(
                f"Streaming finished for thread_id: {final_thread_id}",
                extra={"thread_id": final_thread_id},
            )
            yield f"event: end\ndata: {json.dumps({'thread_id': final_thread_id}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(
                f"Error streaming message for thread_id {thread_id}: {e}",
                exc_info=True,
                extra={"thread_id": thread_id},
            )
            error_message = f"An error occurred: {type(e).__name__}"
            yield f"event: error\ndata: {json.dumps({'error': error_message}, ensure_ascii=False)}\n\n"

    async def get_history(self, thread_id: str) -> Dict[str, Any]:
        if not self.chat_agent:
            logger.error(
                "Chat agent is not initialized.", extra={"thread_id": thread_id}
            )
            raise RuntimeError("ChatService's chat_agent is not initialized.")
        if not self.checkpointer:
            logger.error(
                "Checkpointer is not available in ChatService.",
                extra={"thread_id": thread_id},
            )
            raise RuntimeError("ChatService's checkpointer is not available.")

        if not thread_id or not isinstance(thread_id, str):
            logger.error(
                f"Invalid thread_id provided for get_history: {thread_id}",
                extra={"thread_id": thread_id},
            )
            raise ValueError(
                "A valid thread_id string must be provided to get history."
            )

        config = {"configurable": {"thread_id": thread_id}}
        logger.info(
            f"Getting state for thread_id: {thread_id}", extra={"thread_id": thread_id}
        )
        try:
            state_snapshot = await self.chat_agent.aget_state(config)
            if (
                not state_snapshot
                or not state_snapshot.values
                or "messages" not in state_snapshot.values
            ):
                logger.warning(
                    f"No state or messages found for thread_id: {thread_id}",
                    extra={"thread_id": thread_id},
                )
                return {"thread_id": thread_id, "messages": []}

            logger.info(
                f"State retrieved successfully for thread_id: {thread_id}",
                extra={"thread_id": thread_id},
            )
            messages = []
            state_messages = state_snapshot.values.get("messages", [])
            for msg in state_messages:
                if isinstance(msg, BaseMessage):
                    messages.append(
                        {
                            "role": "user"
                            if isinstance(msg, HumanMessage)
                            else "assistant",
                            "content": msg.content,
                        }
                    )
                else:
                    logger.warning(
                        f"Non-BaseMessage object found in state messages for thread {thread_id}: {type(msg)}",
                        extra={"thread_id": thread_id, "message_type": str(type(msg))},
                    )

            return {"thread_id": thread_id, "messages": messages}
        except Exception as e:
            logger.error(
                f"Error getting history for thread_id {thread_id}: {e}",
                exc_info=True,
                extra={"thread_id": thread_id},
            )
            if "connection" in str(e).lower() and "closed" in str(e).lower():
                logger.critical(
                    "Database connection appears closed during get_state. Check lifespan management.",
                    extra={"thread_id": thread_id},
                )
            if isinstance(e, ValueError) and (
                "no checkpoint found for config" in str(e).lower()
                or "not found" in str(e).lower()
            ):
                logger.warning(
                    f"No checkpoint found for thread_id {thread_id} during get_history.",
                    extra={"thread_id": thread_id},
                )
                return {"thread_id": thread_id, "messages": []}
            raise

    async def list_threads(self) -> List[str]:
        if not self.db_pool:
            logger.error("Database pool is not available in ChatService.")
            raise RuntimeError("Database pool not configured for listing threads.")

        logger.info("Querying database for distinct thread IDs...")
        thread_ids = []
        sql = "SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id IS NOT NULL AND thread_id != '';"
        try:
            async with self.db_pool.connection() as aconn:
                async with aconn.cursor() as acur:
                    await acur.execute(sql)
                    results = await acur.fetchall()
                    thread_ids = [row[0] for row in results]
            logger.info(
                f"Found {len(thread_ids)} distinct thread IDs.",
                extra={"thread_count": len(thread_ids)},
            )
            return thread_ids
        except Exception as db_exc:
            logger.error(
                f"Failed to query thread IDs from database: {db_exc}", exc_info=True
            )
            raise RuntimeError(
                "Failed to retrieve thread list from database"
            ) from db_exc
