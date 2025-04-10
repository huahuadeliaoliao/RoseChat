import logging
import json
import asyncio
import sys
import warnings
import functools
import contextlib
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple, Union, cast

from app.services.custom_openai import ChatOpenAIWithReasoning
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from langgraph.prebuilt import ToolNode, tools_condition

from langchain_mcp_adapters.client import MultiServerMCPClient

from psycopg_pool import AsyncConnectionPool
from app.config import settings
from app.models.schemas import (
    AgentState,
)

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_exception(exception_types=(Exception,), log_message=None):
    try:
        yield
    except exception_types as e:
        if log_message:
            logger.warning(f"{log_message}: {str(e)}")
        else:
            logger.warning(f"Suppressed exception: {str(e)}")


class ChatService:
    def __init__(
        self,
        checkpointer: BaseCheckpointSaver,
        db_pool: AsyncConnectionPool,
        mcp_client: Optional[MultiServerMCPClient] = None,
    ):
        self.checkpointer = checkpointer
        self.db_pool = db_pool
        self.mcp_client = mcp_client
        self.mcp_tools: List[BaseTool] = []

        if not isinstance(checkpointer, BaseCheckpointSaver):
            raise TypeError(
                "ChatService requires a valid BaseCheckpointSaver instance."
            )
        if not isinstance(db_pool, AsyncConnectionPool):
            raise TypeError(
                "ChatService requires a valid AsyncConnectionPool instance."
            )

        if self.mcp_client:
            try:
                self.mcp_tools = self.mcp_client.get_tools()
                logger.info(
                    f"Successfully loaded {len(self.mcp_tools)} tools from MCP client."
                )
                tool_names = [tool.name for tool in self.mcp_tools]
                logger.debug(f"Loaded MCP tool names: {tool_names}")
            except Exception as e:
                logger.error(
                    f"Failed to load tools from MCP client: {e}", exc_info=True
                )
                self.mcp_tools = []
        else:
            logger.info(
                "No MCP client provided to ChatService, skipping MCP tool loading."
            )

        self.chat_agent = self._init_chat_agent()
        if self.chat_agent:
            logger.info(
                "ChatService initialized with checkpointer and compiled chat agent."
            )
        else:
            logger.error("ChatService initialized but chat agent compilation failed.")

    def _init_chat_agent(self) -> Optional[Any]:
        try:
            logger.info("Initializing chat model (ChatOpenAIWithReasoning)...")
            chat_model = ChatOpenAIWithReasoning(
                model=settings.model_name,
                temperature=settings.temperature,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                streaming=True,
            )
            logger.info(f"Chat model initialized: {settings.model_name}")

            tools_for_agent = self.mcp_tools
            tool_names_str = (
                ", ".join([t.name for t in tools_for_agent])
                if tools_for_agent
                else "None"
            )
            logger.info(f"Tools available for agent: [{tool_names_str}]")

            def call_model_node(state: AgentState) -> Dict[str, Any]:
                history = state.get("messages", [])
                current_input = state.get("input_content")

                messages_to_llm = list(history)
                human_message_for_log = None
                if current_input is not None:
                    human_message_for_log = HumanMessage(content=current_input)
                    messages_to_llm.append(human_message_for_log)
                    logger.info("Agent node combining history with new input.")
                else:
                    logger.info("Agent node processing based on history only.")

                logger.info(f"Messages sent to LLM count: {len(messages_to_llm)}")

                if not messages_to_llm:
                    logger.error("No messages to send to LLM.")
                    return {"input_content": None}

                try:
                    if tools_for_agent:
                        logger.debug(
                            f"Binding {len(tools_for_agent)} tools to the model."
                        )
                        model_runnable = chat_model.bind_tools(tools_for_agent)
                    else:
                        logger.debug("No tools to bind to the model.")
                        model_runnable = chat_model

                    response = model_runnable.invoke(messages_to_llm)
                    logger.debug(f"Agent node got raw LLM response: {response}")

                    messages_to_add = [response]
                    update_dict: Dict[str, Any] = {
                        "messages": messages_to_add,
                        "input_content": None,
                    }

                    reasoning_content = None
                    if isinstance(response, AIMessage) and hasattr(
                        response, "additional_kwargs"
                    ):
                        reasoning_content = response.additional_kwargs.get(
                            "reasoning_content"
                        )

                    if reasoning_content:
                        query_for_reasoning = (
                            current_input
                            if current_input is not None
                            else (
                                history[-1].content
                                if history and isinstance(history[-1], HumanMessage)
                                else "N/A"
                            )
                        )
                        reasoning_entry = {
                            "query": query_for_reasoning,
                            "reasoning": reasoning_content,
                        }
                        update_dict["reasoning_history"] = [reasoning_entry]
                        logger.info(
                            f"Captured reasoning content (length: {len(reasoning_content)}) for query: '{query_for_reasoning[:50]}...'"
                        )
                    else:
                        pass

                    logger.info("--- Agent Node Finish ---")
                    return update_dict

                except Exception as node_exc:
                    logger.error(
                        f"Error invoking chat model in agent node: {node_exc}",
                        exc_info=True,
                    )
                    return {"input_content": None}

            tool_node = ToolNode(tools_for_agent) if tools_for_agent else None

            logger.info("Building StateGraph manually...")
            workflow = StateGraph(AgentState)

            workflow.add_node("agent", call_model_node)

            if tool_node and tools_for_agent:
                logger.info("Adding ToolNode and conditional edges to the graph.")
                workflow.add_node("tools", tool_node)
                workflow.add_conditional_edges(
                    "agent",
                    tools_condition,
                    {
                        "tools": "tools",
                        END: END,
                    },
                )
                workflow.add_edge("tools", "agent")
            else:
                logger.info(
                    "No tools configured, adding direct edge from agent to END."
                )
                workflow.add_edge("agent", END)

            workflow.set_entry_point("agent")

            logger.info("Compiling workflow...")
            compiled_graph = workflow.compile(checkpointer=self.checkpointer)
            logger.info("Workflow compiled successfully.")
            return compiled_graph

        except Exception as e:
            logger.error(
                f"Fatal error during chat agent initialization: {str(e)}", exc_info=True
            )
            return None

    async def _patched_astream(
        self, agent, input_payload, config, stream_mode="messages"
    ):
        stream_iterator = agent.astream(
            input_payload,
            config,
            stream_mode=stream_mode,
        )

        current_task = asyncio.current_task()
        is_cancelled = False

        try:
            stream_iter = stream_iterator.__aiter__()

            while True:
                if current_task.cancelled():
                    is_cancelled = True
                    logger.info("Cancellation detected, interrupt stream processing")
                    break

                try:
                    event = await asyncio.wait_for(stream_iter.__anext__(), timeout=1.0)
                    yield event
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.warning(f"An error occurred during stream iteration: {e}")
                    if isinstance(e, asyncio.CancelledError):
                        is_cancelled = True
                    break

        except asyncio.CancelledError:
            is_cancelled = True
        except Exception as e:
            logger.error(
                f"An unhandled exception occurred during stream processing: {e}",
                exc_info=True,
            )
        finally:
            if is_cancelled:
                if hasattr(stream_iterator, "_callbacks"):
                    with suppress_exception(
                        log_message="Error cleaning up stream callback"
                    ):
                        stream_iterator._callbacks = []

                if hasattr(stream_iterator, "close"):
                    with suppress_exception(
                        log_message="Error closing stream iterator"
                    ):
                        await stream_iterator.close()

                stream_iterator = None
                stream_iter = None

    async def stream_message(
        self,
        content: str,
        thread_id: Optional[str] = None,
        include_reasoning: bool = False,
    ) -> AsyncGenerator[str, None]:
        if not self.chat_agent:
            logger.error("Chat agent is not initialized for streaming.")
            yield f"event: error\ndata: {json.dumps({'error': 'Chat service not available'}, ensure_ascii=False)}\n\n"
            return

        config = {"configurable": {"thread_id": thread_id}}
        current_thread_id = thread_id

        input_payload = {"input_content": content}
        logger.info(
            f"Streaming chat agent response for thread_id: {thread_id}",
            extra={"thread_id": thread_id, "input_content_preview": content[:100]},
        )

        current_reasoning_chunk = ""

        try:
            async for event in self._patched_astream(
                self.chat_agent, input_payload, config, stream_mode="messages"
            ):
                current_task = asyncio.current_task()
                if current_task.cancelled():
                    logger.info(
                        f"Streaming task was cancelled for thread_id: {current_thread_id}",
                        extra={"thread_id": current_thread_id},
                    )
                    break

                if isinstance(event, tuple) and len(event) > 0:
                    chunk = event[0]
                    if isinstance(chunk, AIMessageChunk):
                        if (
                            include_reasoning
                            and hasattr(chunk, "additional_kwargs")
                            and "reasoning_content" in chunk.additional_kwargs
                        ):
                            new_reasoning = chunk.additional_kwargs.get(
                                "reasoning_content", ""
                            )
                            if (
                                new_reasoning
                                and new_reasoning != current_reasoning_chunk
                            ):
                                yield f"data: {json.dumps({'type': 'reasoning', 'content': new_reasoning}, ensure_ascii=False)}\n\n"
                                current_reasoning_chunk = new_reasoning

                        if chunk.content:
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk.content}, ensure_ascii=False)}\n\n"
                    elif isinstance(chunk, ToolMessage):
                        yield f"data: {json.dumps({'type': 'tool_result', 'tool_call_id': chunk.tool_call_id, 'name': chunk.name, 'content': str(chunk.content)[:200] + '...'}, ensure_ascii=False)}\n\n"

                elif isinstance(event, Exception):
                    logger.error(
                        f"Error received during stream for thread {current_thread_id}: {event}",
                        exc_info=event,
                    )
                    error_payload = json.dumps(
                        {"error": f"Stream failed: {type(event).__name__}"}
                    )
                    yield f"event: error\ndata: {error_payload}\n\n"
                    return

            current_task = asyncio.current_task()
            if not current_task.cancelled():
                try:
                    final_state = await asyncio.wait_for(
                        self.chat_agent.aget_state(config), timeout=2.0
                    )

                    final_thread_id = "unknown"
                    if final_state and final_state.config:
                        determined_thread_id = final_state.config.get(
                            "configurable", {}
                        ).get("thread_id")
                        if determined_thread_id:
                            final_thread_id = determined_thread_id
                        else:
                            logger.warning(
                                f"Could not determine final thread_id from state config, using initial: {current_thread_id}"
                            )
                            final_thread_id = current_thread_id or "unknown_fallback"
                    else:
                        logger.warning(
                            f"Could not get final state/config, using initial thread_id: {current_thread_id}"
                        )
                        final_thread_id = current_thread_id or "unknown_fallback"

                    logger.info(
                        f"Streaming finished successfully for thread_id: {final_thread_id}",
                        extra={"thread_id": final_thread_id},
                    )
                    yield f"event: end\ndata: {json.dumps({'thread_id': final_thread_id}, ensure_ascii=False)}\n\n"

                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logger.info(
                        f"Timed out or cancelled while getting final state for thread_id: {current_thread_id}"
                    )
            else:
                logger.info(
                    f"Stream cancelled for thread_id: {current_thread_id}",
                    extra={"thread_id": current_thread_id},
                )

        except asyncio.CancelledError:
            logger.info(
                f"Stream cancelled for thread_id: {current_thread_id}",
                extra={"thread_id": current_thread_id},
            )
        except Exception as e:
            logger.error(
                f"Unhandled error during message streaming for thread_id {current_thread_id}: {e}",
                exc_info=True,
                extra={"thread_id": current_thread_id},
            )
            error_message = f"An unexpected error occurred: {type(e).__name__}"
            yield f"event: error\ndata: {json.dumps({'error': error_message}, ensure_ascii=False)}\n\n"

    async def get_history(
        self, thread_id: str, include_reasoning: bool = False
    ) -> Dict[str, Any]:
        if not self.chat_agent:
            logger.error(
                "Chat agent not initialized for get_history.",
                extra={"thread_id": thread_id},
            )
            raise RuntimeError("ChatService's chat_agent is not initialized.")
        if not self.checkpointer:
            logger.error(
                "Checkpointer not available for get_history.",
                extra={"thread_id": thread_id},
            )
            raise RuntimeError("ChatService's checkpointer is not available.")
        if not thread_id or not isinstance(thread_id, str):
            logger.error(f"Invalid thread_id provided for get_history: {thread_id}")
            raise ValueError("A valid thread_id string must be provided.")

        config = {"configurable": {"thread_id": thread_id}}
        logger.info(
            f"Getting state snapshot for thread_id: {thread_id}",
            extra={"thread_id": thread_id},
        )

        try:
            state_snapshot = await self.chat_agent.aget_state(config)

            if not state_snapshot or not state_snapshot.values:
                logger.warning(
                    f"No state found for thread_id: {thread_id}",
                    extra={"thread_id": thread_id},
                )
                return {
                    "thread_id": thread_id,
                    "messages": [],
                    "reasoning_history": [] if include_reasoning else None,
                }

            logger.info(
                f"State retrieved successfully for thread_id: {thread_id}",
                extra={"thread_id": thread_id},
            )

            messages_output = []
            state_messages = state_snapshot.values.get("messages", [])
            for msg in state_messages:
                if isinstance(msg, BaseMessage):
                    role = (
                        "user"
                        if isinstance(msg, HumanMessage)
                        else "assistant"
                        if isinstance(msg, AIMessage)
                        else "tool"
                        if isinstance(msg, ToolMessage)
                        else "system"
                    )
                    message_data = {"role": role, "content": msg.content}
                    if isinstance(msg, ToolMessage) and hasattr(msg, "tool_call_id"):
                        message_data["tool_call_id"] = msg.tool_call_id
                        if hasattr(msg, "name"):
                            message_data["name"] = msg.name
                    if (
                        isinstance(msg, AIMessage)
                        and hasattr(msg, "tool_calls")
                        and msg.tool_calls
                    ):
                        try:
                            json.dumps(msg.tool_calls)
                            message_data["tool_calls"] = msg.tool_calls
                        except TypeError:
                            logger.warning(
                                f"Tool calls for message {msg.id} not JSON serializable, omitting."
                            )
                            message_data["tool_calls"] = []

                    messages_output.append(message_data)
                else:
                    logger.warning(
                        f"Non-BaseMessage object found in state messages for thread {thread_id}: {type(msg)}",
                        extra={"thread_id": thread_id, "message_type": str(type(msg))},
                    )

            result = {"thread_id": thread_id, "messages": messages_output}

            if include_reasoning:
                reasoning_hist = state_snapshot.values.get("reasoning_history", [])
                try:
                    json.dumps(reasoning_hist)
                    result["reasoning_history"] = reasoning_hist
                except TypeError:
                    logger.warning(
                        f"Reasoning history for thread {thread_id} not JSON serializable, omitting."
                    )
                    result["reasoning_history"] = []

            return result

        except ValueError as ve:
            if (
                "no checkpoint found for config" in str(ve).lower()
                or "not found" in str(ve).lower()
            ):
                logger.warning(
                    f"No checkpoint found for thread_id {thread_id} during get_history.",
                    extra={"thread_id": thread_id},
                )
                return {
                    "thread_id": thread_id,
                    "messages": [],
                    "reasoning_history": [] if include_reasoning else None,
                }
            else:
                logger.error(
                    f"Value error getting history for thread_id {thread_id}: {ve}",
                    exc_info=True,
                    extra={"thread_id": thread_id},
                )
                raise
        except Exception as e:
            logger.error(
                f"Unexpected error getting history for thread_id {thread_id}: {e}",
                exc_info=True,
                extra={"thread_id": thread_id},
            )
            if "connection" in str(e).lower() and (
                "closed" in str(e).lower() or "pool" in str(e).lower()
            ):
                logger.critical(
                    "Database connection issue detected during get_state. Check DB pool and lifespan management.",
                    extra={"thread_id": thread_id},
                )
            raise

    async def list_threads(self) -> List[str]:
        if not self.db_pool:
            logger.error(
                "Database pool is not available in ChatService for list_threads."
            )
            raise RuntimeError("Database pool not configured.")
        if not self.checkpointer:
            logger.error("Checkpointer not configured for list_threads.")
            raise RuntimeError("Checkpointer not configured.")
        if not hasattr(self.checkpointer, "alist"):
            logger.warning(
                "Checkpointer doesn't support alist, falling back to direct DB query for list_threads."
            )
            return await self._list_threads_from_db()

        logger.info("Listing distinct thread IDs using checkpointer alist method...")
        thread_ids = set()
        try:
            async for config_tuple in self.checkpointer.alist(config=None):
                if (
                    config_tuple
                    and config_tuple.config
                    and "configurable" in config_tuple.config
                ):
                    thread_id = config_tuple.config["configurable"].get("thread_id")
                    if thread_id:
                        thread_ids.add(thread_id)

            final_list = sorted(list(thread_ids))
            logger.info(
                f"Found {len(final_list)} distinct thread IDs via checkpointer.",
                extra={"thread_count": len(final_list)},
            )
            return final_list
        except NotImplementedError:
            logger.warning(
                "Checkpointer alist failed (NotImplementedError), falling back to direct DB query."
            )
            return await self._list_threads_from_db()
        except Exception as exc:
            logger.error(
                f"Failed to list thread IDs using checkpointer: {exc}", exc_info=True
            )
            logger.warning(
                "Checkpointer alist failed, falling back to direct DB query."
            )
            try:
                return await self._list_threads_from_db()
            except Exception as db_exc:
                logger.error(f"Fallback DB query also failed: {db_exc}", exc_info=True)
                raise RuntimeError(
                    "Failed to retrieve thread list via checkpointer or database."
                ) from exc

    async def _list_threads_from_db(self) -> List[str]:
        if not self.db_pool:
            logger.error("Database pool is unavailable for _list_threads_from_db.")
            raise RuntimeError("Database pool not configured.")

        logger.info("Querying database directly for distinct thread IDs...")
        sql = "SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id IS NOT NULL AND thread_id != '';"
        try:
            async with self.db_pool.connection() as aconn:
                async with aconn.cursor() as acur:
                    await acur.execute(sql)
                    results = await acur.fetchall()
                    thread_ids_set = {row[0] for row in results if row[0]}

            final_list = sorted(list(thread_ids_set))
            logger.info(
                f"Found {len(final_list)} distinct thread IDs via direct DB query.",
                extra={"thread_count": len(final_list)},
            )
            return final_list
        except Exception as db_exc:
            logger.error(
                f"Direct DB query for thread IDs failed: {db_exc}", exc_info=True
            )
            raise RuntimeError(
                "Failed to retrieve thread list from database"
            ) from db_exc
