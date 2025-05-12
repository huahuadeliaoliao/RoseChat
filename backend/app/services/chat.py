"""Core chat service logic using LangGraph and LangChain."""

import logging
import json
import asyncio
import contextlib
from typing import (
    Dict,
    List,
    Any,
    Optional,
    AsyncGenerator,
    Tuple,
    Union,
    cast,
    AsyncIterator,
)

from app.services.custom_openai import ChatOpenAIWithReasoning
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig

from langgraph.graph import StateGraph, END
from langgraph.pregel import Pregel

from langgraph.types import StateSnapshot, StreamMode
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_mcp_adapters.client import MultiServerMCPClient

from psycopg_pool import AsyncConnectionPool
from app.config import settings
from app.models.schemas import (
    AgentState,
)

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_exception(exception_types=(Exception,), log_message: Optional[str] = None):
    """Context manager to suppress specified exceptions and log a warning."""
    try:
        yield
    except exception_types as e:
        if log_message:
            logger.warning(f"{log_message}: {str(e)} ({type(e).__name__})")
        else:
            logger.warning(f"Suppressed exception: {str(e)} ({type(e).__name__})")


class ChatService:
    """Manages chat interactions, state, and history using LangGraph."""

    def __init__(
        self,
        checkpointer: BaseCheckpointSaver,
        db_pool: AsyncConnectionPool,
        mcp_client: Optional[MultiServerMCPClient] = None,
    ):
        """Initializes the ChatService.

        Sets up the checkpointer, database connection pool, and optionally
        loads tools from an MCP client. It then initializes and compiles
        the underlying LangGraph chat agent.

        Args:
            checkpointer: The LangGraph checkpointer instance for state persistence.
            db_pool: An asynchronous PostgreSQL connection pool.
            mcp_client: An optional client for fetching external tools.
        """
        if not isinstance(checkpointer, BaseCheckpointSaver):
            raise TypeError(
                "ChatService requires a valid BaseCheckpointSaver instance."
            )
        if not isinstance(db_pool, AsyncConnectionPool):
            raise TypeError(
                "ChatService requires a valid AsyncConnectionPool instance."
            )

        self.checkpointer = checkpointer
        self.db_pool = db_pool
        self.mcp_client = mcp_client
        self.mcp_tools: List[BaseTool] = []

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

        self.chat_agent: Optional[Pregel] = self._init_chat_agent()
        if self.chat_agent:
            logger.info(
                "ChatService initialized with checkpointer and compiled chat agent."
            )
        else:
            logger.error(
                "ChatService initialization failed: Chat agent could not be compiled."
            )

    def _init_chat_agent(
        self,
    ) -> Optional[Pregel]:
        """Initialize and compile the LangGraph chat agent."""
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
                logger.info("--- Agent Node Start ---")
                messages_to_llm = state.get("messages", [])
                current_input = state.get("input_content")

                if current_input is not None:
                    messages_to_llm = messages_to_llm + [
                        HumanMessage(content=current_input)
                    ]
                    logger.info("Agent node appended new input to message history.")
                else:
                    logger.info("Agent node processing based on existing history only.")

                logger.info(f"Messages sent to LLM count: {len(messages_to_llm)}")
                if not messages_to_llm:
                    logger.warning("Agent node called with no messages.")
                    return {
                        "messages": [],
                        "input_content": None,
                    }

                try:
                    if tools_for_agent:
                        logger.debug(
                            f"Binding {len(tools_for_agent)} tools to the model."
                        )
                        model_runnable = chat_model.bind_tools(tools_for_agent)
                    else:
                        logger.debug("No tools to bind to the model.")
                        model_runnable = chat_model

                    response: BaseMessage = model_runnable.invoke(messages_to_llm)
                    logger.debug(
                        f"Agent node received LLM response type: {type(response)}"
                    )

                    messages_to_add = [response]
                    update_dict: Dict[str, Any] = {
                        "messages": messages_to_add,
                        "input_content": None,
                    }

                    reasoning_content = None
                    if isinstance(response, AIMessage) and response.additional_kwargs:
                        reasoning_content = response.additional_kwargs.get(
                            "reasoning_content"
                        )

                    if reasoning_content:
                        query_for_reasoning = (
                            current_input
                            if current_input is not None
                            else (
                                messages_to_llm[-2].content
                                if len(messages_to_llm) > 1
                                and isinstance(messages_to_llm[-2], HumanMessage)
                                else "N/A"
                            )
                        )
                        reasoning_entry = {
                            "query": str(query_for_reasoning)[:500],
                            "reasoning": str(reasoning_content),
                        }
                        update_dict["reasoning_history"] = [reasoning_entry]
                        logger.info(
                            f"Captured reasoning (length: {len(str(reasoning_content))}) for query: '{str(query_for_reasoning)[:50]}...'"
                        )

                    logger.info("--- Agent Node Finish ---")
                    return update_dict

                except Exception as node_exc:
                    logger.error(
                        f"Error invoking chat model in agent node: {node_exc}",
                        exc_info=True,
                    )
                    error_message = AIMessage(
                        content=f"Error processing request: {type(node_exc).__name__}"
                    )
                    return {"messages": [error_message], "input_content": None}

            tool_node = ToolNode(tools_for_agent) if tools_for_agent else None

            logger.info("Building StateGraph...")
            workflow = StateGraph(AgentState)
            workflow.add_node("agent", call_model_node)

            if tool_node and tools_for_agent:
                logger.info("Adding ToolNode and conditional edges for tool usage.")
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
                logger.info("No tools configured. Agent output directly goes to END.")
                workflow.add_edge("agent", END)

            workflow.set_entry_point("agent")

            logger.info("Compiling workflow with checkpointer...")
            compiled_graph: Pregel = workflow.compile(checkpointer=self.checkpointer)
            logger.info("Workflow compiled successfully.")
            return compiled_graph

        except Exception as e:
            logger.error(
                f"Fatal error during chat agent initialization: {str(e)}", exc_info=True
            )
            return None

    async def _patched_astream(
        self,
        agent: Pregel,
        input_payload: Dict[str, Any],
        config: RunnableConfig,
        stream_mode: StreamMode = "messages",
    ) -> AsyncGenerator[Any, None]:
        """Patched astream to handle cancellation more gracefully."""
        stream_iterator = agent.astream(
            input_payload,
            config,
            stream_mode=stream_mode,
        )

        current_task = asyncio.current_task()
        is_cancelled = False
        stream_iter: Optional[AsyncIterator[Any]] = None

        try:
            stream_iter = cast(AsyncIterator[Any], stream_iterator).__aiter__()

            while True:
                if current_task and current_task.cancelled():
                    is_cancelled = True
                    logger.info("Cancellation detected in _patched_astream loop.")
                    break

                try:
                    event = await asyncio.wait_for(stream_iter.__anext__(), timeout=1.0)
                    yield event
                except StopAsyncIteration:
                    logger.debug("Stream finished normally.")
                    break
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    is_cancelled = True
                    logger.info("Stream iteration explicitly cancelled.")
                    break
                except Exception as e:
                    logger.warning(
                        f"An error occurred during stream iteration: {e} ({type(e).__name__})"
                    )
                    if isinstance(e, ConnectionError):
                        is_cancelled = True
                        logger.error(f"Connection error during stream: {e}")
                    yield f"event: error\ndata: {json.dumps({'error': f'Stream iteration error: {type(e).__name__}'}, ensure_ascii=False)}\n\n"
                    break

        except asyncio.CancelledError:
            is_cancelled = True
            logger.info("Stream generator task cancelled.")
        except Exception as e:
            logger.error(
                f"An unhandled exception occurred during stream processing setup: {e}",
                exc_info=True,
            )
            yield f"event: error\ndata: {json.dumps({'error': f'Stream setup error: {type(e).__name__}'}, ensure_ascii=False)}\n\n"
        finally:
            if is_cancelled:
                logger.info("Performing cleanup after stream cancellation.")
                if hasattr(stream_iterator, "_callbacks"):
                    with suppress_exception(
                        log_message="Error cleaning up stream callbacks during cancellation"
                    ):
                        pass
                if hasattr(stream_iterator, "close"):
                    with suppress_exception(
                        log_message="Error closing stream iterator during cancellation"
                    ):
                        actual_close_method = getattr(stream_iterator, "close")
                        if asyncio.iscoroutinefunction(actual_close_method):
                            await actual_close_method()

            stream_iterator = None
            stream_iter = None
            logger.debug("_patched_astream finished.")

    async def stream_message(
        self,
        content: str,
        thread_id: Optional[str] = None,
        include_reasoning: bool = False,
    ) -> AsyncGenerator[str, None]:
        r"""Processes a user message and streams the agent's response.

        Invokes the chat agent for the given thread ID with the new message
        content. Streams back the agent's thoughts (reasoning, if requested)
        and final response chunks as Server-Sent Events (SSE).

        Args:
            content: The user's message content.
            thread_id: The identifier of the conversation thread. Must be provided.
            include_reasoning: If True, include reasoning steps in the stream.

        Yields:
            str: Server-Sent Event formatted strings containing response chunks,
                 reasoning steps, tool results, status updates (end/cancelled),
                 or errors.
                 Example format: "event: <type>\\ndata: <json_payload>\\n\\n"
        """
        if not self.chat_agent:
            logger.error("Chat agent is not initialized. Cannot stream message.")
            yield f"event: error\ndata: {json.dumps({'error': 'Chat service not available'}, ensure_ascii=False)}\n\n"
            return

        if not thread_id:
            logger.error("stream_message called without a thread_id.")
            yield f"event: error\ndata: {json.dumps({'error': 'Missing thread ID'}, ensure_ascii=False)}\n\n"
            return

        run_config = RunnableConfig(configurable={"thread_id": thread_id})
        input_payload = {"input_content": content}

        logger.info(
            f"Streaming chat agent response for thread_id: {thread_id}",
            extra={"thread_id": thread_id, "input_content_preview": content[:100]},
        )

        current_reasoning_chunk = ""

        try:
            async for event in self._patched_astream(
                self.chat_agent,
                input_payload,
                run_config,
                stream_mode="messages",
            ):
                current_task = asyncio.current_task()
                if current_task and current_task.cancelled():
                    logger.info(
                        f"Streaming task cancelled detected within loop for thread_id: {thread_id}",
                        extra={"thread_id": thread_id},
                    )
                    break

                if isinstance(event, BaseMessage):
                    chunk = event
                    if isinstance(chunk, AIMessageChunk):
                        if include_reasoning and chunk.additional_kwargs:
                            new_reasoning = chunk.additional_kwargs.get(
                                "reasoning_content"
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
                        tool_info = {
                            "type": "tool_result",
                            "tool_call_id": chunk.tool_call_id,
                            "name": chunk.name
                            if hasattr(chunk, "name")
                            else "unknown_tool",
                            "content": str(chunk.content)[:200]
                            + ("..." if len(str(chunk.content)) > 200 else ""),
                        }
                        yield f"data: {json.dumps(tool_info, ensure_ascii=False)}\n\n"

                elif isinstance(event, str) and event.startswith("event: error"):
                    yield event

            current_task = asyncio.current_task()
            if not (current_task and current_task.cancelled()):
                try:
                    logger.debug(
                        f"Attempting to get final state for thread: {thread_id}"
                    )
                    if not self.chat_agent:
                        raise RuntimeError(
                            "Chat agent became unavailable unexpectedly."
                        )

                    final_state_snapshot: Optional[
                        StateSnapshot
                    ] = await asyncio.wait_for(
                        cast(Pregel, self.chat_agent).aget_state(run_config),
                        timeout=5.0,
                    )

                    final_thread_id = thread_id
                    config_from_state: Optional[RunnableConfig] = getattr(
                        final_state_snapshot, "config", None
                    )
                    if config_from_state:
                        configurable_map = config_from_state.get("configurable")
                        retrieved_thread_id = None
                        if configurable_map:
                            retrieved_thread_id = configurable_map.get("thread_id")

                        if retrieved_thread_id:
                            final_thread_id = retrieved_thread_id
                        else:
                            logger.warning(
                                f"Could not determine final thread_id from state config for {thread_id}, using input ID."
                            )
                    else:
                        logger.warning(
                            f"Could not retrieve final state or config for {thread_id}, using input ID."
                        )
                    logger.info(
                        f"Streaming finished successfully for thread_id: {final_thread_id}",
                        extra={"thread_id": final_thread_id},
                    )
                    yield f"event: end\ndata: {json.dumps({'thread_id': final_thread_id}, ensure_ascii=False)}\n\n"

                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timed out waiting for final state for thread_id: {thread_id}"
                    )
                    yield f"event: end\ndata: {json.dumps({'thread_id': thread_id, 'warning': 'Final state retrieval timed out'}, ensure_ascii=False)}\n\n"
                except asyncio.CancelledError:
                    logger.info(
                        f"Task cancelled while getting final state for {thread_id}."
                    )
                    yield f"event: cancelled\ndata: {json.dumps({'thread_id': thread_id}, ensure_ascii=False)}\n\n"
                except Exception as final_state_exc:
                    logger.error(
                        f"Error getting final state for {thread_id}: {final_state_exc}",
                        exc_info=True,
                    )
                    yield f"event: end\ndata: {json.dumps({'thread_id': thread_id, 'error': 'Failed to get final state'}, ensure_ascii=False)}\n\n"
            else:
                logger.info(
                    f"Stream cancelled before final state retrieval for thread_id: {thread_id}",
                    extra={"thread_id": thread_id},
                )
                yield f"event: cancelled\ndata: {json.dumps({'thread_id': thread_id}, ensure_ascii=False)}\n\n"

        except asyncio.CancelledError:
            logger.info(
                f"Stream task explicitly cancelled for thread_id: {thread_id}",
                extra={"thread_id": thread_id},
            )
            yield f"event: cancelled\ndata: {json.dumps({'thread_id': thread_id}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(
                f"Unhandled error during message streaming for thread_id {thread_id}: {e}",
                exc_info=True,
                extra={"thread_id": thread_id},
            )
            error_message = f"An unexpected error occurred: {type(e).__name__}"
            yield f"event: error\ndata: {json.dumps({'error': error_message}, ensure_ascii=False)}\n\n"
        finally:
            logger.debug(
                f"stream_message generator finishing for thread_id: {thread_id}"
            )

    async def get_history(
        self, thread_id: str, include_reasoning: bool = False
    ) -> Dict[str, Any]:
        """Retrieves the message history for a given conversation thread.

        Fetches the state snapshot for the specified thread_id using the
        checkpointer and extracts the message history. Optionally includes
        the reasoning history if requested and available.

        Args:
            thread_id: The identifier of the conversation thread.
            include_reasoning: If True, include the reasoning history in the result.

        Returns:
            Dict[str, Any]: A dictionary containing the 'thread_id', a list of
                            'messages' (each a dict with 'role', 'content', etc.),
                            and optionally 'reasoning_history' (a list of dicts
                            with 'query' and 'reasoning'). Returns empty lists if
                            the thread is not found.

        Raises:
            RuntimeError: If the chat agent or checkpointer is not initialized,
                          or if a fatal error occurs during state retrieval.
            ValueError: If the provided thread_id is invalid.
        """
        if not self.chat_agent:
            logger.error(
                "Chat agent not initialized. Cannot get history.",
                extra={"thread_id": thread_id},
            )
            raise RuntimeError("ChatService's chat_agent is not initialized.")
        if not self.checkpointer:
            logger.error(
                "Checkpointer not available. Cannot get history.",
                extra={"thread_id": thread_id},
            )
            raise RuntimeError("ChatService's checkpointer is not available.")
        if not thread_id or not isinstance(thread_id, str):
            logger.error(f"Invalid thread_id provided for get_history: {thread_id}")
            raise ValueError("A valid thread_id string must be provided.")

        run_config = RunnableConfig(configurable={"thread_id": thread_id})
        logger.info(
            f"Getting state snapshot for thread_id: {thread_id}",
            extra={"thread_id": thread_id},
        )

        try:
            state_snapshot: Optional[StateSnapshot] = await cast(
                Pregel, self.chat_agent
            ).aget_state(run_config)

            if not state_snapshot or not state_snapshot.values:
                logger.warning(
                    f"No state found for thread_id: {thread_id}. Returning empty history.",
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
            snapshot_values = state_snapshot.values
            state_messages = snapshot_values.get("messages", [])

            if isinstance(state_messages, list):
                for msg in state_messages:
                    if isinstance(msg, BaseMessage):
                        role = "system"
                        if isinstance(msg, HumanMessage):
                            role = "user"
                        elif isinstance(msg, AIMessage):
                            role = "assistant"
                        elif isinstance(msg, ToolMessage):
                            role = "tool"

                        message_data: Dict[str, Any] = {
                            "role": role,
                            "content": str(msg.content),
                        }

                        if isinstance(msg, ToolMessage) and hasattr(
                            msg, "tool_call_id"
                        ):
                            message_data["tool_call_id"] = msg.tool_call_id
                            if hasattr(msg, "name"):
                                message_data["name"] = msg.name

                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            try:
                                json.dumps(msg.tool_calls)
                                message_data["tool_calls"] = msg.tool_calls
                            except TypeError:
                                logger.warning(
                                    f"Tool calls for message in thread {thread_id} not JSON serializable, omitting.",
                                    extra={"thread_id": thread_id},
                                )
                                message_data["tool_calls"] = []

                        messages_output.append(message_data)
                    else:
                        logger.warning(
                            f"Non-BaseMessage object found in state messages for thread {thread_id}: {type(msg)}",
                            extra={
                                "thread_id": thread_id,
                                "message_type": str(type(msg)),
                            },
                        )
            else:
                logger.warning(
                    f"Expected 'messages' to be a list in state for thread {thread_id}, but got {type(state_messages)}"
                )

            result: Dict[str, Any] = {
                "thread_id": thread_id,
                "messages": messages_output,
            }

            if include_reasoning:
                reasoning_hist = snapshot_values.get("reasoning_history", [])
                try:
                    if reasoning_hist is None:
                        reasoning_hist = []
                    elif not isinstance(reasoning_hist, list):
                        logger.warning(
                            f"Expected 'reasoning_history' to be a list in state for {thread_id}, got {type(reasoning_hist)}. Setting to empty list."
                        )
                        reasoning_hist = []

                    json.dumps(reasoning_hist)
                    result["reasoning_history"] = reasoning_hist
                except TypeError:
                    logger.warning(
                        f"Reasoning history for thread {thread_id} not JSON serializable, omitting.",
                        extra={"thread_id": thread_id},
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
            raise RuntimeError(f"Failed to get history for thread {thread_id}") from e

    async def list_threads(self) -> List[str]:
        """Lists all unique conversation thread IDs.

        Attempts to retrieve a list of distinct thread IDs primarily using the
        checkpointer's `alist` method if available. If that fails or is not
        implemented, it falls back to querying the underlying database directly.

        Returns:
            List[str]: A sorted list of unique thread ID strings.

        Raises:
            RuntimeError: If the checkpointer is not configured or if listing
                          threads fails via both the checkpointer and the database.
        """
        if not self.checkpointer:
            logger.error("Checkpointer not configured. Cannot list threads.")
            raise RuntimeError("Checkpointer not configured.")

        if hasattr(self.checkpointer, "alist") and callable(self.checkpointer.alist):
            logger.info(
                "Listing distinct thread IDs using checkpointer alist method..."
            )
            thread_ids = set()
            try:
                list_config: Optional[RunnableConfig] = None
                async for config_tuple in self.checkpointer.alist(
                    config=list_config, limit=None
                ):
                    current_config = config_tuple.config
                    if current_config:
                        configurable_map = current_config.get("configurable")
                        if configurable_map:
                            thread_id_val = configurable_map.get("thread_id")
                            if thread_id_val and isinstance(thread_id_val, str):
                                thread_ids.add(thread_id_val)

                final_list = sorted(list(thread_ids))
                logger.info(
                    f"Found {len(final_list)} distinct thread IDs via checkpointer.",
                    extra={"thread_count": len(final_list)},
                )
                return final_list
            except NotImplementedError:
                logger.warning(
                    "Checkpointer's alist method is not implemented. Falling back to direct DB query."
                )
            except Exception as exc:
                logger.error(
                    f"Failed to list thread IDs using checkpointer alist: {exc}",
                    exc_info=True,
                )
                logger.warning(
                    "Checkpointer alist failed. Falling back to direct DB query."
                )
        else:
            logger.warning(
                "Checkpointer does not have a callable 'alist' method. Falling back to direct DB query."
            )

        try:
            return await self._list_threads_from_db()
        except Exception as db_exc:
            logger.error(
                f"Fallback DB query for listing threads also failed: {db_exc}",
                exc_info=True,
            )
            raise RuntimeError(
                "Failed to retrieve thread list via checkpointer or database."
            ) from db_exc

    async def _list_threads_from_db(self) -> List[str]:
        """Helper method to list thread IDs directly from the database."""
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
                    thread_ids_set = {str(row[0]) for row in results if row[0]}

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
