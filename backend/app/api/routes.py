"""API routes definition."""

import asyncio
from fastapi import APIRouter, HTTPException, Request, Query, status
from app.config import settings
from fastapi.responses import StreamingResponse
from typing import List, AsyncGenerator
from uuid import uuid4
import logging
import json
from starlette.background import BackgroundTask

from app.models.schemas import (
    MessageRequest,
    ChatHistoryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat")
async def chat_stream(
    request_body: MessageRequest, request: Request
) -> StreamingResponse:
    """Handle chat requests and stream responses."""
    try:
        chat_service = getattr(request.app.state, "chat_service", None)
        if not chat_service:
            logger.error("Chat service not found in application state.")
            raise HTTPException(status_code=503, detail="Chat service is not available")

        thread_id_input = (
            request_body.thread_id
            if (request_body.thread_id and request_body.thread_id.strip())
            else None
        )
        thread_id_to_use = thread_id_input or str(uuid4())
        logger.info(f"Initiating stream for thread_id: {thread_id_to_use}")

        disconnect_event = asyncio.Event()
        stream_task = None

        async def stream_generator() -> AsyncGenerator[str, None]:
            """Generate SSE chunks for the chat stream."""
            nonlocal stream_task

            try:
                stream_gen = chat_service.stream_message(
                    content=request_body.content,
                    thread_id=thread_id_to_use,
                    include_reasoning=request_body.include_reasoning,
                )

                async for chunk in stream_gen:
                    if disconnect_event.is_set():
                        logger.info(
                            f"Detected disconnect, breaking stream for thread_id: {thread_id_to_use}"
                        )
                        break

                    yield chunk

            except asyncio.CancelledError:
                logger.info(
                    f"Stream generator cancelled for thread_id: {thread_id_to_use}"
                )
            except Exception as e:
                logger.error(
                    f"Error during response streaming for thread {thread_id_to_use}: {e}",
                    exc_info=True,
                )
                error_payload = json.dumps(
                    {"error": f"Stream failed: {type(e).__name__}"}
                )
                yield f"event: error\ndata: {error_payload}\n\n"
            finally:
                logger.info(
                    f"Stream generator completed for thread_id: {thread_id_to_use}"
                )

        async def on_disconnect():
            """Handle client disconnect event."""
            logger.info(
                f"Client disconnected from stream for thread_id: {thread_id_to_use}"
            )
            disconnect_event.set()

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            background=BackgroundTask(on_disconnect),
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except ValueError as ve:
        logger.warning(f"Value error setting up chat stream: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception:
        logger.exception("Error setting up chat stream endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error setting up stream",
        )


@router.get("/history/{thread_id}", response_model=ChatHistoryResponse)
async def get_history(
    thread_id: str,
    request: Request,
    include_reasoning: bool = Query(
        False, description="Include model reasoning history"
    ),
):
    """Retrieve chat history for a given thread ID."""
    try:
        chat_service = getattr(request.app.state, "chat_service", None)
        if not chat_service:
            logger.error("Chat service not found in application state.")
            raise HTTPException(status_code=503, detail="Chat service is not available")

        history = await chat_service.get_history(thread_id, include_reasoning)

        if (
            not isinstance(history, dict)
            or "thread_id" not in history
            or "messages" not in history
        ):
            logger.error(
                f"Invalid history format from chat_service for thread {thread_id}: {history}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error: Invalid history format",
            )

        return history

    except ValueError as ve:
        logger.warning(f"Value error during get_history for thread {thread_id}: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except HTTPException as http_exc:
        raise http_exc
    except Exception:
        logger.exception(f"Error getting history for thread {thread_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/threads", response_model=List[str])
async def list_threads_endpoint(request: Request):
    """List all available chat thread IDs."""
    try:
        chat_service = getattr(request.app.state, "chat_service", None)
        if not chat_service:
            logger.error("Chat service not found in application state.")
            raise HTTPException(status_code=503, detail="Chat service is not available")

        threads = await chat_service.list_threads()
        return threads

    except Exception:
        logger.exception("Error in list_threads endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/current_model")
async def get_current_model():
    """Get the name of the currently configured chat model."""
    try:
        model_name = settings.model_name
        if not model_name:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model name not configured",
            )
        return {"model_name": model_name}
    except Exception as e:
        logger.error(f"Error retrieving current model name: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error retrieving model name",
        )


@router.get("/health")
async def health_check(request: Request):
    """Perform a health check of the service."""
    current_status = "healthy"
    chat_service = getattr(request.app.state, "chat_service", None)
    db_pool = getattr(request.app.state, "db_pool", None)

    if not chat_service:
        current_status = "degraded"
        logger.warning("Health check: Chat service not found in app state.")
    if not db_pool or db_pool.closed:
        current_status = "degraded"
        logger.warning(
            f"Health check: DB pool not found or closed (closed={db_pool.closed if db_pool else 'N/A'})."
        )

    return {"status": current_status}
