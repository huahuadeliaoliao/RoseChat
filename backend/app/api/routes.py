from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from typing import List, AsyncGenerator
from uuid import uuid4
import logging
import json

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

        async def stream_generator() -> AsyncGenerator[str, None]:
            try:
                async for chunk in chat_service.stream_message(
                    content=request_body.content,
                    thread_id=thread_id_to_use,
                    include_reasoning=request_body.include_reasoning,
                ):
                    yield chunk
            except Exception as e:
                logger.error(
                    f"Error during response streaming for thread {thread_id_to_use}: {e}",
                    exc_info=True,
                )
                error_payload = json.dumps(
                    {"error": f"Stream failed: {type(e).__name__}"}
                )
                yield f"event: error\ndata: {error_payload}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    except ValueError as ve:
        logger.warning(f"Value error setting up chat stream: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        logger.exception("Error setting up chat stream endpoint")
        raise HTTPException(
            status_code=500, detail="Internal server error setting up stream"
        )


@router.get("/history/{thread_id}", response_model=ChatHistoryResponse)
async def get_history(
    thread_id: str,
    request: Request,
    include_reasoning: bool = Query(
        False, description="Include model reasoning history"
    ),
):
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
                status_code=500, detail="Internal server error: Invalid history format"
            )

        return history

    except ValueError as ve:
        logger.warning(f"Value error during get_history for thread {thread_id}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        logger.exception(f"Error getting history for thread {thread_id}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/threads", response_model=List[str])
async def list_threads_endpoint(request: Request):
    try:
        chat_service = getattr(request.app.state, "chat_service", None)
        if not chat_service:
            logger.error("Chat service not found in application state.")
            raise HTTPException(status_code=503, detail="Chat service is not available")

        threads = await chat_service.list_threads()
        return threads

    except Exception:
        logger.exception("Error in list_threads endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def health_check(request: Request):
    status = "healthy"
    chat_service = getattr(request.app.state, "chat_service", None)
    db_pool = getattr(request.app.state, "db_pool", None)

    if not chat_service:
        status = "degraded"
        logger.warning("Health check: Chat service not found in app state.")
    if not db_pool or db_pool.closed:
        status = "degraded"
        logger.warning(
            f"Health check: DB pool not found or closed (closed={db_pool.closed if db_pool else 'N/A'})."
        )

    return {"status": status}
