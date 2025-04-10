import logging
import json
import sys
from contextlib import (
    asynccontextmanager,
    AbstractAsyncContextManager,
    AsyncExitStack,
)
from typing import Optional, AsyncGenerator
import datetime

from fastapi import FastAPI
from psycopg_pool import AsyncConnectionPool

from app.config import settings
from app.api.routes import router
from app.services.chat import ChatService

try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
except ImportError:
    raise ImportError(
        "Critical: AsyncPostgresSaver not found. Check langgraph-checkpoint-postgres version and installation."
    ) from None

from langgraph.checkpoint.base import BaseCheckpointSaver

from langchain_mcp_adapters.client import MultiServerMCPClient


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "exc_info") and record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "stack_info") and record.stack_info:
            log_record["stack_info"] = self.formatStack(record.stack_info)
        standard_keys = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "asctime",
            "timestamp",
            "level",
            "exception",
        }
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in standard_keys:
                    log_record[key] = value
        return json.dumps(log_record, ensure_ascii=False)


log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(JsonFormatter())

root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()
root_logger.addHandler(log_handler)
root_logger.setLevel(settings.log_level)

logging.getLogger("uvicorn").propagate = False
logging.getLogger("uvicorn.access").propagate = False
logging.getLogger("uvicorn.error").propagate = False
logging.getLogger("uvicorn").setLevel(settings.log_level)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(settings.log_level)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Application startup: Initializing resources...")
    async with AsyncExitStack() as stack:
        db_pool: Optional[AsyncConnectionPool] = None
        checkpointer_cm: Optional[AbstractAsyncContextManager[BaseCheckpointSaver]] = (
            None
        )
        checkpointer: Optional[BaseCheckpointSaver] = None
        mcp_client: Optional[MultiServerMCPClient] = None
        chat_service_instance: Optional[ChatService] = None

        try:
            logger.info(
                f"Creating database connection pool for: {settings.database_url.split('@')[-1]}"
            )
            db_pool = AsyncConnectionPool(conninfo=settings.database_url, open=False)
            await stack.enter_async_context(db_pool)
            await db_pool.open(wait=True)
            logger.info("Database connection pool created and opened successfully.")
            app.state.db_pool = db_pool

            logger.info("Initializing AsyncPostgresSaver context manager...")
            checkpointer_cm = AsyncPostgresSaver.from_conn_string(settings.database_url)
            logger.info("Entering AsyncPostgresSaver context...")
            checkpointer = await stack.enter_async_context(checkpointer_cm)
            logger.info("AsyncPostgresSaver context entered.")

            try:
                logger.info("Running checkpointer setup (migrations)...")
                await checkpointer.setup()
                logger.info("Checkpointer setup complete.")
            except Exception as setup_exc:
                logger.error(f"Checkpointer setup failed: {setup_exc}", exc_info=True)
                raise RuntimeError(
                    "Failed to setup database checkpointer"
                ) from setup_exc

            mcp_servers_to_connect = settings.mcp_servers
            if mcp_servers_to_connect:
                logger.info(
                    f"Initializing MultiServerMCPClient with config: {mcp_servers_to_connect}"
                )
                try:
                    mcp_client = MultiServerMCPClient(mcp_servers_to_connect)
                    await stack.enter_async_context(mcp_client)
                    logger.info("MultiServerMCPClient initialized and context entered.")
                    app.state.mcp_client = mcp_client
                except Exception as mcp_init_exc:
                    logger.error(
                        f"Failed to initialize or enter context for MultiServerMCPClient: {mcp_init_exc}",
                        exc_info=True,
                    )
                    app.state.mcp_client = None
            else:
                logger.warning(
                    "No MCP server configuration found (MCP_SERVERS_CONFIG). Skipping MCP client initialization."
                )
                app.state.mcp_client = None

            logger.info("Initializing ChatService...")
            try:
                chat_service_instance = ChatService(
                    checkpointer=checkpointer,
                    db_pool=db_pool,
                    mcp_client=mcp_client,
                )
                logger.info("ChatService initialized successfully.")
                app.state.chat_service = chat_service_instance
            except Exception as chat_init_exc:
                logger.error(
                    f"Failed to initialize ChatService: {chat_init_exc}", exc_info=True
                )
                raise RuntimeError(
                    "Failed to initialize ChatService"
                ) from chat_init_exc

            yield

        except Exception as startup_error:
            logger.critical(
                f"Application startup failed: {startup_error}", exc_info=True
            )
            raise RuntimeError("Application startup failed") from startup_error

        logger.info(
            "Application shutdown: Cleaning up resources managed by AsyncExitStack..."
        )
    logger.info("Lifespan ended.")


app = FastAPI(title=settings.api_title, version=settings.api_version, lifespan=lifespan)

app.include_router(router, prefix="/api")


@app.get("/")
async def read_root():
    return {"message": f"{settings.api_title} is running"}


if __name__ == "__main__":
    import uvicorn

    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn"]["level"] = settings.log_level
    log_config["loggers"]["uvicorn.error"]["level"] = settings.log_level
    log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"

    logger.info(
        f"Starting Uvicorn server on host=0.0.0.0, port=8000 with log level={settings.log_level}"
    )

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_config=log_config,
    )
