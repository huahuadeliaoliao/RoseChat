import logging
import json
import sys
from contextlib import (
    asynccontextmanager,
    AbstractAsyncContextManager,
)
from typing import Optional, AsyncGenerator
import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_record["stack_info"] = self.formatStack(record.stack_info)
        # Add any extra fields
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in log_record and key not in [
                    "args",
                    "asctime",
                    "created",
                    "exc_info",
                    "exc_text",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "message",
                    "msg",
                    "name",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "stack_info",
                    "thread",
                    "threadName",
                ]:
                    log_record[key] = value
        return json.dumps(log_record, ensure_ascii=False)


log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(JsonFormatter())

root_logger = logging.getLogger()
root_logger.addHandler(log_handler)
root_logger.setLevel(settings.log_level)

logging.getLogger("uvicorn").propagate = False
logging.getLogger("uvicorn.access").propagate = False
logging.getLogger("uvicorn.error").propagate = False

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Application startup: Initializing resources...")
    checkpointer_cm: Optional[AbstractAsyncContextManager[BaseCheckpointSaver]] = None
    checkpointer: Optional[BaseCheckpointSaver] = None
    chat_service_instance: Optional[ChatService] = None
    db_pool: Optional[AsyncConnectionPool] = None

    try:
        logger.info(f"Creating database connection pool for: {settings.database_url}")
        db_pool = AsyncConnectionPool(conninfo=settings.database_url, open=False)
        await db_pool.open(wait=True)
        logger.info("Database connection pool created successfully.")
        app.state.db_pool = db_pool

        logger.info("Initializing AsyncPostgresSaver context manager...")
        checkpointer_cm = AsyncPostgresSaver.from_conn_string(settings.database_url)
        logger.info("Entering AsyncPostgresSaver context...")
        checkpointer = await checkpointer_cm.__aenter__()
        logger.info("AsyncPostgresSaver context entered.")

        try:
            logger.info("Running checkpointer setup...")
            await checkpointer.setup()
            logger.info("Checkpointer setup complete.")
        except Exception as setup_exc:
            logger.error(f"Checkpointer setup failed: {setup_exc}", exc_info=True)
            if checkpointer_cm:
                try:
                    await checkpointer_cm.__aexit__(
                        type(setup_exc), setup_exc, setup_exc.__traceback__
                    )
                except Exception as exit_exc:
                    logger.error(
                        f"Error exiting async checkpointer context after setup failure: {exit_exc}",
                        exc_info=True,
                    )
            if db_pool:
                await db_pool.close()
            raise RuntimeError("Failed to setup database checkpointer") from setup_exc

        logger.info("Initializing ChatService...")
        chat_service_instance = ChatService(checkpointer=checkpointer, db_pool=db_pool)
        logger.info("ChatService initialized successfully.")

        app.state.chat_service = chat_service_instance
        app.state.checkpointer_cm = checkpointer_cm

        yield

    except Exception as startup_error:
        logger.critical(f"Application startup failed: {startup_error}", exc_info=True)
        if checkpointer_cm and checkpointer:
            try:
                logger.warning(
                    "Exiting async checkpointer context due to startup failure..."
                )
                await checkpointer_cm.__aexit__(
                    type(startup_error), startup_error, startup_error.__traceback__
                )
            except Exception as exit_fail:
                logger.error(
                    f"Failed to exit async checkpointer context during startup failure handling: {exit_fail}",
                    exc_info=True,
                )
        if db_pool:
            logger.warning("Closing database pool due to startup failure...")
            await db_pool.close()
        raise RuntimeError("Application startup failed") from startup_error

    finally:
        logger.info("Application shutdown: Cleaning up resources...")
        cm_to_exit = getattr(app.state, "checkpointer_cm", None)
        if cm_to_exit:
            logger.info("Exiting AsyncPostgresSaver context...")
            try:
                await cm_to_exit.__aexit__(None, None, None)
                logger.info("AsyncPostgresSaver context exited successfully.")
            except Exception as e:
                logger.error(
                    f"Error exiting AsyncPostgresSaver context during shutdown: {e}",
                    exc_info=True,
                )
        else:
            logger.warning(
                "Async Checkpointer context manager not found in app state for shutdown."
            )

        pool_to_close = getattr(app.state, "db_pool", None)
        if pool_to_close:
            logger.info("Closing database connection pool...")
            await pool_to_close.close()
            logger.info("Database connection pool closed.")
        else:
            logger.warning("Database pool not found in app state for shutdown.")


app = FastAPI(title=settings.api_title, version=settings.api_version, lifespan=lifespan)

app.include_router(router, prefix="/api")


if __name__ == "__main__":
    import uvicorn

    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"][""]["level"] = settings.log_level
    log_config["loggers"]["uvicorn"]["level"] = settings.log_level
    log_config["loggers"]["uvicorn.access"]["level"] = settings.log_level
    log_config["loggers"]["uvicorn.error"]["level"] = settings.log_level

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_config=log_config,
    )
