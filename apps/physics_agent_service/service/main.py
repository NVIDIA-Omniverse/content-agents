# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physics Agent FastAPI Service - Main Application."""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from world_understanding.utils.logging import setup_logging

from .utils import AccessLogFilter

# Add parent directories to Python path for physics_agent imports
service_dir = Path(__file__).parent.parent
apps_dir = service_dir.parent
repo_root = apps_dir.parent

for path in [str(apps_dir), str(repo_root)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import io  # noqa: E402
import os  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

# Load .env file BEFORE importing config
load_dotenv()

from .config import config  # noqa: E402
from .routers import (  # noqa: E402
    artifacts_router,
    pipeline_router,
    predict_router,
    sessions_router,
    tune_router,
)
from .session.manager import SessionManager  # noqa: E402

# Load environment variables
load_dotenv()

# Setup logging from config
setup_logging()

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def _get_max_active_sessions() -> int:
    """Parse PA_MAX_ACTIVE_SESSIONS from environment with safe fallback."""
    default_limit = 8
    env_value = os.getenv("PA_MAX_ACTIVE_SESSIONS")

    if env_value is None:
        return default_limit

    try:
        limit = int(env_value)
        if limit < 0:
            logger.error(
                "PA_MAX_ACTIVE_SESSIONS must be non-negative, got '%s'. "
                "Falling back to default: %d",
                env_value,
                default_limit,
            )
            return default_limit
        return limit
    except ValueError:
        logger.error(
            "PA_MAX_ACTIVE_SESSIONS must be a valid integer, got '%s'. "
            "Falling back to default: %d",
            env_value,
            default_limit,
        )
        return default_limit


def _load_aws_config_file_into_env(*, log: logging.Logger) -> None:
    """Expand AWS_CONFIG_FILE contents into process environment variables."""
    from dotenv import dotenv_values

    config_path_raw = os.getenv("AWS_CONFIG_FILE")
    if not config_path_raw:
        return

    config_path = Path(config_path_raw).expanduser()

    if not config_path.exists():
        log.warning("AWS_CONFIG_FILE set but file does not exist: %s", config_path)
        return

    try:
        values = dotenv_values(config_path)
    except Exception:
        log.exception("Failed reading AWS_CONFIG_FILE: %s", config_path)
        return

    def _set_env_if_missing(*, env_key: str, value: str | None) -> bool:
        if not value:
            return False
        if os.getenv(env_key):
            return False
        os.environ[env_key] = value
        return True

    access_key_id = values.get("AWS_ACCESS_KEY_ID") or values.get("aws_access_key_id")
    secret_access_key = values.get("AWS_SECRET_ACCESS_KEY") or values.get(
        "aws_secret_access_key"
    )
    region = (
        values.get("AWS_DEFAULT_REGION")
        or values.get("aws_default_region")
        or values.get("AWS_REGION")
        or values.get("aws_region")
        or values.get("region")
    )

    any_set = False
    any_set = (
        _set_env_if_missing(env_key="AWS_ACCESS_KEY_ID", value=access_key_id) or any_set
    )
    any_set = (
        _set_env_if_missing(env_key="AWS_SECRET_ACCESS_KEY", value=secret_access_key)
        or any_set
    )
    if _set_env_if_missing(env_key="AWS_DEFAULT_REGION", value=region):
        any_set = True
        _set_env_if_missing(env_key="AWS_REGION", value=region)

    if any_set:
        log.info("AWS_CONFIG_FILE loaded.")

    if "AWS_ACCESS_KEY_ID" in os.environ and os.environ["AWS_ACCESS_KEY_ID"]:
        log.info("AWS Access Key ID: present")
    if "AWS_SECRET_ACCESS_KEY" in os.environ and os.environ["AWS_SECRET_ACCESS_KEY"]:
        log.info("AWS Secret Access Key: present")
    region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION")
    if region:
        log.info("AWS Region: %s", region)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    logger.info("Starting Physics Agent Service...")
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.addFilter(AccessLogFilter())
    active_render_backend = os.getenv("PA_RENDER_BACKEND", "remote")

    # Check for required API keys for the active backend/render settings
    if not config.has_required_api_keys:
        logger.warning(
            "Required API keys are not configured for the current backend or "
            "render settings. VLM=%s PA_RENDER_BACKEND=%s",
            config.vlm_backend,
            active_render_backend,
        )

    # Expand AWS_CONFIG_FILE into AWS* env vars
    _load_aws_config_file_into_env(log=logger)

    # Initialize session store and manager
    store = config.build_session_store()
    logger.info(
        "Initializing session storage: kind=%s, local_path=%s",
        store.kind,
        config.session_storage_path,
    )
    if store.kind == "s3":
        logger.info(
            "S3 storage: bucket=%s, prefix=%s",
            config.storage_s3_bucket,
            config.storage_s3_prefix,
        )
    session_mgr = SessionManager(
        storage_path=config.session_storage_path,
        ttl_hours=config.session_ttl_hours,
        store=store,
    )

    # Set global session manager in all routers
    pipeline_router.set_session_manager(session_mgr)
    predict_router.set_session_manager(session_mgr)
    artifacts_router.set_session_manager(session_mgr)
    sessions_router.set_session_manager(session_mgr)
    tune_router.set_session_manager(session_mgr)

    # Only the local warp backend needs prewarming in the main process.
    if active_render_backend.lower() == "warp":
        try:
            import warp as wp

            wp.init()
            logger.info(
                "Warp initialised: %s (devices: %d)",
                wp.__version__,
                wp.get_cuda_device_count(),
            )
            # Pre-import newton raytrace so kernel compilation happens now
            # (main thread) rather than inside asyncio.to_thread where
            # subprocess spawning can deadlock.
            try:
                from newton._src.sensors.warp_raytrace import (  # noqa: F401
                    RenderContext,
                )

                logger.info("Newton warp_raytrace loaded — kernels will be cached")
            except ImportError:
                logger.info("Newton not installed — Warp rendering backend unavailable")
        except ImportError:
            logger.info("warp-lang not installed — Warp rendering backend unavailable")
        except Exception as e:
            logger.warning("Warp initialisation failed: %s", e)
    else:
        logger.info(
            "Skipping local warp initialisation for render backend: %s",
            active_render_backend,
        )

    logger.info("Service started: %s v%s", config.service_name, config.service_version)
    logger.info(
        f"VLM: {config.vlm_backend} / {config.vlm_model} "
        f"(temp={config.vlm_temperature})"
    )
    provider_status = "configured" if config.has_required_api_keys else "NOT SET"
    logger.info("Active provider credentials: %s", provider_status)

    yield

    # Shutdown
    logger.info("Shutting down Physics Agent Service...")


# Create FastAPI app
app = FastAPI(
    title=config.service_name,
    description=config.description,
    version=config.service_version,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# SessionManager raises InvalidSessionIdError (ValueError subclass) for
# malformed session_id path params (e.g. `../`, non-UUID); translate to 400
# so malicious traffic doesn't hit the default 500 handler. We target the
# subclass (not all ValueError) so unrelated ValueError bugs still surface as
# 500 — otherwise pydantic / type-conversion errors would silently map to 400.
from .session.manager import InvalidSessionIdError  # noqa: E402


@app.exception_handler(InvalidSessionIdError)
async def _invalid_session_id_handler(
    request: Request, exc: InvalidSessionIdError
) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# Include routers
app.include_router(pipeline_router.router)
app.include_router(predict_router.router)
app.include_router(artifacts_router.router)
app.include_router(sessions_router.router)
app.include_router(tune_router.router)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": config.service_name,
        "version": config.service_version,
        "api_keys_configured": config.has_required_api_keys,
        "max_active_sessions": _get_max_active_sessions(),
    }


@app.get("/api")
async def root_api_info():
    """Root endpoint with service info."""
    return {
        "service": config.service_name,
        "version": config.service_version,
        "docs": "/docs",
        "health": "/health",
        "api": {
            "pipeline": {
                "create": "POST /pipeline",
                "status": "GET /pipeline/{session_id}/status",
                "results": "GET /pipeline/{session_id}/results",
                "cancel": "POST /pipeline/{session_id}/cancel",
                "events": "GET /pipeline/{session_id}/events",
                "regenerate": "POST /pipeline/{session_id}/regenerate",
            },
            "predict": {
                "create": "POST /predict",
                "status": "GET /predict/{session_id}/status",
                "results": "GET /predict/{session_id}/results",
                "cancel": "POST /predict/{session_id}/cancel",
                "events": "GET /predict/{session_id}/events",
            },
            "artifacts": {
                "predictions": "GET /artifacts/{session_id}/predictions",
                "report": "GET /artifacts/{session_id}/report",
                "dataset": "GET /artifacts/{session_id}/dataset",
            },
            "sessions": {
                "list": "GET /sessions",
                "get": "GET /sessions/{session_id}",
                "delete": "DELETE /sessions/{session_id}",
            },
            "tune": {
                "create": "POST /tune",
                "status": "GET /tune/{session_id}/status",
                "results": "GET /tune/{session_id}/results",
                "events": "GET /tune/{session_id}/events",
                "cancel": "POST /tune/{session_id}/cancel",
                "artifact": "GET /tune/{session_id}/artifacts/{name}",
            },
        },
    }


@app.get("/")
async def root():
    """Root endpoint redirects to API info."""
    return await root_api_info()


def main():
    """Entry point for running the service."""
    import uvicorn

    uvicorn.run(
        "service.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
