from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routers import core, lab


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="EventZilla API",
        version="1.0.0",
        description="FastAPI backend for the EventZilla Angular frontend.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(core.router)
    app.include_router(lab.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app

