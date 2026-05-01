from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from .config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_size=2,
        max_overflow=0,
        pool_pre_ping=True,
        pool_recycle=300,
    )
