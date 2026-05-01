from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys, os

# Ensure backend root is on path so `settings` is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import settings as _s

BACKEND_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    db_user: str
    db_password: str
    db_host: str
    db_port: int
    db_name: str
    cors_origins: list[str]
    backend_dir: Path

    @property
    def database_url(self) -> str:
        return _s.DATABASE_URL

    def artifact_path(self, filename: str) -> Path:
        return self.backend_dir / filename


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    raw = os.getenv("EVENTZILLA_CORS_ORIGINS", "http://localhost:4200,http://127.0.0.1:4200").strip()
    origins = ["*"] if raw == "*" else [o.strip() for o in raw.split(",") if o.strip()]
    return Settings(
        db_user=_s.DB_USER,
        db_password=_s.DB_PASSWORD,
        db_host=_s.DB_HOST,
        db_port=int(_s.DB_PORT),
        db_name=_s.DB_NAME,
        cors_origins=origins,
        backend_dir=BACKEND_DIR,
    )
