from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os


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
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}?client_encoding=utf8"
        )

    def artifact_path(self, filename: str) -> Path:
        return self.backend_dir / filename


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    raw_origins = os.getenv(
        "EVENTZILLA_CORS_ORIGINS",
        "http://localhost:4200,http://127.0.0.1:4200",
    )
    origins = raw_origins.strip()
    origins = ["*"] if origins == "*" else [o.strip() for o in origins.split(",") if o.strip()]

    return Settings(
        db_user=os.getenv("EVENTZILLA_DB_USER", "postgres"),
        db_password=os.getenv("EVENTZILLA_DB_PASSWORD", "1400"),
        db_host=os.getenv("EVENTZILLA_DB_HOST", "localhost"),
        db_port=int(os.getenv("EVENTZILLA_DB_PORT", "5432")),
        db_name=os.getenv("EVENTZILLA_DB_NAME", "DW_event"),
        cors_origins=origins,
        backend_dir=BACKEND_DIR,
    )

