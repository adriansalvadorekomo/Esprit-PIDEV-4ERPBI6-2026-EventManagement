"""Single source of truth for all backend configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

DB_USER     = os.getenv("EVENTZELLA_DB_USER",     "postgres")
DB_PASSWORD = os.getenv("EVENTZELLA_DB_PASSWORD", "1400")
DB_HOST     = os.getenv("EVENTZELLA_DB_HOST",     "localhost")
DB_PORT     = os.getenv("EVENTZELLA_DB_PORT",     "5432")
DB_NAME     = os.getenv("EVENTZELLA_DB_NAME",     "DW_event")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}?client_encoding=utf8"
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
