"""
Application configuration loaded from environment variables.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # --- Database ---
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/delhicommutebot"

    # --- Twilio ---
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_number: str = "whatsapp:+14155238886"

    # --- Optional LLM ---
    openai_api_key: str = ""
    google_api_key: str = ""

    # --- Model / Data Paths ---
    intent_model_path: str = str(BASE_DIR / "models" / "intent_classifier")
    faiss_index_path: str = str(BASE_DIR / "data" / "indices")
    gtfs_data_path: str = str(BASE_DIR / "data" / "gtfs")
    sample_data_path: str = str(BASE_DIR / "data" / "sample")
    metro_data_path: str = str(BASE_DIR / "data" / "metro")
    auto_data_path: str = str(BASE_DIR / "data" / "auto")
    shared_auto_data_path: str = str(BASE_DIR / "data" / "shared_auto")

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def active_gtfs_path(self) -> str:
        """Use sample data if real GTFS not available."""
        real = Path(self.gtfs_data_path)
        if (real / "routes.txt").exists():
            return self.gtfs_data_path
        return self.sample_data_path


settings = Settings()
