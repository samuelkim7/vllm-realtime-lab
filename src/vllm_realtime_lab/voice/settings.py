from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class VoiceSettings(BaseSettings):
    """Where the orchestrator finds the tutor model and Inworld, and where it logs turns.

    Values come from the arguments first, then the environment, then `.env`. The Inworld key keeps
    the name it has in `.env`.
    """

    model_config = SettingsConfigDict(env_prefix="VRL_", env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:8000"
    llm_model: str
    inworld_api_key: SecretStr = Field(validation_alias="INWORLD_API_KEY")
    inworld_base_url: str = "https://api.inworld.ai"
    tts_model: str = "inworld-tts-2"
    tts_voice: str = "Carina"
    voice_log: Path

    def inworld_headers(self) -> dict[str, str]:
        """Return the Inworld auth header."""
        return {"Authorization": f"Basic {self.inworld_api_key.get_secret_value()}"}


class ProxyAuth(BaseSettings):
    """The Modal proxy auth token, read from the environment and then from `.env`.

    The token opens every endpoint with proxy auth in the workspace. Send it only to a URL that
    Modal itself returned.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    modal_proxy_key: SecretStr = Field(validation_alias="MODAL_PROXY_KEY")
    modal_proxy_secret: SecretStr = Field(validation_alias="MODAL_PROXY_SECRET")

    def headers(self) -> dict[str, str]:
        """Return the Modal proxy auth headers."""
        return {
            "Modal-Key": self.modal_proxy_key.get_secret_value(),
            "Modal-Secret": self.modal_proxy_secret.get_secret_value(),
        }
