from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


@dataclass
class Settings:

    business_name: str = field(default_factory=lambda: _env("BUSINESS_NAME", "Lakeside Heating & Plumbing"))
    business_timezone: str = field(default_factory=lambda: _env("BUSINESS_TZ", "America/Chicago"))
    service_area_zips: str = field(default_factory=lambda: _env("SERVICE_AREA_ZIPS", "60601,60602,60607,60614,60618,60625,60640,60647,60657"))
    human_transfer_number: str = field(default_factory=lambda: _env("HUMAN_TRANSFER_NUMBER", ""))

    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "mock"))  # mock | anthropic
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-sonnet-4-5"))

    stt_provider: str = field(default_factory=lambda: _env("STT_PROVIDER", "deepgram"))
    deepgram_api_key: str = field(default_factory=lambda: _env("DEEPGRAM_API_KEY"))

    tts_provider: str = field(default_factory=lambda: _env("TTS_PROVIDER", "elevenlabs"))
    elevenlabs_api_key: str = field(default_factory=lambda: _env("ELEVENLABS_API_KEY"))
    elevenlabs_voice_id: str = field(default_factory=lambda: _env("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"))

    twilio_account_sid: str = field(default_factory=lambda: _env("TWILIO_ACCOUNT_SID"))
    twilio_auth_token: str = field(default_factory=lambda: _env("TWILIO_AUTH_TOKEN"))
    twilio_from_number: str = field(default_factory=lambda: _env("TWILIO_FROM_NUMBER"))
    public_base_url: str = field(default_factory=lambda: _env("PUBLIC_BASE_URL", "http://localhost:8000"))
    validate_twilio_signature: bool = field(default_factory=lambda: _env("VALIDATE_TWILIO_SIGNATURE", "true").lower() == "true")

    calendar_backend: str = field(default_factory=lambda: _env("CALENDAR_BACKEND", "local"))  # local | google
    google_calendar_id: str = field(default_factory=lambda: _env("GOOGLE_CALENDAR_ID"))
    google_service_account_json: str = field(default_factory=lambda: _env("GOOGLE_SERVICE_ACCOUNT_JSON"))
    crm_webhook_url: str = field(default_factory=lambda: _env("CRM_WEBHOOK_URL"))

    db_path: str = field(default_factory=lambda: _env("DB_PATH", "data/calls.db"))

    @property
    def zips(self) -> set[str]:
        return {z.strip() for z in self.service_area_zips.split(",") if z.strip()}


def get_settings() -> Settings:
    return Settings()
