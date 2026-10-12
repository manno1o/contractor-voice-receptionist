"""SMS, CRM webhook, live transfer. Fakes are used in tests/evals."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Protocol

import httpx

log = logging.getLogger(__name__)


class Messenger(Protocol):
    def send_sms(self, to: str, body: str) -> str: ...


class Transferer(Protocol):
    def transfer(self, call_sid: str, to: str, whisper: str) -> None: ...


class CRM(Protocol):
    def push(self, record: dict) -> None: ...


@dataclass
class FakeMessenger:
    sent: list[dict] = field(default_factory=list)

    def send_sms(self, to: str, body: str) -> str:
        self.sent.append({"to": to, "body": body})
        return f"SMfake{len(self.sent)}"


@dataclass
class FakeTransferer:
    transfers: list[dict] = field(default_factory=list)

    def transfer(self, call_sid: str, to: str, whisper: str) -> None:
        self.transfers.append({"call_sid": call_sid, "to": to, "whisper": whisper})


@dataclass
class FakeCRM:
    records: list[dict] = field(default_factory=list)

    def push(self, record: dict) -> None:
        self.records.append(record)


class TwilioClient:
    """Minimal Twilio REST client (SMS + live call update) using httpx."""

    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.sid, self.from_number = account_sid, from_number
        self.http = httpx.Client(
            base_url=f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}",
            auth=(account_sid, auth_token), timeout=10,
        )

    def send_sms(self, to: str, body: str) -> str:
        r = self.http.post("/Messages.json", data={"To": to, "From": self.from_number, "Body": body})
        r.raise_for_status()
        return r.json()["sid"]

    def transfer(self, call_sid: str, to: str, whisper: str) -> None:
        # Redirect the live call: stop the media stream and dial the on-call human.
        from xml.sax.saxutils import escape

        twiml = (
            "<Response><Say>Connecting you to a team member now.</Say>"
            f"<Dial><Number>{escape(to)}</Number></Dial></Response>"
        )
        r = self.http.post(f"/Calls/{call_sid}.json", data={"Twiml": twiml})
        r.raise_for_status()
        log.info("transferred call %s (whisper: %s)", call_sid, whisper[:80])


class WebhookCRM:
    def __init__(self, url: str):
        self.url = url

    def push(self, record: dict) -> None:
        payload = json.dumps(record, default=str, sort_keys=True)
        key = hashlib.sha256(payload.encode()).hexdigest()
        httpx.post(self.url, content=payload, timeout=10,
                   headers={"Content-Type": "application/json", "Idempotency-Key": key}).raise_for_status()
