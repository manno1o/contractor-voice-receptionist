"""The receptionist agent: an LLM tool-use loop over a per-call session."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from .calendar import Booking, Calendar
from .config import Settings
from .integrations import CRM, Messenger, Transferer
from .llm import LLM, LLMResponse
from .redact import redact, redact_transcript
from .tools import SYSTEM_PROMPT, TOOLS

log = logging.getLogger(__name__)
MAX_TOOL_ROUNDS = 6
DIAGNOSTIC_FEE = 89


@dataclass
class Deps:
    calendar: Calendar
    messenger: Messenger
    transferer: Transferer
    crm: CRM


@dataclass
class CallSession:
    settings: Settings
    call_sid: str = "CAlocal"
    caller_number: str = ""
    job: dict = field(default_factory=dict)
    offered_slots: list[datetime] = field(default_factory=list)
    booking: Booking | None = None
    transferred: dict | None = None
    ended: str | None = None
    sms_sent: list[str] = field(default_factory=list)
    flags: dict = field(default_factory=dict)
    last_tool_results: dict = field(default_factory=dict)
    transcript: list[dict] = field(default_factory=list)
    tool_log: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return bool(self.transferred or self.ended)

    def fmt_window(self, start: datetime) -> str:
        tz = ZoneInfo(self.settings.business_timezone)
        s = start.astimezone(tz)
        e_hour = (s.hour + 2) % 24

        def h(x: int) -> str:
            return f"{(x - 1) % 12 + 1}{'am' if x < 12 else 'pm'}"

        return f"{s.strftime('%A')} {s.strftime('%b')} {s.day} between {h(s.hour)} and {h(e_hour)}"

    def summary(self) -> dict:
        return {
            "call_sid": self.call_sid,
            "job": {k: v for k, v in self.job.items() if k not in ("callback_phone", "service_address")},
            "priority": self.job.get("priority"),
            "booking": self.booking and {"id": self.booking.id, "start": self.booking.start.isoformat()},
            "transferred": self.transferred,
            "ended": self.ended,
            "transcript": redact_transcript(self.transcript),
        }


@dataclass
class Turn:
    text: str
    latency_ms: float
    tool_calls: list[str]


class ReceptionistAgent:
    def __init__(self, llm: LLM, deps: Deps, settings: Settings, clock: Callable[[], datetime] | None = None):
        self.llm, self.deps, self.settings = llm, deps, settings
        self.clock = clock or (lambda: datetime.now(ZoneInfo(settings.business_timezone)))

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            business_name=self.settings.business_name, zips=", ".join(sorted(self.settings.zips)),
            diagnostic_fee=DIAGNOSTIC_FEE, now=self.clock().strftime("%A %Y-%m-%d %H:%M"),
        )

    def greeting(self, s: CallSession) -> str:
        g = f"Thanks for calling {self.settings.business_name}, this is the virtual receptionist. How can I help?"
        s.transcript.append({"role": "agent", "text": g})
        s.messages.append({"role": "assistant", "content": g})
        return g

    # ------------------------------------------------------------------ main loop
    def respond(self, s: CallSession, caller_text: str) -> Turn:
        t0 = time.perf_counter()
        s.transcript.append({"role": "caller", "text": caller_text})
        s.messages.append({"role": "user", "content": caller_text})
        s.last_tool_results = {}
        spoken: list[str] = []
        called: list[str] = []
        system = self.system_prompt()
        for _ in range(MAX_TOOL_ROUNDS):
            resp: LLMResponse = self.llm.complete(system, s.messages, TOOLS, session=s)
            if resp.text:
                spoken.append(resp.text)
            content = resp.raw_content or (
                ([{"type": "text", "text": resp.text}] if resp.text else [])
                + [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in resp.tool_calls]
            )
            if not resp.tool_calls:
                if content:
                    s.messages.append({"role": "assistant", "content": content})
                break
            s.messages.append({"role": "assistant", "content": content})
            results, s.last_tool_results = [], {}
            for c in resp.tool_calls:
                out = self._run_tool(s, c.name, c.input)
                called.append(c.name)
                s.last_tool_results[c.name] = out
                s.tool_log.append({"tool": c.name, "input": c.input, "output": out})
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": json.dumps(out, default=str)})
            s.messages.append({"role": "user", "content": results})
        text = " ".join(x.strip() for x in spoken if x.strip())
        if not text and s.ended:
            text = "Thanks for calling. Goodbye!"
        s.transcript.append({"role": "agent", "text": text})
        if s.done and not s.flags.get("crm_pushed"):
            s.flags["crm_pushed"] = True
            try:
                self.deps.crm.push(s.summary())
            except Exception:  # CRM outage must never drop the call
                log.exception("crm push failed")
        return Turn(text, (time.perf_counter() - t0) * 1000, called)

    # ------------------------------------------------------------------ tools
    def _run_tool(self, s: CallSession, name: str, inp: dict) -> Any:
        try:
            return getattr(self, f"_tool_{name}")(s, inp or {})
        except AttributeError:
            return {"error": f"unknown tool {name}"}
        except Exception as e:  # tool errors are reported back to the model, not raised
            log.exception("tool %s failed", name)
            return {"error": str(e)}

    def _tool_record_job_details(self, s: CallSession, inp: dict) -> dict:
        clean = {k: v for k, v in inp.items() if v not in (None, "")}
        if "callback_phone" in clean:
            digits = "".join(ch for ch in clean["callback_phone"] if ch.isdigit())[-10:]
            if len(digits) != 10:
                return {"error": "callback_phone must have 10 digits"}
            clean["callback_phone"] = "+1" + digits
        s.job.update(clean)
        missing = [f for f in ("issue", "customer_name", "callback_phone", "service_address", "zip_code") if not s.job.get(f)]
        res: dict = {"ok": True, "missing": missing}
        if s.job.get("zip_code") and s.job["zip_code"] not in self.settings.zips:
            res["warning"] = "zip_code outside service area"
        return res

    def _tool_check_availability(self, s: CallSession, inp: dict) -> dict:
        if not s.job.get("issue"):
            return {"error": "record the issue first"}
        slots = self.deps.calendar.available_slots(self.clock(), s.job.get("priority", "routine"), limit=3)
        s.offered_slots = slots
        return {"windows": [{"index": i, "label": s.fmt_window(x)} for i, x in enumerate(slots)]}

    def _tool_book_appointment(self, s: CallSession, inp: dict) -> dict:
        if s.booking:
            return {"error": "already booked", "booking_id": s.booking.id}
        missing = [f for f in ("customer_name", "callback_phone", "service_address", "zip_code") if not s.job.get(f)]
        if missing:
            return {"error": f"missing {missing}"}
        if s.job["zip_code"] not in self.settings.zips:
            return {"error": "outside service area"}
        idx = int(inp.get("slot_index", 0))
        if not (0 <= idx < len(s.offered_slots)):
            return {"error": "invalid slot_index; call check_availability"}
        try:
            s.booking = self.deps.calendar.book(s.offered_slots[idx], s.job["customer_name"], s.job.get("issue", ""),
                                                s.job.get("priority", "routine"))
        except ValueError as e:
            s.offered_slots = []
            return {"error": str(e)}
        return {"ok": True, "booking_id": s.booking.id, "window": s.fmt_window(s.booking.start)}

    def _tool_send_sms_confirmation(self, s: CallSession, inp: dict) -> dict:
        if not s.booking:
            return {"error": "no booking to confirm"}
        to = s.job.get("callback_phone") or s.caller_number
        body = (f"{self.settings.business_name}: your technician visit is booked for {s.fmt_window(s.booking.start)}. "
                f"Ref {s.booking.id}. Reply C to cancel.")
        sid = self.deps.messenger.send_sms(to, body)
        s.sms_sent.append(sid)
        return {"ok": True, "sid": sid}

    def _tool_transfer_to_human(self, s: CallSession, inp: dict) -> dict:
        reason, summary = inp.get("reason", "other"), redact(inp.get("summary", ""))
        s.transferred = {"reason": reason, "summary": summary}
        if self.settings.human_transfer_number:
            self.deps.transferer.transfer(s.call_sid, self.settings.human_transfer_number, summary)
        else:
            self.deps.transferer.transfer(s.call_sid, "+10000000000", summary)
        return {"ok": True}

    def _tool_end_call(self, s: CallSession, inp: dict) -> dict:
        s.ended = inp.get("reason", "completed")
        return {"ok": True}
