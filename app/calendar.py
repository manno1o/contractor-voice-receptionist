"""In-memory scheduler for dev/tests, Google Calendar for real use."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from typing import Protocol
from zoneinfo import ZoneInfo


@dataclass
class Booking:
    id: str
    start: datetime
    end: datetime
    customer_name: str
    issue: str
    priority: str


class Calendar(Protocol):
    def available_slots(self, now: datetime, priority: str, limit: int = 3) -> list[datetime]: ...
    def book(self, start: datetime, customer_name: str, issue: str, priority: str) -> Booking: ...


WORK_START, WORK_END, SLOT_HOURS = time(8), time(17), 2


@dataclass
class LocalCalendar:
    tz: str = "America/Chicago"
    bookings: list[Booking] = field(default_factory=list)

    def _is_free(self, start: datetime) -> bool:
        end = start + timedelta(hours=SLOT_HOURS)
        return all(end <= b.start or start >= b.end for b in self.bookings)

    def available_slots(self, now: datetime, priority: str, limit: int = 3) -> list[datetime]:
        tzinfo = ZoneInfo(self.tz)
        now = now.astimezone(tzinfo)
        slots: list[datetime] = []
        day = now.date()
        # Emergencies may use today's remaining slots; routine work starts tomorrow.
        if priority not in ("emergency", "urgent"):
            day += timedelta(days=1)
        for _ in range(14):
            if day.weekday() < 6:  # Mon-Sat
                h = WORK_START.hour
                while h + SLOT_HOURS <= WORK_END.hour:
                    start = datetime.combine(day, time(h), tzinfo)
                    if start >= now + timedelta(hours=1) and self._is_free(start):
                        slots.append(start)
                        if len(slots) >= limit:
                            return slots
                    h += SLOT_HOURS
            day += timedelta(days=1)
        return slots

    def book(self, start: datetime, customer_name: str, issue: str, priority: str) -> Booking:
        if not self._is_free(start):
            raise ValueError("slot no longer available")
        b = Booking(uuid.uuid4().hex[:8], start, start + timedelta(hours=SLOT_HOURS), customer_name, issue, priority)
        self.bookings.append(b)
        return b


class GoogleCalendar:
    """Needs requirements-google.txt."""

    def __init__(self, calendar_id: str, service_account_json: str, tz: str):
        from google.oauth2 import service_account  # type: ignore
        from googleapiclient.discovery import build  # type: ignore

        creds = service_account.Credentials.from_service_account_file(
            service_account_json, scopes=["https://www.googleapis.com/auth/calendar"]
        )
        self.svc = build("calendar", "v3", credentials=creds, cache_discovery=False)
        self.calendar_id, self.tz = calendar_id, tz
        self._local = LocalCalendar(tz)

    def _sync_busy(self, now: datetime) -> None:
        body = {
            "timeMin": now.isoformat(),
            "timeMax": (now + timedelta(days=14)).isoformat(),
            "items": [{"id": self.calendar_id}],
        }
        busy = self.svc.freebusy().query(body=body).execute()["calendars"][self.calendar_id]["busy"]
        self._local.bookings = [
            Booking("busy", datetime.fromisoformat(b["start"].replace("Z", "+00:00")),
                    datetime.fromisoformat(b["end"].replace("Z", "+00:00")), "", "", "")
            for b in busy
        ]

    def available_slots(self, now: datetime, priority: str, limit: int = 3) -> list[datetime]:
        self._sync_busy(now)
        return self._local.available_slots(now, priority, limit)

    def book(self, start: datetime, customer_name: str, issue: str, priority: str) -> Booking:
        end = start + timedelta(hours=SLOT_HOURS)
        ev = self.svc.events().insert(calendarId=self.calendar_id, body={
            "summary": f"[{priority.upper()}] {customer_name}: {issue[:60]}",
            "start": {"dateTime": start.isoformat(), "timeZone": self.tz},
            "end": {"dateTime": end.isoformat(), "timeZone": self.tz},
        }).execute()
        return Booking(ev["id"], start, end, customer_name, issue, priority)
