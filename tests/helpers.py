from datetime import datetime
from zoneinfo import ZoneInfo

from app.agent import CallSession, Deps, ReceptionistAgent
from app.calendar import LocalCalendar
from app.config import Settings
from app.integrations import FakeCRM, FakeMessenger, FakeTransferer
from app.llm import MockLLM

NOW = datetime(2026, 1, 14, 9, 0, tzinfo=ZoneInfo("America/Chicago"))


def make(llm=None):
    st = Settings()
    st.human_transfer_number = "+13125550100"
    deps = Deps(LocalCalendar(), FakeMessenger(), FakeTransferer(), FakeCRM())
    agent = ReceptionistAgent(llm or MockLLM(), deps, st, clock=lambda: NOW)
    return agent, CallSession(st, call_sid="CAtest"), deps
