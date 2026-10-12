from app.calendar import LocalCalendar
from tests.helpers import NOW
import pytest


def test_routine_starts_tomorrow_emergency_today():
    c = LocalCalendar()
    assert c.available_slots(NOW, "routine")[0].date() > NOW.date()
    assert c.available_slots(NOW, "emergency")[0].date() == NOW.date()


def test_double_booking_prevented():
    c = LocalCalendar()
    slot = c.available_slots(NOW, "urgent")[0]
    c.book(slot, "A", "x", "urgent")
    with pytest.raises(ValueError):
        c.book(slot, "B", "y", "urgent")
    assert slot not in c.available_slots(NOW, "urgent")
