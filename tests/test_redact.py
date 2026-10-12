from app.redact import redact


def test_phone_email_card_ssn_address():
    t = "Call 312-555-0147 or (773) 555 0112, mail jo@example.com, card 4111 1111 1111 1111, ssn 123-45-6789, I live at 2150 N Clark St, 60614"
    r = redact(t)
    for leaked in ["555-0147", "555 0112", "jo@example.com", "4111", "123-45-6789", "Clark", "60614"]:
        assert leaked not in r
    assert "[PHONE]" in r and "[EMAIL]" in r and "[ADDRESS]" in r


def test_spoken_digits():
    assert "[PHONE]" in redact("it's three one two five five five zero one four seven")


def test_plain_text_untouched():
    s = "My furnace is not heating since Monday."
    assert redact(s) == s
