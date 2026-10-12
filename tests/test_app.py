import base64
import json

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.twilio_security import compute_signature
from tests.helpers import make


class ScriptedSTT:
    """Each inbound media frame 'transcribes' to the next scripted line."""

    def __init__(self, lines):
        self.lines = list(lines)

    async def start(self, on_final, on_speech_start):
        self.on_final = on_final

    async def send_audio(self, mulaw):
        if self.lines:
            await self.on_final(self.lines.pop(0))

    async def close(self):
        pass


class WordTTS:
    async def synthesize(self, text):
        for _ in text.split():
            yield b"\xff" * 160


def _settings(tmp_path):
    st = Settings()
    st.db_path = str(tmp_path / "calls.db")
    st.twilio_auth_token = "test_token"
    st.public_base_url = "https://example.ngrok.app"
    return st


def test_voice_webhook_requires_valid_signature(tmp_path):
    st = _settings(tmp_path)
    client = TestClient(create_app(st, stt_factory=lambda: ScriptedSTT([]), tts=WordTTS()))
    form = {"CallSid": "CA1", "From": "+13125550147"}
    assert client.post("/twilio/voice", data=form).status_code == 403
    sig = compute_signature("test_token", "https://example.ngrok.app/twilio/voice", form)
    r = client.post("/twilio/voice", data=form, headers={"X-Twilio-Signature": sig})
    assert r.status_code == 200
    assert '<Stream url="wss://example.ngrok.app/twilio/stream">' in r.text


def test_media_stream_end_to_end(tmp_path):
    st = _settings(tmp_path)
    agent, _, deps = make()
    lines = ["My furnace isn't working", "This is Maria Lopez", "312-555-0147", "2150 N Clark St, 60614", "first one"]
    app = create_app(st, stt_factory=lambda: ScriptedSTT(lines), tts=WordTTS(), agent=agent)
    client = TestClient(app)
    with client.websocket_connect("/twilio/stream") as ws:
        ws.send_text(json.dumps({"event": "connected"}))
        ws.send_text(json.dumps({"event": "start", "start": {"streamSid": "MZ1", "callSid": "CAws",
                                                              "customParameters": {"from": "+13125550147"}}}))
        frames = 0
        while True:
            m = ws.receive_json()
            if m["event"] == "media":
                frames += 1
                assert m["streamSid"] == "MZ1" and base64.b64decode(m["media"]["payload"])
            if m["event"] == "mark":
                if deps.crm.records:
                    break
                ws.send_text(json.dumps({"event": "mark", "mark": {"name": "utterance_end"}}))
                ws.send_text(json.dumps({"event": "media", "media": {"payload": base64.b64encode(b"\xff" * 160).decode()}}))
        ws.send_text(json.dumps({"event": "stop"}))
    assert frames > 10 and deps.messenger.sent
    saved = client.get("/calls").json()
    assert saved[0]["call_sid"] == "CAws" and saved[0]["booking"]
    assert "555-0147" not in json.dumps(saved)


def test_health(tmp_path):
    client = TestClient(create_app(_settings(tmp_path), stt_factory=lambda: ScriptedSTT([]), tts=WordTTS()))
    assert client.get("/health").json()["ok"]
