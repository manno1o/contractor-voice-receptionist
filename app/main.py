"""FastAPI app: Twilio voice webhook + bidirectional Media Streams websocket."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Callable
from xml.sax.saxutils import escape

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from .agent import CallSession, Deps, ReceptionistAgent
from .calendar import GoogleCalendar, LocalCalendar
from .config import Settings, get_settings
from .integrations import FakeCRM, FakeMessenger, FakeTransferer, TwilioClient, WebhookCRM
from .llm import AnthropicLLM, MockLLM
from .speech import DeepgramSTT, ElevenLabsTTS, SilentTTS
from .store import CallStore
from .twilio_security import valid_signature

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("receptionist")


def build_agent(settings: Settings) -> ReceptionistAgent:
    llm = (AnthropicLLM(settings.anthropic_api_key, settings.anthropic_model)
           if settings.llm_provider == "anthropic" else MockLLM())
    twilio = (TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token, settings.twilio_from_number)
              if settings.twilio_account_sid else None)
    calendar = (GoogleCalendar(settings.google_calendar_id, settings.google_service_account_json, settings.business_timezone)
                if settings.calendar_backend == "google" else LocalCalendar(settings.business_timezone))
    deps = Deps(
        calendar=calendar,
        messenger=twilio or FakeMessenger(),
        transferer=twilio or FakeTransferer(),
        crm=WebhookCRM(settings.crm_webhook_url) if settings.crm_webhook_url else FakeCRM(),
    )
    return ReceptionistAgent(llm, deps, settings)


def create_app(settings: Settings | None = None, stt_factory: Callable | None = None, tts=None,
               agent: ReceptionistAgent | None = None) -> FastAPI:
    settings = settings or get_settings()
    agent = agent or build_agent(settings)
    store = CallStore(settings.db_path)
    stt_factory = stt_factory or (lambda: DeepgramSTT(settings.deepgram_api_key))
    tts = tts or (ElevenLabsTTS(settings.elevenlabs_api_key, settings.elevenlabs_voice_id)
                  if settings.elevenlabs_api_key else SilentTTS())
    app = FastAPI(title="Contractor Voice Receptionist")
    app.state.agent, app.state.store, app.state.sessions = agent, store, {}

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "llm": settings.llm_provider, "calendar": settings.calendar_backend}

    @app.post("/twilio/voice")
    async def voice(request: Request) -> Response:
        form = dict(await request.form())
        if settings.validate_twilio_signature and settings.twilio_auth_token:
            url = settings.public_base_url.rstrip("/") + "/twilio/voice"
            if not valid_signature(settings.twilio_auth_token, url, form, request.headers.get("X-Twilio-Signature", "")):
                raise HTTPException(403, "invalid signature")
        ws_url = settings.public_base_url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/twilio/stream"
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?><Response><Connect>'
            f'<Stream url="{escape(ws_url)}"><Parameter name="from" value="{escape(str(form.get("From", "")))}"/></Stream>'
            "</Connect></Response>"
        )
        return Response(twiml, media_type="application/xml")

    @app.get("/calls")
    def calls(limit: int = 20) -> JSONResponse:
        return JSONResponse(store.recent(limit))

    @app.websocket("/twilio/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        session: CallSession | None = None
        stream_sid = ""
        stt = stt_factory()
        speaking = asyncio.Event()
        lock = asyncio.Lock()

        async def speak(text: str) -> None:
            if not text:
                return
            speaking.set()
            async for chunk in tts.synthesize(text):
                if not speaking.is_set():
                    break  # barge-in
                await ws.send_json({"event": "media", "streamSid": stream_sid,
                                    "media": {"payload": base64.b64encode(chunk).decode()}})
            await ws.send_json({"event": "mark", "streamSid": stream_sid, "mark": {"name": "utterance_end"}})

        async def on_final(text: str) -> None:
            nonlocal session
            if session is None or session.done:
                return
            async with lock:
                turn = await asyncio.to_thread(agent.respond, session, text)
                log.info("turn %s latency=%.0fms tools=%s", session.call_sid, turn.latency_ms, turn.tool_calls)
                await speak(turn.text)
                if session.done:
                    store.save(session.summary())
                    if session.ended and isinstance(agent.deps.transferer, TwilioClient):
                        await asyncio.sleep(1.5)
                        await asyncio.to_thread(agent.deps.transferer.http.post,
                                                f"/Calls/{session.call_sid}.json", data={"Status": "completed"})

        async def on_speech_start() -> None:
            if speaking.is_set():
                speaking.clear()
                await ws.send_json({"event": "clear", "streamSid": stream_sid})

        try:
            while True:
                msg = json.loads(await ws.receive_text())
                ev = msg.get("event")
                if ev == "start":
                    st = msg["start"]
                    stream_sid = st["streamSid"]
                    session = CallSession(settings, call_sid=st.get("callSid", ""),
                                          caller_number=st.get("customParameters", {}).get("from", ""))
                    app.state.sessions[session.call_sid] = session
                    await stt.start(on_final, on_speech_start)
                    await speak(agent.greeting(session))
                elif ev == "media":
                    await stt.send_audio(base64.b64decode(msg["media"]["payload"]))
                elif ev == "mark":
                    speaking.clear()
                elif ev == "stop":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            await stt.close()
            if session is not None:
                store.save(session.summary())
                app.state.sessions.pop(session.call_sid, None)

    return app


app = create_app()
