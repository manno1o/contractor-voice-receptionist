"""Deepgram streaming STT and ElevenLabs TTS, both 8kHz mu-law for Twilio."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Awaitable, Callable, Protocol

import httpx

log = logging.getLogger(__name__)

OnTranscript = Callable[[str], Awaitable[None]]
OnSpeechStart = Callable[[], Awaitable[None]]


class STTStream(Protocol):
    async def start(self, on_final: OnTranscript, on_speech_start: OnSpeechStart) -> None: ...
    async def send_audio(self, mulaw: bytes) -> None: ...
    async def close(self) -> None: ...


class TTS(Protocol):
    def synthesize(self, text: str) -> AsyncIterator[bytes]: ...


class DeepgramSTT:
    URL = ("wss://api.deepgram.com/v1/listen?model=nova-2-phonecall&encoding=mulaw&sample_rate=8000"
           "&channels=1&interim_results=true&endpointing=300&utterance_end_ms=1000&vad_events=true&smart_format=true")

    def __init__(self, api_key: str, language: str = "multi"):
        self.api_key, self.language = api_key, language
        self.ws = None
        self._task: asyncio.Task | None = None
        self._buf: list[str] = []

    async def start(self, on_final: OnTranscript, on_speech_start: OnSpeechStart) -> None:
        import websockets

        url = self.URL + (f"&language={self.language}" if self.language else "")
        self.ws = await websockets.connect(url, additional_headers={"Authorization": f"Token {self.api_key}"})

        async def reader() -> None:
            async for raw in self.ws:
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "SpeechStarted":
                    await on_speech_start()
                elif kind == "Results":
                    alt = msg["channel"]["alternatives"][0]["transcript"]
                    if msg.get("is_final") and alt:
                        self._buf.append(alt)
                    if msg.get("speech_final") and self._buf:
                        text, self._buf = " ".join(self._buf), []
                        await on_final(text)
                elif kind == "UtteranceEnd" and self._buf:
                    text, self._buf = " ".join(self._buf), []
                    await on_final(text)

        self._task = asyncio.create_task(reader())

    async def send_audio(self, mulaw: bytes) -> None:
        if self.ws:
            await self.ws.send(mulaw)

    async def close(self) -> None:
        if self.ws:
            await self.ws.send(json.dumps({"type": "CloseStream"}))
            await self.ws.close()
        if self._task:
            self._task.cancel()


class ElevenLabsTTS:
    def __init__(self, api_key: str, voice_id: str, model: str = "eleven_flash_v2_5"):
        self.api_key, self.voice_id, self.model = api_key, voice_id, model

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream?output_format=ulaw_8000"
        async with httpx.AsyncClient(timeout=30) as http:
            async with http.stream("POST", url, headers={"xi-api-key": self.api_key},
                                   json={"text": text, "model_id": self.model}) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes(640):  # 80 ms of 8 kHz mu-law
                    yield chunk


class SilentTTS:
    """Silence, one frame per word. Used when no TTS key is set."""

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        for _ in text.split():
            yield b"\xff" * 160
