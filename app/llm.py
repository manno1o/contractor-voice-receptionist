"""Claude backend + a rule-based mock that speaks the same tool protocol (for CI)."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_content: list[dict] | None = None  # assistant content blocks, for message history


class LLM(Protocol):
    def complete(self, system: str, messages: list[dict], tools: list[dict], session: Any = None) -> LLMResponse: ...


class AnthropicLLM:
    def __init__(self, api_key: str, model: str, max_tokens: int = 300):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model, self.max_tokens = model, max_tokens

    def complete(self, system: str, messages: list[dict], tools: list[dict], session: Any = None) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system, messages=messages, tools=tools,
        )
        text, calls, raw = "", [], []
        for block in resp.content:
            if block.type == "text":
                text += block.text
                raw.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                calls.append(ToolCall(block.id, block.name, dict(block.input)))
                raw.append({"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input)})
        return LLMResponse(text.strip(), calls, raw)


# --------------------------------------------------------------------------- mock policy

SAFETY = re.compile(r"\b(gas smell|smell(s|ing)? (of |like )?gas|smell gas|rotten egg|carbon monoxide|co alarm|co detector|"
                    r"monoxide|sparks?|smoke|burning smell|on fire|huele a gas|olor a gas)\b", re.I)
EMERGENCY = re.compile(r"\b(burst|flood(ing|ed)?|water everywhere|sewage|sewer backup|backing up|pipe (broke|burst)|"
                       r"inundaci[oó]n|tuber[ií]a rota)\b", re.I)
URGENT = re.compile(r"\b(no heat|furnace (isn'?t|is not|not|won'?t)|heat (isn'?t|is not|not) working|no (ac|a/?c|air)|ac (isn'?t|is not|not|stopped)|"
                    r"no hot water|water heater|freezing|not cooling|not heating|sin calefacci[oó]n|no hay agua caliente)\b", re.I)
ROUTINE = re.compile(r"\b(tune[- ]?up|maintenance|quote|estimate|install|replace|drip|leak(y|ing)?|clog(ged)?|slow drain|"
                     r"toilet|faucet|thermostat|filter|noise|noisy|mantenimiento|gotea)\b", re.I)
VULNERABLE = re.compile(r"\b(baby|infant|newborn|elderly|grandm|grandp|oxygen|medical|below zero|minus|bebé|abuela)\b", re.I)
HUMAN = re.compile(r"\b(real person|human|manager|supervisor|owner|operator|representative|someone real|talk to a person|persona real)\b", re.I)
BILLING = re.compile(r"\b(billing|my bill|the bill|invoice|charged me|refund|overcharged|factura)\b", re.I)
EXISTING = re.compile(r"\b(reschedul|cancel|my appointment|already (booked|scheduled)|technician (is|was) (late|supposed)|existing (job|appointment))\w*", re.I)
COMPLAINT = re.compile(r"\b(complain|terrible|unacceptable|worst|furious|lawyer|sue)\b", re.I)
OUT_OF_SCOPE = re.compile(r"\b(roof(ing)?|electrical panel|breaker panel|rewir|refrigerator|fridge|dishwasher|washing machine|"
                          r"dryer|solar|pool|gutter)\b", re.I)
SPAM = re.compile(r"\b(seo|google listing|extended warranty|business loan|merchant services|marketing services|"
                  r"you'?ve won|press one|this is not a sales|lead gen|ranking)\b", re.I)
PRICE = re.compile(r"\b(how much|price|cost|charge|rate|cu[aá]nto)\b", re.I)
SPANISH = re.compile(r"\b(hola|necesito|tengo|por favor|gracias|mi nombre|calefacci[oó]n|agua|s[ií])\b", re.I)
NAME = re.compile(r"\b(?i:my name is|this is|name'?s|it'?s|i'?m|me llamo|mi nombre es)[,:]?\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)")
PHONE = re.compile(r"(?:\+?1[\s.-]?)?\(?(\d{3})\)?[\s.-]?(\d{3})[\s.-]?(\d{4})")
ADDRESS = re.compile(r"\b(\d{1,6}\s+(?:[NSEW]\.?\s+)?[A-Za-z][A-Za-z ]{1,30}?\s(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Dr|Drive|Ln|Lane|Ct|Court|Pl|Place|Way))\b\.?", re.I)
ZIP = re.compile(r"\b(\d{5})\b")
CHOICE_FIRST = re.compile(r"\b(first|earlier|earliest|morning|that works|sounds good|yes|yeah|sure|ok(ay)?|perfect|primero|est[aá] bien|s[ií])\b", re.I)
CHOICE_SECOND = re.compile(r"\b(second|later one|afternoon|other one|segundo)\b", re.I)
GOODBYE = re.compile(r"\b(never ?mind|hang up|bye|goodbye|forget it|no thanks|not interested)\b", re.I)
NOT_NAMES = {"Calling", "Having", "Not", "Just", "Looking", "Sorry", "Here", "Fine", "Good"}


def _tc(name: str, inp: dict | None = None) -> ToolCall:
    return ToolCall("toolu_" + uuid.uuid4().hex[:12], name, inp or {})


class MockLLM:

    def __init__(self, diagnostic_fee: int = 89):
        self.fee = diagnostic_fee

    @staticmethod
    def _last_user_text(messages: list[dict]) -> str:
        for m in reversed(messages):
            if m["role"] == "user" and isinstance(m["content"], str):
                return m["content"]
        return ""

    @staticmethod
    def _pending_tool_results(messages: list[dict]) -> bool:
        return bool(messages) and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list)

    def _say(self, s, en: str, es: str | None = None) -> LLMResponse:
        return LLMResponse(es if (s.job.get("language") == "es" and es) else en)

    def complete(self, system: str, messages: list[dict], tools: list[dict], session: Any = None) -> LLMResponse:
        s = session
        text = self._last_user_text(messages)
        job = s.job

        # After tool results, decide next step from state (do not re-parse the utterance).
        if self._pending_tool_results(messages):
            return self._next_step(s, text, after_tools=True)

        # ---- 1. extract facts from the utterance
        upd: dict = {}
        if SPANISH.search(text) and job.get("language") != "es" and not re.search(r"\b(the|my|is|and)\b", text, re.I):
            upd["language"] = "es"
        if m := NAME.search(text):
            if m.group(1).split()[0] not in NOT_NAMES:
                upd["customer_name"] = m.group(1)
        if m := PHONE.search(text):
            upd["callback_phone"] = "+1" + "".join(m.groups())
        if m := ADDRESS.search(text):
            upd["service_address"] = m.group(1).strip()
        if (m := ZIP.search(re.sub(PHONE, "", text))):
            upd["zip_code"] = m.group(1)
        if not job.get("issue"):
            pr = None
            if EMERGENCY.search(text) or (URGENT.search(text) and VULNERABLE.search(text)):
                pr = "emergency"
            elif URGENT.search(text):
                pr = "urgent"
            elif ROUTINE.search(text):
                pr = "routine"
            if pr:
                upd["issue"] = text.strip()[:160]
                upd["priority"] = pr
                upd["trade"] = "plumbing" if re.search(r"pipe|drain|toilet|faucet|water heater|hot water|sewage|sewer|leak|flood|clog|tuber|agua", text, re.I) else "hvac"

        # ---- 2. hard routing rules (safety > human > scope > spam)
        if SAFETY.search(text):
            calls = ([_tc("record_job_details", {**upd, "priority": "emergency", "issue": text.strip()[:160]})] if True else [])
            calls.append(_tc("transfer_to_human", {"reason": "safety_emergency", "summary": "Possible gas/CO/fire hazard: " + text[:100]}))
            say = ("Please leave the building right now and call 911 or your gas utility from outside. "
                   "I'm connecting you to our on-call technician.")
            if job.get("language") == "es" or upd.get("language") == "es":
                say = "Por favor salga del edificio ahora y llame al 911 desde afuera. Le comunico con nuestro técnico de guardia."
            return LLMResponse(say, calls)
        if SPAM.search(text):
            return LLMResponse("Thanks, but we're not interested. Have a good day.", [_tc("end_call", {"reason": "spam"})])
        if HUMAN.search(text) or BILLING.search(text) or EXISTING.search(text) or COMPLAINT.search(text):
            reason = ("billing" if BILLING.search(text) else "existing_job" if EXISTING.search(text)
                      else "complaint" if COMPLAINT.search(text) else "caller_request")
            calls = [_tc("record_job_details", upd)] if upd else []
            calls.append(_tc("transfer_to_human", {"reason": reason, "summary": f"{reason.replace('_', ' ')}: {text[:100]}"}))
            return self._with(calls, s, "Of course, let me connect you with someone from our team now.",
                              "Claro, le comunico con alguien de nuestro equipo.")
        if OUT_OF_SCOPE.search(text) and not job.get("issue"):
            return self._with([_tc("end_call", {"reason": "out_of_scope"})], s,
                              "Sorry, we only do heating, cooling and plumbing, so we can't help with that one. Good luck!")
        if GOODBYE.search(text) and not s.offered_slots:
            return LLMResponse("No problem, thanks for calling. Goodbye!", [_tc("end_call", {"reason": "caller_hangup"})])

        # ---- 3. slot choice
        if s.offered_slots and not s.booking:
            idx = 1 if CHOICE_SECOND.search(text) else 0 if CHOICE_FIRST.search(text) else None
            if idx is not None:
                calls = [_tc("record_job_details", upd)] if upd else []
                return LLMResponse("", calls + [_tc("book_appointment", {"slot_index": idx})])

        if upd:
            # Price questions get the policy answer, after recording facts.
            return LLMResponse("", [_tc("record_job_details", upd)])
        if PRICE.search(text):
            s.flags["fee_quoted"] = True
            return self._say(s, f"I can't quote repairs over the phone, but the diagnostic visit is ${self.fee}, "
                                f"and that's credited toward the repair. Want me to get a technician out?",
                             f"No puedo cotizar reparaciones por teléfono, pero la visita de diagnóstico cuesta ${self.fee}. ¿Le agendo una visita?")
        return self._next_step(s, text, after_tools=False)

    def _with(self, calls, s, en, es=None) -> LLMResponse:
        r = self._say(s, en, es)
        r.tool_calls = calls
        return r

    def _next_step(self, s, text: str, after_tools: bool) -> LLMResponse:
        job, es = s.job, s.job.get("language") == "es"
        last = s.last_tool_results
        if s.ended or s.transferred:
            return LLMResponse("")
        if "end_call" in last:
            return LLMResponse("")
        if "send_sms_confirmation" in last:
            return LLMResponse(
                "Gracias, ¡hasta pronto!" if es else "You're all set and I've texted you the details. Thanks for calling!",
                [_tc("end_call", {"reason": "completed"})])
        if "book_appointment" in last and s.booking:
            w = s.fmt_window(s.booking.start)
            return LLMResponse(
                f"Listo, su cita es {w}." if es else f"You're booked for {w}.",
                [_tc("send_sms_confirmation")])
        if "book_appointment" in last and not s.booking:
            return LLMResponse("", [_tc("check_availability")])
        if "check_availability" in last and s.offered_slots:
            opts = " or ".join(s.fmt_window(x) for x in s.offered_slots[:2])
            return self._say(s, f"I can get a technician out {opts}. Which works better?",
                             f"Puedo enviar un técnico {' o '.join(s.fmt_window(x) for x in s.offered_slots[:2])}. ¿Cuál prefiere?")
        if "check_availability" in last and not s.offered_slots:
            return LLMResponse("I'm sorry, we're fully booked. Let me get a dispatcher for you.",
                               [_tc("transfer_to_human", {"reason": "other", "summary": "No availability found"})])
        if job.get("zip_code") and job["zip_code"] not in s.settings.zips:
            return LLMResponse(
                "Lo siento, esa zona está fuera de nuestra área de servicio." if es else
                "I'm sorry, that ZIP code is outside our service area, so we can't send a technician. Thanks for calling.",
                [_tc("end_call", {"reason": "out_of_area"})])
        # ask for missing fields in order
        if not job.get("issue"):
            return self._say(s, f"Thanks for calling {s.settings.business_name}. What's going on with your heating, cooling or plumbing?",
                             "Gracias por llamar. ¿Qué problema tiene con su calefacción, aire o plomería?")
        prefix = ""
        if PRICE.search(text) and not s.flags.get("fee_quoted"):
            s.flags["fee_quoted"] = True
            prefix = (f"La visita de diagnóstico cuesta ${self.fee}. " if es else
                      f"I can't quote repairs by phone, but the diagnostic visit is ${self.fee}, credited toward the repair. ")
        if job.get("priority") == "emergency" and not s.flags.get("ack_emergency"):
            s.flags["ack_emergency"] = True
            prefix = ("Entendido, lo tratamos como emergencia. " if es else
                      "Okay, I'm treating this as an emergency. If water is spreading, shut off the main water valve if you can. ")
        if not job.get("customer_name"):
            return self._say(s, prefix + "Can I get your name?", prefix + "¿Me da su nombre?")
        if not job.get("callback_phone"):
            return self._say(s, prefix + "What's the best callback number?", prefix + "¿Cuál es su número de teléfono?")
        if not s.flags.get("phone_confirmed"):
            s.flags["phone_confirmed"] = True
            p = job["callback_phone"][-10:]
            prefix += (f"Tengo {p[:3]}-{p[3:6]}-{p[6:]}. " if es else f"I have {p[:3]}-{p[3:6]}-{p[6:]}. ")
        if not job.get("service_address") or not job.get("zip_code"):
            return self._say(s, prefix + "And the service address with ZIP code?", prefix + "¿Y la dirección con código postal?")
        if not s.offered_slots:
            return LLMResponse(prefix, [_tc("check_availability")])
        if PRICE.search(text):
            return self._say(s, f"The diagnostic visit is ${self.fee}, credited toward the repair. Which window works?")
        opts = " or ".join(s.fmt_window(x) for x in s.offered_slots[:2])
        return self._say(s, f"Which works better: {opts}?", "¿Cuál horario prefiere?")
