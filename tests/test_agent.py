from app.llm import LLMResponse, ToolCall
from tests.helpers import make


def talk(agent, s, *lines):
    agent.greeting(s)
    return [agent.respond(s, l) for l in lines]


def test_happy_path_books_texts_and_pushes_redacted_crm():
    agent, s, deps = make()
    talk(agent, s, "My furnace isn't working", "This is Maria Lopez", "312-555-0147", "2150 N Clark St, 60614", "first one")
    assert s.booking and s.ended == "completed"
    assert deps.messenger.sent[0]["to"] == "+13125550147"
    rec = deps.crm.records[0]
    assert "callback_phone" not in rec["job"]
    assert "555-0147" not in str(rec["transcript"]) and "Clark" not in str(rec["transcript"])


def test_gas_leak_transfers_with_safety_message():
    agent, s, deps = make()
    (t,) = talk(agent, s, "I smell gas in the kitchen")
    assert "leave" in t.text.lower() and "911" in t.text
    assert s.transferred["reason"] == "safety_emergency"
    assert deps.transferer.transfers[0]["to"] == "+13125550100"
    assert not s.booking


def test_out_of_area_never_books():
    agent, s, deps = make()
    talk(agent, s, "AC is not cooling", "I'm Nina Rossi", "630-555-0110", "100 Main St, 60540", "first one")
    assert s.ended == "out_of_area" and not s.booking and not deps.messenger.sent


class RogueLLM:
    """Tries to book and text before collecting anything - tools must refuse."""

    def complete(self, system, messages, tools, session=None):
        if isinstance(messages[-1]["content"], list):
            return LLMResponse("ok")
        return LLMResponse("", [ToolCall("t1", "book_appointment", {"slot_index": 0}),
                                ToolCall("t2", "send_sms_confirmation", {}),
                                ToolCall("t3", "nonexistent_tool", {})])


def test_tools_enforce_preconditions():
    agent, s, deps = make(RogueLLM())
    agent.greeting(s)
    agent.respond(s, "book me")
    outs = {e["tool"]: e["output"] for e in s.tool_log}
    assert "error" in outs["book_appointment"]
    assert "error" in outs["send_sms_confirmation"]
    assert "error" in outs["nonexistent_tool"]
    assert not s.booking and not deps.messenger.sent


def test_bad_phone_rejected():
    agent, s, _ = make()
    out = agent._tool_record_job_details(s, {"callback_phone": "312-555"})
    assert "error" in out and "callback_phone" not in s.job


def test_tool_rounds_are_bounded():
    class Loop:
        def complete(self, *a, **k):
            return LLMResponse("", [ToolCall("x", "record_job_details", {})])
    agent, s, _ = make(Loop())
    t = agent.respond(s, "hi")
    assert len(t.tool_calls) == 6


def test_message_history_is_valid_anthropic_shape():
    agent, s, _ = make()
    talk(agent, s, "My furnace isn't working", "This is Maria Lopez")
    for prev, nxt in zip(s.messages, s.messages[1:]):
        if prev["role"] == "assistant" and isinstance(prev["content"], list) and any(b["type"] == "tool_use" for b in prev["content"]):
            ids = {b["id"] for b in prev["content"] if b["type"] == "tool_use"}
            assert {b["tool_use_id"] for b in nxt["content"]} == ids
