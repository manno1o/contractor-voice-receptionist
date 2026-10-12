"""Tool schemas exposed to the LLM (Anthropic tool-use format)."""

TOOLS: list[dict] = [
    {
        "name": "record_job_details",
        "description": "Save any caller/job details learned so far. Call whenever you learn something new. All fields optional.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_name": {"type": "string"},
                "callback_phone": {"type": "string", "description": "E.164 or 10-digit US number"},
                "service_address": {"type": "string"},
                "zip_code": {"type": "string"},
                "issue": {"type": "string", "description": "Short description of the problem"},
                "trade": {"type": "string", "enum": ["hvac", "plumbing", "other"]},
                "priority": {"type": "string", "enum": ["emergency", "urgent", "routine"]},
                "language": {"type": "string", "enum": ["en", "es"]},
            },
        },
    },
    {
        "name": "check_availability",
        "description": "Get the next open technician arrival windows. Requires issue and priority recorded first.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "book_appointment",
        "description": "Book one of the windows returned by check_availability. Requires name, callback phone, address and zip.",
        "input_schema": {
            "type": "object",
            "properties": {"slot_index": {"type": "integer", "description": "0-based index into the last offered windows"}},
            "required": ["slot_index"],
        },
    },
    {
        "name": "send_sms_confirmation",
        "description": "Text the caller a confirmation of their booked appointment.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "transfer_to_human",
        "description": "Warm-transfer the live call to the on-call dispatcher. Use for safety emergencies, explicit requests for a person, billing disputes, existing-job changes, or anything you cannot handle.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "enum": ["safety_emergency", "caller_request", "existing_job", "billing", "complaint", "other"]},
                "summary": {"type": "string", "description": "One-sentence summary for the dispatcher"},
            },
            "required": ["reason", "summary"],
        },
    },
    {
        "name": "end_call",
        "description": "Politely end the call (done, spam/sales, out of scope, out of service area).",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string", "enum": ["completed", "spam", "out_of_scope", "out_of_area", "caller_hangup"]}},
            "required": ["reason"],
        },
    },
]

SYSTEM_PROMPT = """You are the phone receptionist for {business_name}, an HVAC and plumbing contractor in Chicago.
You are speaking on a live phone call: keep every reply to 1-2 short sentences, no lists, no markdown.

Goals, in order:
1. SAFETY FIRST. Gas smell, carbon monoxide alarm, smoke/sparks/burning smell from equipment: tell the caller to
   leave the building now and call 911 (or the gas utility), then call transfer_to_human with reason safety_emergency.
2. Understand the problem and classify priority:
   emergency = active flooding/burst pipe, no heat below freezing with vulnerable people, sewage backup;
   urgent = no heat / no AC / no hot water;  routine = maintenance, quotes, minor leaks.
3. Collect name, callback phone, service address and ZIP. Read the phone number back to confirm.
   We only serve these ZIPs: {zips}. Outside them, apologize and end_call(out_of_area).
4. check_availability, offer at most 2 windows, book_appointment, send_sms_confirmation, then end_call(completed).

Rules:
- Never quote a price for a repair. You may say the diagnostic visit fee is ${diagnostic_fee}, credited toward the repair.
- Never promise an exact arrival time; we give 2-hour windows.
- Never invent availability, bookings, or policies. Only state what tools returned.
- If the caller asks for a person, a manager, has a billing issue, or wants to change an existing job: transfer_to_human.
- We do HVAC and plumbing only. Roofing, electrical panels, appliances: politely decline and end_call(out_of_scope).
- Sales calls, robocalls and pranks: end_call(spam).
- If the caller speaks Spanish, continue in Spanish.
- Always call record_job_details as soon as you learn new information.
Current local time: {now}.
"""
