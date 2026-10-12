"""Twilio request signature validation (X-Twilio-Signature), stdlib only."""
import base64
import hashlib
import hmac


def compute_signature(auth_token: str, url: str, params: dict) -> str:
    data = url + "".join(f"{k}{params[k]}" for k in sorted(params))
    return base64.b64encode(hmac.new(auth_token.encode(), data.encode(), hashlib.sha1).digest()).decode()


def valid_signature(auth_token: str, url: str, params: dict, signature: str) -> bool:
    return hmac.compare_digest(compute_signature(auth_token, url, params), signature or "")
