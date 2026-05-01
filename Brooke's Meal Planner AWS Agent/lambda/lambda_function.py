"""Lambda entrypoint.

Expects an HTTP event from API Gateway or Lambda Function URL with a JSON body:
    {"user_id": "...", "message": "..."}

Returns:
    {"reply": "...", "request_id": "..."}

CORS is handled at the Function URL level; the headers returned here are a
backup for the API Gateway path.

Security: if APP_TOKEN is set in the environment, every request must include
a matching x-app-token header. If APP_TOKEN is unset, the check is skipped
(useful for local testing, not recommended for production).
"""
import json
import os
import uuid

from agent import run_agent
from db import load_history, save_history
from obs import log_event


# Shared secret for defense-in-depth against someone discovering the public
# Function URL. If APP_TOKEN is not set, the check is skipped.
EXPECTED_TOKEN = os.environ.get("APP_TOKEN", "")


def lambda_handler(event, context):
    request_id = (
        context.aws_request_id if context and hasattr(context, "aws_request_id")
        else str(uuid.uuid4())
    )

    # Handle CORS preflight without requiring auth. Preflight requests can't
    # carry custom headers, so checking the token here would block every POST.
    method = (
        event.get("httpMethod")
        or event.get("requestContext", {}).get("http", {}).get("method")
    )
    if method == "OPTIONS":
        return _response(200, {}, request_id)

    # Shared-secret check. API Gateway v2 and Lambda Function URLs both
    # lowercase incoming header names, so we look up the lowercase key.
    if EXPECTED_TOKEN:
        req_headers = event.get("headers") or {}
        token = req_headers.get("x-app-token", "")
        if token != EXPECTED_TOKEN:
            log_event("unauthorized", {"request_id": request_id})
            return _response(401, {"error": "Unauthorized"}, request_id)

    try:
        body_raw = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            import base64
            body_raw = base64.b64decode(body_raw).decode("utf-8")
        body = json.loads(body_raw)
    except (json.JSONDecodeError, ValueError) as e:
        log_event("bad_request", {"request_id": request_id, "error": str(e)})
        return _response(400, {"error": "Invalid JSON body"}, request_id)

    user_id = (body.get("user_id") or "demo-user").strip()
    user_message = (body.get("message") or "").strip()

    if not user_message:
        return _response(400, {"error": "Empty message"}, request_id)

    log_event(
        "request",
        {
            "request_id": request_id,
            "user_id": user_id,
            "message_preview": user_message[:200],
            "message_len": len(user_message),
        },
    )

    try:
        history = load_history(user_id, limit=20)
        reply, updated_history = run_agent(
            user_id, user_message, history, request_id
        )
        save_history(user_id, updated_history)
    except Exception as e:
        log_event(
            "handler_error",
            {"request_id": request_id, "error": str(e), "type": type(e).__name__},
        )
        return _response(
            500,
            {"error": "Internal error", "request_id": request_id},
            request_id,
        )

    log_event(
        "response",
        {"request_id": request_id, "reply_len": len(reply)},
    )
    return _response(200, {"reply": reply, "request_id": request_id}, request_id)


def _response(status: int, body: dict, request_id: str) -> dict:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
        },
        "body": json.dumps(body),
    }
