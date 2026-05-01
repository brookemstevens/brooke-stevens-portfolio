"""Structured logging for CloudWatch.

Every event is a single JSON line, so CloudWatch Logs Insights can query by
field: e.g. `fields @timestamp, tool, latency_ms | filter event = 'tool_call'`.
"""
import json
import sys
import time


def log_event(event_type: str, payload: dict) -> None:
    """Emit a structured log line that CloudWatch will ingest.

    Uses print() because Lambda captures stdout into CloudWatch automatically.
    """
    record = {"ts": time.time(), "event": event_type, **payload}
    # default=str handles Decimal, datetime, and other non-JSON types gracefully
    print(json.dumps(record, default=str), file=sys.stdout, flush=True)
