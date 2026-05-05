"""Agent loop built on Bedrock Converse API.

Flow per user message:
    1. Append user message to conversation history.
    2. Call Bedrock with the full conversation + tools.
    3. If stopReason == 'tool_use', execute the tool calls, append results,
       and loop back to step 2.
    4. If stopReason != 'tool_use', extract the text and return.

A MAX_TURNS safety limit prevents a runaway tool loop from burning tokens.
"""
import json
import os
import time

import boto3

from obs import log_event
from system_prompt import SYSTEM_PROMPT
from tools import TOOLS, dispatch_tool

REGION = os.environ.get("AWS_REGION", "us-east-2")
MODEL_ID = os.environ.get("MODEL_ID", "us.anthropic.claude-sonnet-4-6")
MAX_TURNS = int(os.environ.get("MAX_AGENT_TURNS", "24"))
MAX_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "10000"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))

_bedrock = boto3.client("bedrock-runtime", region_name=REGION)


def run_agent(
    user_id: str,
    user_message: str,
    history: list[dict],
    request_id: str,
) -> tuple[str, list[dict]]:
    """Run the agent loop until the model stops calling tools.

    Returns (final_text, updated_history).
    """
    # Defensive: if history contains any orphan tool_use or tool_result
    # blocks (from a prior version that didn't sanitize on save, or from a
    # trim cap that sliced through a round-trip), drop them so Bedrock
    # doesn't reject the whole conversation.
    history = _strip_dangling_tool_use(list(history))
    messages = history + [
        {"role": "user", "content": [{"text": user_message}]}
    ]

    for turn in range(MAX_TURNS):
        t0 = time.time()
        try:
            resp = _bedrock.converse(
                modelId=MODEL_ID,
                messages=messages,
                system=[{"text": SYSTEM_PROMPT}],
                toolConfig={"tools": TOOLS},
                inferenceConfig={
                    "maxTokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                },
            )
        except Exception as e:
            log_event(
                "model_error",
                {"request_id": request_id, "turn": turn, "error": str(e)},
            )
            raise

        latency_ms = int((time.time() - t0) * 1000)
        usage = resp.get("usage", {})
        log_event(
            "model_call",
            {
                "request_id": request_id,
                "turn": turn,
                "latency_ms": latency_ms,
                "stop_reason": resp.get("stopReason"),
                "input_tokens": usage.get("inputTokens"),
                "output_tokens": usage.get("outputTokens"),
            },
        )

        assistant_msg = resp["output"]["message"]
        messages.append(assistant_msg)

        if resp.get("stopReason") != "tool_use":
            text = "".join(
                block["text"]
                for block in assistant_msg["content"]
                if "text" in block
            )
            return text, messages

        # Execute every tool_use block in this assistant message, in order.
        tool_results = []
        for block in assistant_msg["content"]:
            if "toolUse" not in block:
                continue
            tu = block["toolUse"]
            tool_t0 = time.time()
            status = "success"
            try:
                result = dispatch_tool(tu["name"], tu.get("input", {}), user_id)
            except Exception as e:
                result = {"error": str(e)}
                status = "error"
            tool_latency = int((time.time() - tool_t0) * 1000)

            log_event(
                "tool_call",
                {
                    "request_id": request_id,
                    "turn": turn,
                    "tool": tu["name"],
                    "input": tu.get("input", {}),
                    "status": status,
                    "latency_ms": tool_latency,
                    "output_preview": json.dumps(result, default=str)[:500],
                },
            )

            tool_results.append(
                {
                    "toolResult": {
                        "toolUseId": tu["toolUseId"],
                        "content": [{"text": json.dumps(result, default=str)}],
                        "status": status,
                    }
                }
            )

        messages.append({"role": "user", "content": tool_results})

    # Exhausted turns — return a graceful fallback so the user isn't stuck.
    log_event("max_turns_exceeded", {"request_id": request_id})
    return (
        "I got stuck working through that request. Could you try rephrasing "
        "or breaking it into smaller steps?",
        _strip_dangling_tool_use(messages),
    )


def _strip_dangling_tool_use(messages: list[dict]) -> list[dict]:
    """Remove ANY assistant tool_use block whose toolUseId doesn't appear
    in a following tool_result, plus any tool_result whose toolUseId doesn't
    have a preceding tool_use.

    Bedrock's Converse API rejects any conversation containing a toolUse
    without a corresponding toolResult (and vice versa). Corruption can land
    in stored history three ways:

      1. MAX_TURNS hit mid-loop — the trailing assistant message has an
         orphan tool_use (handled by the original tail-only check).
      2. save_history's trim cap slices through a tool round-trip, leaving
         the head of history with an orphan tool_use because the matching
         tool_result got cut. This is what was actually breaking requests
         in production — the original sanitizer only checked the tail and
         missed this case.
      3. Mid-conversation gap — shouldn't happen, but defending against it
         costs nothing and catches future bugs.

    Strategy: collect all tool_use IDs and all tool_result IDs across the
    entire conversation. For each message, drop any block whose ID isn't
    matched on the other side. Drop messages that become empty.
    """
    if not messages:
        return messages

    # First pass: collect all matched IDs.
    tool_use_ids = set()
    tool_result_ids = set()
    for msg in messages:
        for block in msg.get("content", []) or []:
            if "toolUse" in block:
                tu_id = block["toolUse"].get("toolUseId")
                if tu_id:
                    tool_use_ids.add(tu_id)
            elif "toolResult" in block:
                tr_id = block["toolResult"].get("toolUseId")
                if tr_id:
                    tool_result_ids.add(tr_id)

    matched_ids = tool_use_ids & tool_result_ids
    has_orphans = (tool_use_ids | tool_result_ids) != matched_ids
    if not has_orphans:
        return messages

    # Second pass: filter out orphan blocks. Drop messages that become empty.
    cleaned = []
    for msg in messages:
        kept_blocks = []
        for block in msg.get("content", []) or []:
            if "toolUse" in block:
                tu_id = block["toolUse"].get("toolUseId")
                if tu_id in matched_ids:
                    kept_blocks.append(block)
            elif "toolResult" in block:
                tr_id = block["toolResult"].get("toolUseId")
                if tr_id in matched_ids:
                    kept_blocks.append(block)
            else:
                # text and other block types pass through untouched
                kept_blocks.append(block)
        if kept_blocks:
            cleaned.append({**msg, "content": kept_blocks})

    return cleaned
