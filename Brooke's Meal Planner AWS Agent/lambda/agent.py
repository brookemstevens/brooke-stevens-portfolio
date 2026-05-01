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
    # Defensive: if history ends with a dangling tool_use (from a prior
    # version that didn't sanitize on save), drop it so Bedrock doesn't
    # reject the whole conversation.
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
    """Remove a trailing assistant message whose tool_use blocks have no
    matching tool_results in the next message.

    Bedrock's Converse API rejects any conversation containing a toolUse
    without a corresponding toolResult. This can happen when the agent loop
    terminates (e.g. MAX_TURNS hit) right after the model requested a tool
    call but before we executed it. Saving that state would permanently
    corrupt the conversation — the next request would immediately
    ValidationException on load.
    """
    if not messages:
        return messages
    last = messages[-1]
    if last.get("role") != "assistant":
        return messages
    has_tool_use = any("toolUse" in block for block in last.get("content", []))
    if has_tool_use:
        return messages[:-1]
    return messages
