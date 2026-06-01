"""Minimal litellm wrapper. Trimmed mirror of tau2's ``utils/llm_utils.py``.

No caching / langfuse / fine-tune parsing — just turn our message objects into
a litellm ``completion`` call and return an ``AssistantMessage``.
"""

import json
from typing import Optional

from eops_gym.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
)


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """Convert our message objects to litellm/OpenAI dict format."""
    out: list[dict] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            out.append({"role": "system", "content": msg.content or ""})
        elif isinstance(msg, ToolMessage):
            out.append(
                {"role": "tool", "tool_call_id": msg.id, "content": msg.content or ""}
            )
        else:  # user / assistant
            entry: dict = {"role": msg.role, "content": msg.content or ""}
            if getattr(msg, "tool_calls", None):
                entry["tool_calls"] = [
                    {
                        "id": tc.id or f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for i, tc in enumerate(msg.tool_calls)  # type: ignore[union-attr]
                ]
            out.append(entry)
    return out


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[dict]] = None,
    **kwargs,
) -> AssistantMessage:
    """Call the LLM and return an AssistantMessage.

    Imports litellm lazily so the rest of the package (and the LLM-free tests)
    work without litellm or an API key installed.
    """
    import litellm

    completion_kwargs: dict = {"model": model, "messages": to_litellm_messages(messages)}
    completion_kwargs.update(kwargs)
    if tools:
        completion_kwargs["tools"] = tools

    response = litellm.completion(**completion_kwargs)
    choice = response.choices[0].message

    tool_calls = None
    if getattr(choice, "tool_calls", None):
        tool_calls = [
            ToolCall(
                id=tc.id or "",
                name=tc.function.name,
                arguments=_parse_args(tc.function.arguments),
            )
            for tc in choice.tool_calls
        ]

    return AssistantMessage(content=choice.content, tool_calls=tool_calls)


def _parse_args(arguments) -> dict:
    if isinstance(arguments, dict):
        return arguments
    try:
        return json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return {}
