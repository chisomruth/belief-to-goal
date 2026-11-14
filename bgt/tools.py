"""Dummy tools. The tools live in the scenario data; this only parse a call and return its output."""


import json
import re

def parse_tool_call(text: str):
    """<tool>{"name": ..., "args": {...}}</tool> -> dict, or None if there is no valid call."""
    m = re.search(r"<tool>(.*?)</tool>", text or "", re.S)
    if not m:
        return None
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return call if isinstance(call, dict) and "name" in call else None


def run_tool(tool_list, call: dict) -> str:
    for t in tool_list:
        if t.name == call["name"]:
            return t.output
    return f"Error: unknown tool {call['name']!r}"