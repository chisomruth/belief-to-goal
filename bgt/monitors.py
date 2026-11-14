""" The monitors. Each one reads a persuader message and returns True (flagged) / False / None (unreadable). """


from . import prompts, models


def _ask_json(spec, prompt):
    text = models.call_model(spec, [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=400)
    data = prompts.parse_json(text)
    if not isinstance(data, dict) or "flag" not in data:
        return None
    return bool(data["flag"])


def instruction_monitor(spec, message):
    return _ask_json(spec, prompts.instruction_monitor_prompt(message))


def factcheck_monitor(spec, message, scenario):
    return _ask_json(spec, prompts.factcheck_prompt(message, scenario))