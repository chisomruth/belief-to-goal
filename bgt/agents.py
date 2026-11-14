""" The two characters in this pipeline/experiment.

  persuadee_* : Agent A, the TARGET. It has a task, a system prompt, and dummy tools.
  persuader_* : Agent B, the PEER. It writes messages trying to change A's belief, then A's goal.
"""
from . import prompts
from . import models
from . import tools


def persuadee_turn(spec, history, scenario, temperature, max_tokens):
    """Agent reply. Returns (reply_text, thinking_text)."""
    return models.call_model_full(spec, history, system=prompts.persuadee_system_prompt(scenario.tools),
                                  temperature=temperature, max_tokens=max_tokens)


def persuadee_act(spec, history, scenario, temperature, max_tokens, max_steps=4):
    """Let the agent reply and use tools until it replies without a tool call (or max_steps).
    Appends everything to `history`. Returns (reasoning_per_turn, names_of_tools_called)."""
    reasoning, calls = [], []
    for _ in range(max_steps):
        text, thinking = persuadee_turn(spec, history, scenario, temperature, max_tokens)
        history.append({"role": "assistant", "content": text})
        reasoning.append((thinking + "\n" + text).strip())
        call = tools.parse_tool_call(text)
        if call is None:
            break
        calls.append(call["name"])
        history.append({"role": "user", "content": f"[TOOL RESULT] {tools.run_tool(scenario.tools, call)}"})
    return reasoning, calls


def persuader_belief_message(spec, belief, r0, reference, temperature, max_tokens):
    prompt = prompts.persuader_belief_prompt(belief, r0[:600], reference)
    text = models.call_model(spec, [{"role": "user", "content": prompt}],
                             temperature=temperature, max_tokens=max_tokens)
    return prompts.parse_message(text)


def persuader_goal_message(spec, scenario, convo, premise, temperature, max_tokens):
    prompt = prompts.persuader_goal_prompt(scenario, convo, premise)
    text = models.call_model(spec, [{"role": "user", "content": prompt}],
                             temperature=temperature, max_tokens=max_tokens)
    return prompts.parse_message(text)