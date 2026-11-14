"""reading and writing files."""


import json
import os

from . import schema


def read_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def load_scenarios(path: str) -> list:
    """Read the task suite (.jsonl) and validate each scenario."""
    return [schema.Scenario.model_validate(s) for s in read_jsonl(path)]


def done_episode_ids(path: str) -> set:
    """skips episodes already saved in the results file when a run is restarted."""
    if not os.path.exists(path):
        return set()
    return {row["episode_id"] for row in read_jsonl(path)}