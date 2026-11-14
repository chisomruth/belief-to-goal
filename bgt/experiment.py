"""runs MANY conversations and saves them."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from . import schema
from . import data
from . import chat_episodes

_write_lock = threading.Lock()


def run_all(cfg: dict) -> None:
    scenarios = data.load_scenarios(cfg["scenarios"])
    out = cfg["out"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    finished = data.done_episode_ids(out)

    for arm in cfg["arms"]:
        assert arm in schema.ARMS, f"unknown arm {arm!r}; choose from {schema.ARMS}"

    for persuadee in cfg["persuadees"]:
        print(f"\n target model: {persuadee}", flush=True)
        jobs = [(s, arm, rep)
                for s in scenarios for arm in cfg["arms"] for rep in range(cfg["reps"])
                if f"{persuadee}|{s.id}|{arm}|{rep}" not in finished]
        print(f"{len(jobs)} conversations to run", flush=True)

        with ThreadPoolExecutor(cfg["workers"]) as pool:
            futures = [pool.submit(chat_episodes.run_episode, cfg, s, arm, persuadee, rep) for s, arm, rep in jobs]
            for f in tqdm(as_completed(futures), total=len(futures)):
                try:
                    row = f.result()
                except Exception as e:
                    print("conversation failed:", e, flush=True)
                    continue
                with _write_lock:
                    data.append_jsonl(out, row.model_dump())