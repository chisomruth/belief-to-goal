"""this turns the saved conversations into tables."""

import sys
import numpy as np
import pandas as pd
from . import data


def wilson(k, n, z=1.96):
    """95% confidence interval for rate k/n."""
    if n == 0:
        return np.nan, np.nan
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def load(path):
    df = pd.DataFrame(data.read_jsonl(path))
    total = len(df)
    df = df[df["belief_valid"].fillna(True).astype(bool)].copy()
    print(f"{len(df)} usable conversations out of {total} (dropped: agent started with the wrong belief)")
    for c in ["belief_flipped", "diverted", "flag_instruction", "flag_factcheck_msg"]:
        df[c] = df[c].fillna(False).astype(bool)
    df["flagged"] = df["flag_instruction"] | df["flag_factcheck_msg"]
    df["silent"] = df["diverted"] & ~df["flagged"]
    return df


def summarize(path):
    df = load(path)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    rows = []
    for (who, arm), d in df.groupby(["persuadee", "arm"]):
        n, k = len(d), int(d["diverted"].sum())
        lo, hi = wilson(k, n)
        r = {"persuadee": who, "arm": arm, "n": n,
             "diverted_%": 100 * k / n, "ci_lo": 100 * lo, "ci_hi": 100 * hi}
        if arm == "linked":
            p = d[d["belief_flipped"]]
            r["persuaded_%"] = 100 * d["belief_flipped"].mean()
            r["diverted_if_persuaded_%"] = 100 * p["diverted"].mean() if len(p) else np.nan
            r["flag_instruction_%"] = 100 * p["flag_instruction"].mean() if len(p) else np.nan
            r["flag_factcheck_msg_%"] = 100 * p["flag_factcheck_msg"].mean() if len(p) else np.nan
            r["silent_diverted_%"] = 100 * d["silent"].mean()
        rows.append(r)
    table = pd.DataFrame(rows)
    print("\n== results ==")
    print(table.round(1).to_string(index=False))
    return table


if __name__ == "__main__":
    summarize(sys.argv[1])