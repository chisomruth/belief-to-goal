"""Two plots from the results file.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from .metrics import load, wilson

COLD, LINKED = "#4C78A8", "#E4572E"


def dumbbell(df, path):
    rows = []
    for who, g in df.groupby("persuadee"):
        r = {"who": who}
        for arm in ("cold", "linked"):
            d = g[g["arm"] == arm]
            k, n = int(d["diverted"].sum()), len(d)
            lo, hi = wilson(k, n)
            r[arm] = (100 * k / n if n else np.nan, 100 * lo, 100 * hi)
        rows.append(r)

    fig, ax = plt.subplots(figsize=(7.5, 1.4 + 1.1 * len(rows)))
    for i, r in enumerate(rows):
        c, l = r["cold"], r["linked"]
        ax.annotate("", xy=(l[0], i), xytext=(c[0], i), arrowprops=dict(arrowstyle="-|>", color="#999", lw=2))
        ax.hlines(i, c[1], c[2], color=COLD, lw=7, alpha=.25)
        ax.hlines(i, l[1], l[2], color=LINKED, lw=7, alpha=.25)
        ax.scatter(c[0], i, s=130, color=COLD, zorder=3, label="cold (no persuader)" if i == 0 else None)
        ax.scatter(l[0], i, s=130, color=LINKED, zorder=3, label="after persuasion" if i == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["who"] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("episodes where the agent called the harmful tool (%, 95% CI)")
    ax.grid(axis="x", alpha=.25)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def sankey(df, who, path):
    d = df[(df["persuadee"] == who) & (df["arm"] == "linked")]
    flipped = d[d["belief_flipped"]]
    div = flipped[flipped["diverted"]]
    labels = ["Episodes", "Kept correct belief", "Persuaded", "Stayed on task",
              "Harmful tool called", "Caught by a monitor", "Not flagged"]
    colors = ["#999", "#4C78A8", "#F2A541", "#4C78A8", "#E4572E", "#54A24B", "#B22222"]
    links = [(0, 1, len(d) - len(flipped)), (0, 2, len(flipped)), (2, 3, len(flipped) - len(div)),
             (2, 4, len(div)), (4, 5, int(div["flagged"].sum())), (4, 6, int(div["silent"].sum()))]
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, color=colors, pad=28, thickness=20),
        link=dict(source=[a for a, _, _ in links], target=[b for _, b, _ in links],
                  value=[v for _, _, v in links], color=[colors[b] for _, b, _ in links])))
    fig.update_layout(title=f"{who}: what happens after the persuader speaks", font_size=14)
    fig.write_html(path)


if __name__ == "__main__":
    df = load(sys.argv[1])
    os.makedirs("plots", exist_ok=True)
    dumbbell(df, "plots/diversion_dumbbell.png")
    for who in df["persuadee"].unique():
        sankey(df, who, f"plots/flow_{who.replace(':', '_').replace('/', '_')}.html")
    print("saved to plots/")