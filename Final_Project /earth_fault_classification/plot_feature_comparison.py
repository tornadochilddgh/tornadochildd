"""Plot fault F1 and the paired effect of the two derived features."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw():
    root = Path(__file__).resolve().parent
    data = root / "repeated_training"
    scores = pd.read_csv(data / "summary.csv", index_col="model")
    changes = pd.read_csv(data / "ablation.csv", index_col="model")
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, (left, right) = plt.subplots(1, 2, figsize=(7, 2.65), layout="constrained",
                                      gridspec_kw={"width_ratios": [1.4, 1]})
    labels = ["Logistic regression", "Random forest"]
    for y, model in enumerate(("logistic", "forest")):
        for size, offset, marker, color in [(5, .15, "o", "#23618a"), (7, -.15, "s", "#a84b19")]:
            score = scores.loc[f"{model}_{size}", "mean_f1"]
            left.plot(score, y + offset, marker=marker, color=color, linestyle="none",
                      label=f"{size} features" if y == 0 else None)
            left.annotate(f"{score:.3f}", (score, y + offset), xytext=(6, 0),
                          textcoords="offset points", va="center")
        row = changes.loc[model]
        diff = scores.loc[f"{model}_7", "mean_f1"] - scores.loc[f"{model}_5", "mean_f1"]
        if not np.isclose(diff, row.delta_f1):
            raise ValueError(f"Inconsistent summary and ablation files for {model}")
        right.hlines(y, row.low95, row.high95, color="#335567", lw=2)
        right.plot([row.low95, row.high95], [y, y], "|", color="#335567")
        right.plot(diff, y, "D", color="#213947", markersize=5)
        right.annotate(f"{diff:+.3f}", (diff, y), xytext=(0, 9),
                       textcoords="offset points", ha="center")
    left.set(yticks=[0, 1], yticklabels=labels, xlabel="Mean fault F1", xlim=(.76,1.01))
    left.legend(loc="upper left", bbox_to_anchor=(0,1.19), frameon=False,
                ncols=2, handletextpad=.2, columnspacing=.8)
    right.axvline(0, color="#7d8688", lw=1, ls="--", zorder=0)
    right.set(yticks=[], xlabel="F1 change (7 minus 5)", xlim=(-.12,.31))
    right.set_title("Paired 95% bootstrap intervals", fontsize=9)
    for ax in (left, right):
        ax.set_ylim(1.45, -.45)
        ax.spines[["right","top","left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e7eaeb", lw=.6)
        ax.set_axisbelow(True)
    dest = root / "figures"
    dest.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(dest / f"feature_comparison.{ext}", dpi=300)
    plt.close(fig)
    print("Saved figures/feature_comparison.pdf and figures/feature_comparison.png")


if __name__ == "__main__":
    draw()
