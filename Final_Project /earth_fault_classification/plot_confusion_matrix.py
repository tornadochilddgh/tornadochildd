"""Plot held-out cycle counts and row percentages for the seed-443 random forest."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


def main():
    root = Path(__file__).resolve().parent
    predictions = pd.read_csv(root / "predictions.csv", usecols=["label", "forest_5"])
    if predictions.empty or not predictions.isin([0, 1]).all().all():
        raise ValueError("Expected binary labels and predictions for held-out cycles.")
    counts = confusion_matrix(predictions.label, predictions.forest_5, labels=[0, 1])
    if (counts.sum(axis=1) == 0).any():
        raise ValueError("Both true classes must be present for row percentages.")
    fractions = counts / counts.sum(axis=1, keepdims=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.45, 2.7), layout="constrained")
    ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
    for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)):
        color = "white" if fractions[i, j] > .6 else "#182634"
        ax.text(j, i-.10, str(counts[i,j]), ha="center", va="center",
                size=13, weight="bold", color=color)
        ax.text(j, i+.16, f"{fractions[i,j]:.1%}", ha="center", va="center", color=color)
    labels = ["Switching", "Earth fault"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
           xlabel="Predicted class", ylabel="Actual class")
    ax.tick_params(length=0, pad=4)
    output = root / "figures"
    output.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"forest5_confusion_matrix.{ext}", dpi=300)
    plt.close(fig)
    print(counts)
    print("Saved figures/forest5_confusion_matrix.pdf and .png")


if __name__ == "__main__":
    main()
