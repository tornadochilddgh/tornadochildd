"""Repeat the final fits on the original recording split, holding tuning fixed."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from classify import BASE, SPREAD, candidates

SEEDS = range(443, 448)


def main():
    folder = Path(__file__).resolve().parent
    data = pd.read_csv(folder / "cycles.csv", float_precision="round_trip")
    settings = pd.read_csv(folder / "results.csv").set_index("model")
    train = data[data.split == "train"]
    test = data[data.split == "test"].copy()
    if set(train.recording) & set(test.recording):
        raise ValueError("A recording occurs in both partitions.")
    flags = test.groupby("recording").label.max()
    rng = np.random.default_rng(443)
    strata = [g.index.to_numpy() for _, g in flags.groupby(flags)]
    draws = np.concatenate([rng.choice(s, (5000, len(s)), replace=True) for s in strata], axis=1)
    out = folder / "repeated_training"
    out.mkdir(exist_ok=True)
    scores, boot = [], {}
    for key, row in settings.iterrows():
        name, size = key.split("_")
        cols = BASE if size == "5" else BASE + SPREAD
        distributions = []
        for seed in SEEDS:
            model = dict(candidates(name))[row.parameter]
            if name == "forest":
                model.set_params(random_state=seed)
            model.fit(train[cols], train.label)
            pred = model.predict_proba(test[cols])[:, 1] >= row.threshold
            f1 = f1_score(test.label, pred)
            if seed == 443 and not np.isclose(f1, row.f1):
                raise ValueError(f"Seed 443 F1 for {key} differs from results.csv: {f1} vs {row.f1}")
            scores.append(dict(model=key, seed=seed, f1=f1,
                               precision=precision_score(test.label,pred,zero_division=0),
                               recall=recall_score(test.label,pred)))
            counts = pd.DataFrame(dict(recording=test.recording, tp=test.label.eq(1)&pred,
                                       errors=test.label.ne(pred))).groupby("recording").sum()
            totals = counts.to_numpy()[counts.index.get_indexer(draws.ravel())].reshape(
                *draws.shape, 2).sum(axis=1)
            distributions.append(2 * totals[:,0] / (2 * totals[:,0] + totals[:,1]))
        boot[key] = np.mean(distributions, axis=0)
    runs = pd.DataFrame(scores)
    runs.to_csv(out / "runs.csv", index=False)
    summary = runs.groupby("model").agg(mean_f1=("f1","mean"), min_f1=("f1","min"), max_f1=("f1","max"))
    for key in summary.index:
        summary.loc[key, ["low95","high95"]] = np.quantile(boot[key],[.025,.975])
    summary.to_csv(out / "summary.csv")
    differences = []
    for name in ("logistic","forest"):
        a,b = name+"_5",name+"_7"
        lo,hi = np.quantile(boot[b]-boot[a],[.025,.975])
        differences.append(dict(model=name, delta_f1=summary.loc[b,"mean_f1"]-summary.loc[a,"mean_f1"],
                                low95=lo,high95=hi))
    pd.DataFrame(differences).to_csv(out / "ablation.csv",index=False)
    print(summary.round(3).to_string())
    print("Saved repeated_training/runs.csv, summary.csv and ablation.csv")


if __name__ == "__main__":
    main()
