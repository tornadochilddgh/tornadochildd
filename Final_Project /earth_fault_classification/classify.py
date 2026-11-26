"""Train and evaluate the OscGrid two-class models on disjoint recordings."""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_cycles import CHANNELS, CYCLE, audit

BASE = [name for name in CHANNELS if name != "IB"]
SPREAD = ["A-C current RMS difference", "voltage RMS range"]


def extract(path):
    cycles = audit(path, BASE)
    keys = ["file_name", "start_sample"]
    chunks = []
    for raw in pd.read_csv(path, usecols=["file_name", "sample", *BASE], chunksize=100000):
        raw["start_sample"] = raw["sample"] // CYCLE * CYCLE
        rows = raw.merge(cycles[keys], on=keys, validate="many_to_one")
        chunks.append(rows)
    samples = pd.concat(chunks, ignore_index=True)
    blocks = samples.groupby(keys)
    assert blocks.size().eq(CYCLE).all() and blocks["sample"].nunique().eq(CYCLE).all()
    samples[BASE] = samples[BASE].pow(2)
    rms = samples.groupby(keys)[BASE].mean().pow(.5)
    rms[SPREAD[0]] = (rms.IA - rms.IC).abs()
    volts = rms[["UA BB", "UB BB", "UC BB"]]
    rms[SPREAD[1]] = volts.max(axis=1) - volts.min(axis=1)
    return cycles.merge(rms, on=keys, validate="one_to_one")


def candidates(name):
    if name == "logistic":
        return [(f"C={c}", make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000)))
                for c in (.1, 1, 10)]
    return [(f"depth={depth}", RandomForestClassifier(n_estimators=120, max_depth=depth,
             random_state=443, n_jobs=-1)) for depth in (3, 6, None)]


def tune(train, cols, name):
    flags = train.groupby("recording").label.max()
    folds = StratifiedKFold(4, shuffle=True, random_state=443)
    best = -1
    for param, estimator in candidates(name):
        prob = np.zeros(len(train))
        for it, iv in folds.split(flags.index, flags):
            fit = train.recording.isin(flags.index[it])
            val = train.recording.isin(flags.index[iv])
            model = clone(estimator).fit(train.loc[fit, cols], train.loc[fit, "label"])
            prob[val] = model.predict_proba(train.loc[val, cols])[:, 1]
        for threshold in (.3, .4, .5, .6, .7):
            score = f1_score(train.label, prob >= threshold)
            if score > best:
                best = score
                chosen = (param, threshold, estimator)
    return chosen


def evaluate(data):
    flags = data.groupby("recording").label.max()
    train_ids, test_ids = train_test_split(flags.index, test_size=.25, random_state=443, stratify=flags)
    train = data[data.recording.isin(train_ids)]
    test = data[data.recording.isin(test_ids)].copy()
    assert set(train.recording).isdisjoint(test.recording)
    data.assign(split=np.where(data.recording.isin(test_ids), "test", "train")).to_csv("cycles.csv",index=False)
    results = []
    for name in ("logistic", "forest"):
        for size, cols in ((5, BASE), (7, BASE + SPREAD)):
            param, threshold, estimator = tune(train, cols, name)
            estimator.fit(train[cols], train.label)
            key = f"{name}_{size}"
            pred = estimator.predict_proba(test[cols])[:, 1] >= threshold
            test[key] = pred.astype(int)
            tn, fp, fn, tp = confusion_matrix(test.label, pred, labels=[0,1]).ravel()
            results.append(dict(model=key, parameter=param, threshold=threshold,
                                precision=precision_score(test.label,pred,zero_division=0),
                                recall=recall_score(test.label,pred), f1=f1_score(test.label,pred),
                                tn=tn, fp=fp, fn=fn, tp=tp))
    report = pd.DataFrame(results).set_index("model")
    # Paired bootstrap: sample whole test recordings within fault-presence strata.
    rng = np.random.default_rng(443)
    strata = [g.index.to_numpy() for _, g in flags.loc[test_ids].groupby(flags.loc[test_ids])]
    draws = np.concatenate([rng.choice(s,(5000,len(s)),replace=True) for s in strata],axis=1)
    deltas = []
    boot = {}
    for key in report.index:
        by_rec = pd.DataFrame(dict(tp=(test.label.eq(1) & test[key].eq(1)).astype(int),
                                   errors=test.label.ne(test[key]).astype(int),
                                   recording=test.recording)).groupby("recording").sum()
        counts = by_rec.to_numpy()[by_rec.index.get_indexer(draws.ravel())].reshape(*draws.shape,2).sum(axis=1)
        boot[key] = 2*counts[:,0]/(2*counts[:,0]+counts[:,1])
        report.loc[key, ["low95","high95"]] = np.quantile(boot[key],[.025,.975])
    for name in ("logistic","forest"):
        a,b = name+"_5",name+"_7"
        lo,hi = np.quantile(boot[b]-boot[a],[.025,.975])
        deltas.append(dict(model=name,delta_f1=report.loc[b,"f1"]-report.loc[a,"f1"],low95=lo,high95=hi))
    pd.DataFrame(deltas).to_csv("ablation.csv",index=False)
    report.to_csv("results.csv")
    test.to_csv("predictions.csv",index=False)
    print(f"{len(train_ids)} training recordings; {len(test_ids)} test recordings, "
          f"{int(flags.loc[test_ids].sum())} containing a fault")
    print(report.round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home()/".cache")
    parser.add_argument("--csv",type=Path,default=cache/"oscgrid/v6/labeled.csv")
    evaluate(extract(parser.parse_args().csv))
