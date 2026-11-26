"""Reproduce the released OscGrid MLP defaults on version 6 labeled.csv."""
import argparse
import copy
import hashlib
import os
import shutil
import tempfile
from urllib.request import urlopen
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

CHANNELS = ["IA", "IC", "UA BB", "UB BB", "UC BB"]
CLASSES = ["No event", "Operational switching", "Abnormal events", "Fault events"]


def cached_csv():
    """Download the pinned Figshare archive once and publish only a complete CSV."""
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    cache = cache / "oscgrid" / "v6"
    dest = cache / "labeled.csv"
    size = 383963307
    if dest.is_file() and dest.stat().st_size == size:
        print(f"Using cached data: {dest}", flush=True)
        return dest

    import py7zr
    cache.mkdir(parents=True, exist_ok=True)
    url = "https://ndownloader.figshare.com/files/60075722"
    digest = "f46903b45cfc3232cef80f3254c34f8bcfcf11a47fae307e5e253d250ec6173e"
    print("Downloading OscGrid v6 labeled data from Figshare...", flush=True)
    with tempfile.TemporaryDirectory(dir=cache) as temp:
        tmp = Path(temp)
        archive = tmp / "labeled.7z"
        with urlopen(url, timeout=90) as src, archive.open("wb") as out:
            shutil.copyfileobj(src, out)
        with archive.open("rb") as src:
            if hashlib.file_digest(src, "sha256").hexdigest() != digest:
                raise ValueError("Figshare download failed the SHA-256 check. Run again.")
        with py7zr.SevenZipFile(archive, "r") as bundle:
            bundle.extract(path=tmp, targets=["labeled.csv"])
        csv = tmp / "labeled.csv"
        if csv.stat().st_size != size:
            raise ValueError("Extracted CSV size differs from the version 6 release.")
        csv.replace(dest)
    return dest


def read_data(path):
    def selected(col):
        return col in ["file_name", *CHANNELS] or col.startswith(("ML_1", "ML_2", "ML_3"))

    parts = []
    for chunk in pd.read_csv(path, usecols=selected, chunksize=100000):
        chunk = chunk.dropna(subset=CHANNELS)
        part = chunk[["file_name", *CHANNELS]].copy()
        part["target"] = 0
        for cls in (1, 2, 3):
            part.loc[chunk.filter(regex=f"^ML_{cls}").eq(1).any(axis=1), "target"] = cls
        parts.append(part)
    data = pd.concat(parts, ignore_index=True)
    data["block"] = (data.file_name.ne(data.file_name.shift()) |
                     data.target.ne(data.target.shift())).cumsum()
    return data


def prepare(data, seed):
    blocks = data.groupby("block", sort=False).agg(
        name=("file_name", "first"), target=("target", "first"), size=("target", "size"))
    blocks = blocks[blocks["size"] >= 32].sample(frac=1, random_state=seed).copy()
    blocks["split"] = "train"
    for _, group in blocks.groupby("target"):
        limits = {"val": .10 * group["size"].sum(), "test": .30 * group["size"].sum()}
        for block, size in group["size"].items():
            for split in limits:
                if size <= limits[split]:
                    limits[split] -= size
                    blocks.loc[block, "split"] = split
                    break
    subset = data[data.block.isin(blocks.index)].copy()
    split = subset.block.map(blocks.split)
    scaler = StandardScaler().fit(subset.loc[split == "train", CHANNELS])
    signals = scaler.transform(subset[CHANNELS]).astype("float32")
    y = subset.target.to_numpy()
    weights = 1 / np.bincount(y[split == "train"])
    windows = {}
    starts = np.flatnonzero(~subset.block.duplicated())
    for name in ("train", "val", "test"):
        indices = [np.arange(start, start + row["size"] - 31, 10)
                   for start, (_, row) in zip(starts, blocks.sort_index().iterrows())
                   if row["split"] == name]
        idx = np.concatenate(indices)
        windows[name] = (signals[idx[:, None] + np.arange(32)].reshape(-1, 160), y[idx])
    recs = {s: set(blocks.loc[blocks.split == s, "name"].str.split("_Bus").str[0])
            for s in windows}
    print("Seed", seed, "train/test shared recording IDs", len(recs["train"] & recs["test"]),
          "of", len(recs["test"]), "test recording IDs", flush=True)
    print("Window counts", {s:len(v[1]) for s,v in windows.items()}, flush=True)
    return windows, weights / weights.sum()


def train(windows, weights, seed):
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(160,128), nn.BatchNorm1d(128), nn.ReLU(),
                        nn.Dropout(.3), nn.Linear(128,4))
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001)
    loss = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    data = {s:(torch.from_numpy(x),torch.from_numpy(y)) for s,(x,y) in windows.items()}
    def predictions(split):
        net.eval()
        with torch.no_grad():
            return torch.cat([net(x).argmax(1) for x in data[split][0].split(1024)])
    best = -1.
    for epoch in range(1,101):
        net.train()
        x,y = data["train"]
        for idx in torch.randperm(len(y)).split(128):
            optimizer.zero_grad()
            loss(net(x[idx]), y[idx]).backward()
            nn.utils.clip_grad_norm_(net.parameters(),10)
            optimizer.step()
        acc = (predictions("val") == data["val"][1]).float().mean().item()
        if acc > best:
            best, best_epoch, checkpoint = acc, epoch, copy.deepcopy(net.state_dict())
        if epoch - best_epoch >= 10:
            break
    net.load_state_dict(checkpoint)
    p,r,f,n = precision_recall_fscore_support(data["test"][1], predictions("test"),
                                              labels=range(4), zero_division=0)
    print("Best epoch",best_epoch,"F1",f.round(3),flush=True)
    return pd.DataFrame(dict(seed=seed, best_epoch=best_epoch, epochs=epoch,
                             event=CLASSES, precision=p, recall=r, f1=f, windows=n))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, nargs="?", help="Optional local labeled.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=[443,444,445,446,447])
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    data = read_data(args.csv or cached_csv())
    reports = []
    for seed in args.seeds:
        windows, weights = prepare(data, seed)
        if not args.audit_only:
            reports.append(train(windows, weights, seed))
            pd.concat(reports).to_csv("runs.csv",index=False)
