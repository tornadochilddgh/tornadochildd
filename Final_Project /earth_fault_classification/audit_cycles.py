"""Count eligible OscGrid v6 busbar cycles for the proposed two-class experiment."""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ["IA", "IB", "IC", "UA BB", "UB BB", "UC BB"]
# Verified against all 480 CFG headers in v6's Labeled_raw_v1.1 archive.
CYCLE = 1600 // 50
LABELS = {0: "Routine switching", 1: "Single-phase earth fault"}


def belongs(column, root):
    return column == root or column.startswith(root + "_")


def eligible_samples(chunk, channels):
    names = chunk["file_name"].str.extract(r"^([0-9a-fA-F]{32})_Bus (\d+)(?:\s+_event N\d+)?$")
    flags = chunk.filter(regex=r"^ML_").eq(1)

    def active(root):
        return flags[[c for c in flags if belongs(c, root)]].any(axis=1)

    switching, abnormal, critical = (active(code) for code in ("ML_1", "ML_2", "ML_3"))
    other = [c for c in flags if c.startswith("ML_2_") and not belongs(c, "ML_2_1")]
    fault = active("ML_2_1") & ~flags[other].any(axis=1) & ~switching & ~critical
    routine = switching & ~abnormal & ~critical
    valid = (fault | routine) & names[0].notna() & np.isfinite(chunk[channels]).all(axis=1)
    rows = pd.DataFrame({"recording": names[0], "bus": names[1], "file_name": chunk.file_name, "sample": chunk["sample"],
                         "label": fault.astype(int)})
    return rows.loc[valid]


def audit(path, channels):
    header = pd.read_csv(path, nrows=0)
    cols = ["file_name", "sample", *channels, *header.filter(regex=r"^ML_").columns]
    samples = pd.concat([eligible_samples(c, channels) for c in pd.read_csv(path, usecols=cols,
                                                                  chunksize=100000)])
    samples["cycle"] = samples["sample"] // CYCLE
    keys = ["recording", "bus", "file_name", "cycle"]
    blocks = samples.groupby(keys).agg(n=("sample", "size"), unique=("sample", "nunique"),
                                       classes=("label", "nunique"), label=("label", "first"))
    full = blocks.query("n == @CYCLE and unique == @CYCLE and classes == 1").reset_index()
    full["start_sample"] = full.pop("cycle") * CYCLE
    return full[["recording", "bus", "file_name", "start_sample", "label"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(os.environ.get("XDG_CACHE_HOME") or Path.home()/".cache")/"oscgrid/v6/labeled.csv"
    parser.add_argument("--csv", type=Path, default=default)
    parser.add_argument("--five-channel", action="store_true",
                        help="Check feasibility when phase-B current is optional")
    args = parser.parse_args()
    channels = [c for c in CHANNELS if c != "IB"] if args.five_channel else CHANNELS
    windows = audit(args.csv, channels)
    output = "eligible_cycles_five_channel.csv" if args.five_channel else "eligible_cycles.csv"
    windows.to_csv(output, index=False)
    for label, name in LABELS.items():
        subset = windows[windows.label == label]
        print(f"{name}: {subset.recording.nunique()} recordings, {len(subset)} complete cycles")
    print("Recordings contributing to both classes:",
          windows.groupby("recording").label.nunique().eq(2).sum())
