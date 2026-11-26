"""Summarize the five independent seeded runs, using Table 2 as the reference."""
import numpy as np
import pandas as pd
from scipy.stats import t

published = {"No event": .84, "Operational switching": .45,
             "Abnormal events": .81, "Fault events": .79}
runs = pd.read_csv("runs.csv")
rows = []
for event, reference in published.items():
    group = runs[runs.event == event]
    if len(group) != 5 or group.seed.nunique() != 5:
        raise ValueError(f"Expected five distinct runs for {event}")
    scores = group.f1
    mean = scores.mean()
    radius = t.ppf(.975, len(scores)-1) * scores.std(ddof=1) / np.sqrt(len(scores))
    rows.append(dict(event=event, published_f1=reference, mean_f1=mean,
                     delta=mean-reference, ci_low=mean-radius, ci_high=mean+radius))
summary = pd.DataFrame(rows)
summary.to_csv("summary.csv", index=False)
print(summary.round(3).to_string(index=False))
