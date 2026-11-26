# Earth fault versus routine switching in OscGrid v6

From this directory run the following using the same project environment and cached labeled.csv as the four-class reproduction:

    uv run python classify.py

Alternatively supply a path with --csv. The script generates cycles.csv (features and split), predictions.csv (held-out cycles), results.csv and ablation.csv. Label 1 denotes an annotated single-phase earth fault, and 0 denotes operational switching. Source: https://doi.org/10.6084/m9.figshare.28465427.v6.

The audit requires 32 distinct consecutive indices aligned to multiples of 32 within a recording, bus and event segment, with a consistent eligible label. Each of the five channels must be finite at every sample. It uses the released CSV values and computes RMS per cycle. Logistic regression standardizes features using its training data. The two extra features are the absolute A–C current RMS difference and the range of three busbar voltage RMS values. No phase-B current is estimated.

The holdout uses seed 443, stratifies recordings by presence of eligible fault cycles and assigns both buses and all cycles from each recording to the same partition. Four recording-disjoint training folds select hyperparameters and the probability threshold by pooled validation fault F1. Each feature set is tuned separately. Reported scores are cycle-weighted on the same holdout. Percentile intervals resample whole test recordings within fault-presence strata (5,000 paired draws); these intervals condition on the fitted models and this split. This is one held-out experiment. The archive includes the comparison and paired ablation results. Running the script regenerates them locally.
