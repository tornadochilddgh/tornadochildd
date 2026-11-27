# OscGrid four-class MLP benchmark

This experiment compares five CPU training runs with Table 2 of [Evdakov et al.](https://doi.org/10.1038/s41597-026-06587-8). Download [labeled.7z from OscGrid v6](https://ndownloader.figshare.com/files/60075722), extract labeled.csv, and keep the CSV outside the Git repository. [Dataset record and attribution](https://doi.org/10.6084/m9.figshare.28465427.v6).

## Run in VS Code

From this folder, with NumPy, pandas, scikit-learn, and PyTorch installed in the project environment, run the following in the terminal. Replace the path with the location of your extracted CSV.

```bash
uv run python reproduce.py /path/to/labeled.csv
```

For the split audit alone, append `--seeds 443 --audit-only`. The full run writes runs.csv in the current folder. Run `uv run python summarize.py` there to regenerate the summary. Tested with Python 3.12, CPU PyTorch 2.7.1, pandas 2.2.3, NumPy 2.3.5, SciPy 1.17.0 and scikit-learn 1.8.0.

## Reproduction choices

- Reference implementation: [training script](https://github.com/AIRI-Institute/oscgrid/blob/4cbdf81f9a05f646b5e4ee508e273a7b8e92b056/validation/train.py) and [model](https://github.com/AIRI-Institute/oscgrid/blob/4cbdf81f9a05f646b5e4ee508e273a7b8e92b056/models/models.py). The paper describes Adam. This implementation follows the released script’s AdamW default.
- Inputs are IA, IC, UA BB, UB BB and UC BB. Drop samples missing any of these channels. An active ML_3 annotation takes precedence over ML_2, then ML_1. Samples with none of these active receive class zero.
- Form blocks on changes of file_name or class in the filtered data, discard blocks shorter than 32 samples, shuffle blocks and assign by class with approximate train/validation/test proportions of 60/10/30 percent by sample count. Extract 32-sample windows at stride 10. Every window stays within a single-class block, so its target equals the authors’ weighted vote. Fit the channel scaler on training samples.
- The input has 160 values. The hidden layer has 128 units, batch normalization, ReLU and dropout 0.3. Training uses inverse-class-frequency weighted cross-entropy, AdamW at 0.001, weight decay 0.0001, batch size 128 and gradient clipping at norm 10. Restore the checkpoint with the highest validation accuracy, stopping after ten epochs without improvement or 100 epochs.
- Seed 443 comes from the released code. Seeds 444–447 are our choices. Each seed controls both block assignment and training. This implementation batches precomputed windows and uses a seeded permutation each epoch. Its minibatch order can differ from the upstream data loader. We verified every standardized input and target in all three seed-443 splits against upstream preprocessing.
- A recording can contribute different blocks to multiple splits. For seed 443, 229 of the 283 test recording identifiers also occur in training after stripping bus suffixes. These scores describe performance under the released block-split protocol. The separate earth-fault experiment requires grouping by whole recording.
- summary.csv gives published F1, reproduced mean, difference and an approximate 95% Student-t interval for the five run scores. The interval describes variation across the selected runs on this dataset. The overlapping source recordings prevent interpreting it as uncertainty for entirely new grids.
