# How to run the OscGrid final project

This project has two experiments. The first repeats the published four-class
OscGrid neural-network benchmark. The second compares logistic regression and
random forest for distinguishing single-phase earth faults from routine switching.

Both experiments use the same [OscGrid version 6 dataset on Figshare](https://doi.org/10.6084/m9.figshare.28465427.v6).
The four-class reference scores come from Table 2 of the
[OscGrid paper](https://doi.org/10.1038/s41597-026-06587-8).

## 1. Clone the repository and prepare the environment

Use a Bash terminal with Git and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.
Run the steps below in order in the same terminal. The first run needs internet access
to install dependencies and download the dataset. Both experiments run on CPU.

```bash
git clone https://github.com/tornadochilddgh/tornadochildd.git
cd tornadochildd
uv sync --locked
cd Final_Project*/oscgrid_benchmark
```

If you already cloned the repository, start from its root and run the last two
commands. The `*` accommodates the trailing space in the project folder name.

## 2. Reproduce the four-class neural-network benchmark

```bash
uv run --with py7zr python reproduce.py
uv run python summarize.py
```

The first command downloads the labeled archive from Figshare, verifies its
checksum, extracts `labeled.csv`, and trains the MLP with seeds 443–447.
The second command summarizes all five runs. Open `runs.csv` for individual
results and `summary.csv` for mean F1 scores and confidence intervals.

| Class | Published F1 | Mean F1 in this project |
| --- | ---: | ---: |
| No event | 0.840 | 0.846 |
| Operational switching | 0.450 | 0.501 |
| Abnormal events | 0.810 | 0.779 |
| Fault events | 0.790 | 0.609 |

The benchmark splits signal blocks. Its terminal output reports how many
recording IDs appear in both training and testing.

## 3. Check which recordings support the binary experiment

```bash
cd ../earth_fault_classification
uv run python audit_cycles.py
uv run python audit_cycles.py --five-channel
```

Each example is one complete electrical cycle of 32 samples. The first command
requires all six current and voltage channels. The second allows phase-B current
to be absent. Both commands reuse the CSV downloaded in Step 2.

| Required channels | Routine switching | Single-phase earth fault |
| --- | --- | --- |
| Six | 35 recordings, 842 cycles | 1 recording, 8 cycles |
| Five | 83 recordings, 1,572 cycles | 23 recordings, 844 cycles |

The five-channel experiment contains 86 distinct recordings. Twenty recordings
contribute cycles to both classes. Requiring six channels would leave only one
fault recording, too few to train and test on separate fault recordings.

## 4. Train logistic regression and random forest

```bash
uv run python classify.py
```

The target is `1` for a single-phase earth fault and `0` for routine switching.
The five base features are the per-cycle RMS values of currents IA and IC and
busbar voltages UA, UB, and UC. The seven-feature version adds the absolute
difference between the two current RMS values and the range of the three
voltage RMS values.

The script uses 64 recordings for model development and 22 held-out recordings
for testing. All cycles from a recording stay together. Four validation folds
within the development set are used to select model settings and decision
thresholds.

| File | What to inspect |
| --- | --- |
| `cycles.csv` | Cycle features, labels, recording IDs, and split assignments |
| `results.csv` | Selected settings, precision, recall, F1, and confusion counts |
| `predictions.csv` | True and predicted labels for held-out cycles |
| `ablation.csv` | Effect of adding the two derived features |

## 5. Repeat training and compare the models

```bash
uv run python repeat_training.py
```

This repeats training across five seeds with the same 22 held-out recordings.
Open `repeated_training/summary.csv` for mean fault F1 and confidence intervals.
Open `repeated_training/ablation.csv` for the paired feature comparison.
The bootstrap intervals resample complete test recordings.

| Model | Five features | Seven features |
| --- | ---: | ---: |
| Logistic regression | 0.827 | 0.937 |
| Random forest | 0.946 | 0.931 |

These are mean fault F1 scores for the binary task. The four-class benchmark
in Step 2 answers a different question and uses a different split.

## 6. Generate the figures

```bash
uv run python plot_confusion_matrix.py
uv run python plot_feature_comparison.py
```

Open `figures/forest5_confusion_matrix.pdf` for the five-feature forest's
single-run confusion matrix. Open `figures/feature_comparison.pdf` for the
five-seed comparison. Both scripts also save PNG copies.

## 7. Read or compile the paper

Open `Final_paper/` in the project directory to find the submitted PDF and the
editable LaTeX source. To compile in Overleaf, upload `main.tex`, `IEEEtran.cls`,
and the three figure PDFs together, select `main.tex`, and click **Recompile**.

To compile locally, open a terminal in the folder containing `main.tex` and run:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Open `main.pdf`. The second pass resolves citations and cross-references.