# Detecting single-phase earth faults in power-grid recordings

**Md. Aslam Hossain | mh0708@uah.edu | A25377404 | Final Project**

<p align="justify"> Can a classifier distinguish a single-phase earth fault from routine switching using voltage and current measurements? Both events can produce an abrupt change in a relay recording. A protection engineer reviewing a disturbance needs to decide whether the waveform shows a fault or an expected change in the network. My project will test whether a classifier can flag fault intervals for that review. </p>

## Dataset and original source

<p align="justify"> I will use <strong>OscGrid, version 6</strong>, a collection of 50,765 oscillograms. An oscillogram is a short recording of electrical signals sampled together over time. The original measurements came from relay protection and automation terminals on operating power grids. The researchers assembled the recordings from industrial databases and collaborating organizations, then anonymized them. I will use their public release for this project. Aleksey Evdakov and his team released the anonymized recordings on Figshare. I will use the 480 expert-annotated recordings as the starting pool and select intervals with the event labels defined below. 

[Original dataset, version 6](https://doi.org/10.6084/m9.figshare.28465427.v6) · [Data collection and annotation paper](https://doi.org/10.1038/s41597-026-06587-8)

## Project questions and engineering justification

1. Can one cycle of three-phase current and voltage distinguish a single-phase earth fault from routine switching? I will compare logistic regression and a random forest on the same recordings.

2. For a published point of comparison, how closely can I reproduce the OscGrid paper’s four-class multilayer perceptron (MLP) results? I will compare per-class F1 scores against Table 2 of the original study.

<p align="justify"> This is a power-system protection problem. When a breaker switches a load, an abrupt change in current can be expected. A phase-to-earth fault can also disturb current and voltage, but it calls for investigation of the affected circuit. A missed fault can leave a disturbance unexplained. A routine switching event flagged as a fault consumes the engineer’s review time. I will therefore measure both the fraction of faults detected (recall) and the fraction of fault flags that are correct (precision).

## Measurements and features

<p align="justify"> A waveform sample gives the channel readings at one instant. One example for my classifier will span one electrical cycle. I will use the sampling rate and grid frequency of each recording to determine the number of samples per cycle.

| Input or metadata                | Meaning                                             | How I will use it                                                                                                                              |
| -------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Phase currents A, B, C           | Current flowing in each measured phase              | Calculate RMS current for each phase and the difference between the largest and smallest phase RMS values.                                     |
| Phase voltages A, B, C           | Voltage on each measured phase                      | Calculate RMS voltage per phase and measure how unevenly the phase voltages change. I will select recordings with compatible voltage channels. |
| Recording identifier             | Identifies samples from the same oscillogram        | Keep all windows from one recording in the same data split.                                                                                    |
| Sampling rate and grid frequency | Samples per second and electrical cycles per second | Align the feature window to one cycle across devices.                                                                                          |

### Input features and target label

For the binary classifier, one example is one electrical cycle from one recording.
I calculate eight numerical features from the normalized three-phase waveforms.
RMS means root mean square.

| Feature | Definition |
| --- | --- |
| 1. Phase A current RMS | RMS current in phase A |
| 2. Phase B current RMS | RMS current in phase B |
| 3. Phase C current RMS | RMS current in phase C |
| 4. Phase A voltage RMS | RMS voltage in phase A |
| 5. Phase B voltage RMS | RMS voltage in phase B |
| 6. Phase C voltage RMS | RMS voltage in phase C |
| 7. Current imbalance | Largest minus smallest of the three phase-current RMS values |
| 8. Voltage imbalance | Largest minus smallest of the three phase-voltage RMS values |

**Target label:** `1 = single-phase earth fault` and `0 = routine switching`.
I assign a label when the expert annotation is consistent throughout the cycle.

## Labels and evaluation

OscGrid provides expert annotations at individual time samples. The published benchmark groups the annotations into four classes.

| Original benchmark class | Meaning in a recording                                                         |
| ------------------------ | ------------------------------------------------------------------------------ |
| No event                 | No annotated disturbance in the interval.                                      |
| Operational switching    | An equipment operation, such as closing a breaker or energizing a transformer. |
| Abnormal events          | A deviation such as a single-phase earth fault or a minor voltage dip.         |
| Fault events             | A critical disturbance such as a short circuit or voltage collapse.            |

For my two-class experiment, I will select a subset of these annotations. A single-phase earth fault has its own entry under **abnormal events**, so I will use that specific entry to define the target. [Annotation hierarchy and event definitions](https://doi.org/10.1038/s41597-026-06587-8)

| Project label                | Rule for including a window                                                                             |
| ---------------------------- | ------------------------------------------------------------------------------------------------------- |
| 1 — single-phase earth fault | The entire cycle carries an expert annotation for a single-phase earth fault or one of its subtypes.    |
| 0 — routine switching        | The entire cycle carries an operational switching annotation and no concurrent abnormal or fault event. |

<p align="justify"> I will exclude ambiguous transitions, mixed-class windows, and records without the required channels. I will report the number of distinct recordings per class after selection. I will compare logistic regression with a random forest on held-out recordings and report fault recall and precision. Recall measures how many labeled fault windows I find. Precision measures how often a flagged window actually contains a labeled fault. Event annotations will be used only to construct and score the targets.

## Obtaining the data

1. Open the authors’ [Figshare release, version 6](https://doi.org/10.6084/m9.figshare.28465427.v6).
2. Download [labeled.7z](https://ndownloader.figshare.com/files/60075722) for the processed labeled waveforms and [Description.7z](https://ndownloader.figshare.com/files/60074606) for the channel and annotation definitions. Extract both archives with 7-Zip and open `labeled.csv`. The authors also provide the [raw labeled COMTRADE recordings](https://ndownloader.figshare.com/files/60073955).
3. In the description folder, open the event-typing spreadsheet to identify the switching and single-phase earth-fault annotations. Use the analog-signal dictionary to map the phase channels before calculating features.

The authors distribute the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Their [processing code](https://github.com/AIRI-Institute/oscgrid) documents how to regenerate the CSV from the raw recordings.

## GitHub submission

This dataset description is stored at `FinalProject/README.md` in my [Tornadochildd repository](https://github.com/tornadochilddgh/tornadochildd).
