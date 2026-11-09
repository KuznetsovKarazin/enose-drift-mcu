# Experimental protocol (DRAFT, not yet frozen)

Status: draft of 2026-11-09. The protocol is frozen by a tagged commit
(`protocol-v1`) before the first full run. After freezing, any change is
logged in `PROTOCOL_CHANGES.md` with its reason; nothing is re-tuned on test data.

## Aim

Measure how much drift-compensation and calibration-transfer accuracy a gas
sensor array obtains per byte of persistent state and per microsecond of
update time on Arduino Mega2560 (AVR), ESP32-C3 (RISC-V) and ESP32-S3
(Xtensa, FPU). Accuracy is not compared head-to-head with SVM ensembles or
DAELM; published figures are quoted for context only.

## Data

| Key | Source | Use |
| --- | --- | --- |
| D1 / D1b | UCI 224 / 270, 13,910 samples, 16 MOX x 8 features, 6 gases, 10 batches over 36 months; D1b adds concentration | E1-E3 |
| D2 | UCI 361, 640 raw traces, 5 twin boards x 8 Figaro MOX, 4 gases x 10 levels, 600 s at 100 Hz | E4-E5 |

Archives are fetched by `scripts/download_data.py` and verified against the
SHA-256 digests in `data/datasets.json`. Raw data are not redistributed.

Observed data facts (2026-11-09):

- D1 and D1b features agree to 1e-6; D1b is used throughout.
- D1 batches are unbalanced (161 to 3,613 samples); toluene is absent from batches 3-5.
- D2: boards 1-3 have repetitions R1-R4, boards 4-5 R1-R2, matching the number
  of test days per board on the UCI page. We assume repetition order equals day order (to confirm with the dataset paper).
- D2: gas arrival (first >3% deviation from the 0-30 s median) has median 56.6 s;
  the earliest clear onset is ~36 s (board 3, ethylene, R2). Baseline window: 0-30 s.
- D2: 9 of 640 traces have missing samples (19,289-49,815 rows instead of
  ~60,000; largest gap 1.47 s). The extractor is sample-indexed, as on a
  microcontroller; these traces are kept and flagged.

## Common model

A linear head on fixed features: multinomial logistic regression for gas
identity, ridge regression of log10 concentration per gas. Features: signed
log transform, standardised on the training block. Feature subsets:
d = 16 (dR of each sensor) and d = 128 (all features) for D1; d = 64 (8 features x 8 sensors) and d = 16 for D2.

## Methods (persistent state in `src/enose_drift/budget.py`)

M0 static head; M1 EWMA feature standardisation; M2 adaptive class
centroids; M3 component correction; M4 per-channel gain/offset from
reference samples; M5 NLMS on the head; M6 RLS on the head; M7 block OLS
over a buffer of k references; M8 direct standardisation; M9 piecewise DS.

## Experiments

- E1: D1 Setting 1 (train batch 1, test 2-10) and Setting 2 (train k-1, test k).
- E2 (main): D1 stream in batch order, 20 random within-batch permutations,
  prequential evaluation (predict, then update). Reference budget: 0, 1, 5, 20
  reference samples per batch; a reference is one known gas at known concentration.
- E3: D1b concentration regression under drift (per gas, log10 ppm).
- E4: D2 calibration transfer, all 20 ordered board pairs, k = 0..40
  transfer samples; within-board day-to-day drift for boards 1-3.
- E5: D2 on-board streaming features vs host features; time per sample on each MCU.
- All of E2-E5 are repeated with Q15/int16 state.

## Metrics and statistics

Accuracy and macro-F1 per batch; RMSE in log10 ppm; persistent state bytes;
flash and RAM from the linker map; stack peak by stack painting; microseconds per
prediction and per update (median and IQR, two sessions per board).
95% bootstrap intervals over permutations and batches; paired Wilcoxon
tests over batches (D1) and board pairs (D2) with Holm correction.

## Leakage control

Hyperparameters (NLMS step, RLS forgetting factor, EWMA alpha, number of
components, DS ridge) are selected only on D1 batches 1-2 and on D2 pair
board 1 -> board 2, then frozen. Seeds: `SEED_BASE = 20270131` plus the
permutation index; no seed is shared with any other study.
