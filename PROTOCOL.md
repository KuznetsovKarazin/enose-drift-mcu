# Experimental protocol

Status: E1/E2 frozen on 2026-11-09 by commit 4a439fb before their first full run.
E3/E4 frozen by the commit that adds the "E3/E4 details" section below,
before their first full run. E5 (MCU) was fixed by commit 578719c (firmware, fixture and expected outputs) before the board runs. After freezing, any change is
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

A linear head on fixed features: ridge regression onto one-hot class targets
(least-squares classifier), so that NLMS, RLS and block OLS update exactly the
deployed model; ridge regression of log10 concentration per gas for E3. A
multinomial logistic head is reported in E1 as a host reference only.
Features: signed log transform sign(x) log(1+|x|), standardised with the
source-block mean and standard deviation. Feature subsets:
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

## E1/E2 details (frozen)

- Source: batch 1. Head ridge lambda from {0.01, 0.1, 1, 10, 100}, chosen by
  static accuracy on batch 2 (ties: larger lambda).
- Method grids, chosen by mean accuracy on batch 2 over 5 tuning seeds
  (k = 5 for labelled methods, k = 0 for M0-M2): M1 alpha {0.001, 0.003,
  0.01, 0.03, 0.1}; M2 beta, same grid; M3 r {1, 2, 3} x eta {0.1, 0.3, 1};
  M5 mu {0.03, 0.1, 0.3, 0.6, 1}; M6 forgetting {0.9, 0.95, 0.99, 0.999, 1},
  covariance warm-started from the source fit; M7 gamma {0.1, 1, 10, 100};
  M1+M5 and M1+M7 reuse the selected M1 alpha and tune mu / gamma.
- Stream: batches 2-10 in order. In each batch a seeded permutation; the
  first k samples are labelled references (one calibration session), drawn
  uniformly from that batch; the rest are predicted in order, unlabelled.
  State carries over between batches. Budgets k in {0, 1, 5, 20}.
- Headline: accuracy averaged over batches 3-10 (batch 2 was used for tuning),
  20 test seeds, 95% bootstrap CI over seeds. M4 fits gain only with k >= 3.
- Host arithmetic is float64 with float32 state sizes; float32/Q15 replicas follow.

## E3/E4 details (frozen)

E3 (D1b concentration):
- Target log10(ppmv); gas identity known; one model per gas (6 gases). Source:
  batch-1 samples of that gas; stream: batches 2-10, that gas only, seeded
  permutation, first k samples are references with known concentration.
- Methods: M0, M1, M5, M6, M7, M1+M5, M1+M7 with a one-output ridge head.
- Selection on batch 2, mean RMSE over gases, 5 tuning seeds (k=5; M1 k=0):
  head lambda {0.1, 1, 10, 100, 1000}; M1 alpha {0.003, 0.01, 0.03, 0.1, 0.3};
  M5 mu {0.01, 0.03, 0.1, 0.3, 1}; M6 forgetting {0.8, 0.9, 0.95, 0.99, 1};
  M7 gamma {0.1, 1, 10, 100, 1000}; combinations reuse M1 alpha.
- Headline: RMSE (log10 ppm) and median relative error over batches 3-10,
  averaged over gases with data; seeds base 20270131+300.

E4 (D2 transfer and day drift):
- Features: 8 per channel from the streaming extractor (d = 64), or dR and
  dR/R0 per channel (d = 16). Head: 4-gas least-squares classifier on all
  records of the master board.
- Transfer i -> j, 20 ordered pairs; k in {0, 2, 4, 8, 16, 40} of the 40
  (gas, level) conditions drawn per seed; repetition-1 records on both boards
  form pairs (M4 paired affine, M8 DS, M9 per-channel DS, all ridge-shrunk
  to identity) or labelled references (M5, M6, M7). Evaluation: every other
  record of board j. M1 runs on the slave stream with 0 labels.
- Selection on pair 1 -> 2 only (k = 8, 5 tuning seeds); headline: mean and
  worst pair over the other 19 pairs, 20 seeds (base 20270131+400).
- Day drift: boards 1-3, train on R1, stream R2-R4 (R taken as day order, unconfirmed)
  with k in {0, 2, 4} references per day; hyperparameters reused from transfer selection.

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
