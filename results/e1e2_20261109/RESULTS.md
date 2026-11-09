# E1/E2 on D1b, host run of 2026-11-09

Code: commit 4a439fb (protocol frozen before this run; `src/` and `experiments/` unchanged since). `run_info.json` reports a dirty worktree only because `results/` and `docs/` were untracked at the end of the run. Two configurations re-run independently reproduce their per-batch rows exactly. Runtime 694 s on 4 cores.

Host arithmetic float64; state sizes are float32 layouts. Accuracy = mean over batches 3-10 (batch 2 used for tuning), 20 test seeds, 95% bootstrap CI over seeds. MCU timing not yet measured.

## E2: accuracy vs reference budget and persistent state

### d = 16 (head payload 408 B, head lambda 100.0)

| Method | k=0 | k=1 | k=5 | k=20 | State, B (k=20) |
| --- | --- | --- | --- | --- | --- |
| M0 | 0.434 | 0.434 | 0.434 | 0.434 | 0 |
| M1 | 0.541 | 0.541 | 0.541 | 0.541 | 132 |
| M2 | 0.302 | 0.302 | 0.302 | 0.301 | 388 |
| M3 | =M0 | 0.464 | 0.438 | 0.503 | 196 |
| M4 | =M0 | 0.476 | 0.430 | 0.465 | 128 |
| M5 | =M0 | 0.447 | 0.479 | 0.574 | 4 |
| M6 | =M0 | 0.447 | 0.631 | 0.876 | 1,160 |
| M7 | =M0 | 0.243 | 0.614 | 0.898 | 1,300 |
| M1+M5 | =M1 | 0.546 | 0.557 | 0.575 | 136 |
| M1+M7 | =M1 | 0.246 | 0.590 | 0.842 | 1,432 |

### d = 128 (head payload 3096 B, head lambda 10.0)

| Method | k=0 | k=1 | k=5 | k=20 | State, B (k=20) |
| --- | --- | --- | --- | --- | --- |
| M0 | 0.498 | 0.499 | 0.498 | 0.498 | 0 |
| M1 | 0.614 | 0.614 | 0.614 | 0.614 | 1,028 |
| M2 | 0.435 | 0.435 | 0.436 | 0.438 | 3,076 |
| M3 | =M0 | 0.479 | 0.585 | 0.658 | 1,540 |
| M4 | =M0 | 0.511 | 0.428 | 0.464 | 1,024 |
| M5 | =M0 | 0.579 | 0.676 | 0.772 | 4 |
| M6 | =M0 | 0.634 | 0.793 | 0.923 | 66,568 |
| M7 | =M0 | 0.481 | 0.721 | 0.855 | 10,260 |
| M1+M5 | =M1 | 0.645 | 0.710 | 0.798 | 1,032 |
| M1+M7 | =M1 | 0.503 | 0.671 | 0.803 | 11,288 |

## E1: static settings (accuracy averaged over test batches 2-10)

| d | Head | Test-batch standardisation | Setting 1 | Setting 2 |
| --- | --- | --- | --- | --- |
| 16 | ridge | no | 0.468 | 0.606 |
| 16 | ridge | yes | 0.599 | 0.599 |
| 16 | logistic | no | 0.469 | 0.614 |
| 16 | logistic | yes | 0.525 | 0.593 |
| 128 | ridge | no | 0.542 | 0.783 |
| 128 | ridge | yes | 0.684 | 0.785 |
| 128 | logistic | no | 0.508 | 0.769 |
| 128 | logistic | yes | 0.691 | 0.756 |

## Findings

1. Zero labels: EWMA standardisation (M1) adds about 11 points over the static head at both d (0.434 -> 0.541 at d=16, 132 B; 0.498 -> 0.614 at d=128, 1,028 B). Adaptive centroids (M2) are worse than the static head.
2. With 20 references per batch, RLS on the head (M6) gives the best accuracy at d=128 (0.923), but its 66.6 kB covariance does not fit the 8 kB SRAM of the Mega2560. At d=16 it needs 1,160 B and reaches 0.876; block OLS (M7) reaches 0.898 with a 1,300 B buffer.
3. NLMS (M5) needs 4 B of extra state and gives 0.772 at d=128, k=20; with M1 in front, 0.798 for 1,032 B.
4. With 1 reference per batch, label-driven methods gain little or lose: M7 collapses at d=16 (0.243). The frozen gamma = 0.1 was tuned at k=5.
5. Affine correction (M4) and component correction (M3) give small or no gains in this protocol.

## Limitations to carry into the paper

- Several selected hyperparameters lie on a grid edge: M1 alpha 0.001 (d=16), M2 beta 0.001, M3 r=3, M5 mu 0.03 (d=16), M6 forgetting 0.9, M7 gamma 0.1 (d=16) and 100 (d=128). Under the frozen protocol they are not re-tuned. A sensitivity sweep must be labelled post hoc.
- Hyperparameters were tuned at k=5 and applied to all k; the k=1 collapse of M7 follows from this.
- References are drawn uniformly at random with true labels, which is optimistic compared with a single calibration gas.
- The protocol-v1 tag could not be pushed from this session (HTTP 403 on tag push); the freezing commit 4a439fb on branch e1e2-host predates the results.
