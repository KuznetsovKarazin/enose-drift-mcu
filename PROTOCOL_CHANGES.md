# Changes after the protocol was frozen

All entries below were made after E1-E5 had been run. None changes a frozen
hyperparameter, split or seed of E1-E5, and none of their results was re-run
or replaced. The new analyses are reported separately as post hoc (E6, E7).

## 2026-11-10: revision after two internal reviews

Reason: two reviews of the first manuscript draft (2026-11-10).

1. **Statistics.** The Wilcoxon tests of the draft treated the 19 board pairs
   of E4 and the 8 batches of E2 as independent units. Pairs share boards and
   records, and E2 batches share the carried-over state. The tests are
   withdrawn. E4 differences are now given with a board-level cluster
   bootstrap (resampling the five boards); E2 differences are given per batch,
   with intervals over the 20 reference draws described as conditional on the
   single dataset. The DS versus EWMA comparison now uses EWMA at the same
   k = 16 (identical test records), not EWMA at k = 0.
2. **E6, float32 replica (post hoc).** Complete E2 streams (all seeds, k and
   methods M0, M1, M5, M6, M7, M1+M5, M1+M7) and E4 DS are re-run with the
   float32 C kernels of the firmware (`mcu/host/f32_lib.c`) and compared with
   the float64 host prediction by prediction.
3. **E6, scenarios (post hoc).** E2 with gases arriving in blocks within each
   batch, and with all k references of a session taken from one gas.
4. **E6, implementation checks (post hoc).** Block least squares with the
   intercept penalised like the weights (the code of E2 penalises it by 1e-9);
   RLS with a bounded covariance trace and with forgetting factors 0.95 and
   0.99 for E2 and E3 (sensitivity only, no re-selection).
5. **E6, D2 data checks (post hoc).** E4 repeated without the 20 methane
   recordings of board 5 and without the 9 recordings with missing samples.
   DS fused into the head (W' = F W) checked against the separate map.
6. **E7, complete streams on the boards.** New firmware environments
   `*_hil` receive real samples over USB and run complete E2 streams (seed 0,
   k = 20), DS transfer with the fit and fusion on the board, and complete
   600 s D2 recordings through the extractor, with stack painting.
7. **Descriptions corrected to match the code:** the intercept of every ridge
   fit (head, block least squares, DS, RLS warm start) is penalised by 1e-9;
   M9 is block-diagonal DS with one affine 2x2 (d = 16) or 8x8 (d = 64) map per
   sensor, not a band.
8. **E6, extractor on complete recordings (post hoc).** The float32 C extractor
   (`f_ext_run` in `mcu/host/f32_lib.c`) over all 640 complete D2 recordings,
   compared with the features used in E4 (`experiments/e6/ext_full.py`).
9. **D2 exclusion, comparison basis.** Without the methane recordings of board 5
   a board-5 master head has no methane class; the like-for-like comparison uses
   the 15 pairs whose master is not board 5 (`accuracy_master_not5` in
   `results/e6_20261110/summary.json`).
10. **E6, transfer pairs from restricted conditions (post hoc, round 2).** E4 at
    d = 16, k = 16 with the 16 transfer pairs drawn only from the four lowest
    concentration levels, from two gases or from one gas
    (`experiments/e6/run_d2_conc.py`, `results/e6_20261110/d2_concentrated_runs.json`).
11. **E6, day matrix (post hoc, round 2).** Static d = 16 head trained on each
    repetition of a board and tested on every other repetition, so that the
    day-to-day conclusion does not rest on the assumed order of the days
    (`experiments/e6/day_matrix.py`, `results/e6_20261110/day_matrix.json`).
12. **Board-5 methane.** Fonollosa et al. (2016), Sec. 4.3, state that these
    recordings were discarded because of acquisition problems; the paper now
    cites this and bases its practical conclusions on the cleaned 15-pair analysis.
13. **hil.py robustness.** The run loop writes `raw.log` and `manifest.json`
    (with `completed` and `error`) even when a run stops with an exception.
