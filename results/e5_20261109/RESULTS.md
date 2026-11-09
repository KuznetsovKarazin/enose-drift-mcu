# E5 on Mega2560, ESP32-C3 and ESP32-S3, board run of 2026-11-09

Firmware: commit 578719c (branch e5-mcu). Captured by the author with
`mcu/run_boards.py`, two sessions per board after one upload. Raw logs, parsed
JSON, size and upload logs are in the board folders. Firmware images are not
committed; their SHA-256 digests are in each `manifest.json`, and the full
capture archive (sha256 18e458d9…) is kept outside the repository.

Checks (`python experiments/e5/analyse.py results/e5_20261109`):
- All six sessions are `validated`. Every kernel passed its on-board check
  before timing: predictions equal the host reference on every fixture sample,
  checksums within 1e-4 relative, extractor features within 1e-5 relative
  (observed gap 0 on all boards).
- Board source hashes equal the repository sources at 578719c (the capture
  machine checked them out with CRLF line endings).
- The two sessions on each board agree within 0.13 % for every kernel.
- RLS at d=128 (66,568 B of state) is reported as `SKIP` on the Mega2560.

## Time per call (median of 20 timings, µs) with host accuracy

Host accuracy is from E2 (D1b, batches 3-10, k=20 references per batch) or E4
(D2 transfer, k=16 pairs, d=16). It is not measured on the boards; the boards
reproduce the host predictions on the fixture.

| Kernel | One call | State, B | Host acc. | Mega2560 16 MHz | ESP32-C3 160 MHz | ESP32-S3 240 MHz |
| --- | --- | --- | --- | --- | --- | --- |
| M0 static, d=16 | classify | 0 | 0.434 | 5,401 | 505 | 25.2 |
| M0 static, d=128 | classify | 0 | 0.498 | 42,542 | 4,039 | 197.0 |
| M1 EWMA, d=16 | classify | 132 | 0.541 | 7,791 | 663 | 36.7 |
| M1 EWMA, d=128 | classify | 1,028 | 0.614 | 61,438 | 5,298 | 289.0 |
| M5 NLMS, d=16 | classify + update | 4 | 0.574 | 9,603 | 830 | 37.3 |
| M5 NLMS, d=128 | classify + update | 4 | 0.772 | 74,769 | 6,528 | 287.2 |
| M6 RLS, d=16 | classify + update | 1,160 | 0.876 | 22,102 | 1,805 | 72.8 |
| M6 RLS, d=128 | classify + update | 66,568 | 0.923 | does not fit | 57,904 | 2,114 |
| M7 block OLS, d=16 | one refit, k=20 | 1,300 | 0.898 | 164,426 | 11,650 | 797 |
| M8 DS, d=16 | fit from 16 pairs | 1,088 | 0.955 | 274,708 | 19,792 | 1,224 |
| M8 DS, d=16 | map + classify | 1,088 | 0.955 | 9,412 | 826 | 35.2 |
| Streaming extractor | one 8-channel sample | 426 / 428 | n/a | 1,265 | 98.1 | 4.79 |

Extractor state is 426 B on AVR and 428 B on the ESP32 targets (struct padding).
`e5_timing.csv` has min/max, cycles and the session spread for each row.

## Findings

1. Every method fits a measurement cycle of an e-nose: the slowest per-sample
   kernel on the Mega2560 (NLMS at d=128) takes 75 ms, and one DS fit takes
   0.27 s, done once per calibration session.
2. The streaming extractor at 100 Hz uses 12.7 % of the Mega2560, 1.0 % of the
   ESP32-C3 and 0.05 % of the ESP32-S3 (time per sample / 10 ms).
3. The ESP32-C3 has no FPU and is 10.7-14.1x faster than the Mega2560, close to
   the clock ratio of 10. The ESP32-S3 with FPU is a further 15-27x faster.
4. Label-driven adaptation is cheap relative to inference: NLMS adds 1.4-1.8x
   to the static classify time at 4 B of state; RLS at d=16 adds 2.9-4.1x.
5. For the same budget of about 1.1 kB, DS at d=16 (0.955 on D2) and RLS at d=16
   (0.876 on D1b) run on all three boards; the 0.923 of RLS at d=128 needs
   66.6 kB and an ESP32 class device, and costs 2.1 ms per sample on the S3.

## Limitations

- One unit per board type; timings include the `micros()` resolution (4 µs on
  AVR), negligible against per-call times above 25 µs.
- Accuracy on the boards is shown only on the fixture (64/32/44/64 samples);
  the accuracy column is host accuracy under the frozen protocol.
- Default Arduino optimisation (-Os) with -fno-fast-math -ffp-contract=off;
  faster builds are possible but would change the float results.
