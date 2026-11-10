# E7 board captures (2026-11-10)

- `mega_e7`: first Mega2560 run; D1 streams valid, stack reading 0 (measurement bug,
  fixed in 00a17b0), stopped at DS because the PC script needed gcc (fixed).
- `mega_e7b`: Mega2560 with the stack fix; D1 streams complete; stopped at DS because
  the DS pair frame (129 B) overflowed the 64-byte AVR RX buffer while the slave
  half was preprocessed (fixed: the whole frame is read first). DS and extractor
  are captured separately with `--skip-d1`.
- `c3_e7`, `s3_e7`: complete runs (source hashes match the repository, CRLF-normalised).
  `all_d1_predictions_match` is false only because RLS differs from the x86 float32
  replica on 2 of 13,285 (d=16) and 4 of 13,285 (d=128) near-tied predictions.
