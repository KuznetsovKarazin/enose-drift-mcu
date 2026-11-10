# E7 board captures (2026-11-10)

- `mega_e7`: first Mega2560 run; D1 streams valid, stack reading 0 (measurement bug,
  fixed in 00a17b0), stopped at DS because the PC script needed gcc (fixed).
- `mega_e7ds`: Mega2560 DS and extractor (`--skip-d1`) with the fixed firmware; validated.
- `mega_e7b`: Mega2560 with the stack fix; D1 streams complete; stopped at DS because
  the DS pair frame (129 B) overflowed the 64-byte AVR RX buffer while the slave
  half was preprocessed (fixed: the whole frame is read first). DS and extractor
  are captured separately with `--skip-d1`.
- `c3_e7`, `s3_e7`: complete runs (source hashes match the repository, CRLF-normalised).
  `all_d1_predictions_match` is false only because RLS differs from the x86 float32
  replica on 2 of 13,285 (d=16) and 4 of 13,285 (d=128) near-tied predictions.

## Notes added in revision round 2

- `mega_e7` and `mega_e7b` stopped with an exception before `hil.py` wrote
  `raw.log` and `manifest.json`; their console output is the record. `hil.py`
  now writes both files in all cases.
- Manifests record the commit `578719c` because the runs were made from a
  copy of the repository whose git metadata was not updated. The firmware and
  host sources of all E7 runs match commit `00a17b0` by their SHA-256 digests
  (`sources` field of each manifest); this is the code that produced E7.
- The test firmware keeps every head in RAM for convenience. Its arena
  therefore exceeds the algorithm accounting of Table VII by the head size for
  M0 and M1; for methods that modify the head the two agree within 4 B.
  Mega2560: 6,036 B static RAM + 425 B measured stack = 6,461 B of 8,192 B.
