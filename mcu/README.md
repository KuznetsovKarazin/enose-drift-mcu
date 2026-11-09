# E5 firmware: drift-compensation kernels on Mega2560, ESP32-C3, ESP32-S3

The firmware runs every compensation kernel on a fixed fixture of real data
(D1b batch 7 for M0/M1/M5/M6/M7, D2 board 1 -> board 2 for DS, one raw D2
trace for the streaming extractor). For each kernel it first checks its outputs
against the host reference (`include/enose_expected.h`; predictions must match
exactly, checksums within 1e-4 relative), and only then times it 10 times.

Host verification (done in the repository): the C kernels give the same
predictions as the Python methods on every fixture sample, and the C streaming
extractor equals the numpy float32 extractor bit for bit
(`fixture_manifest.json`). `RLS d=128` (66.6 kB of state) is compiled only for
the ESP32 boards; the Mega reports `SKIP,rls128,does_not_fit_sram`.

## Build sizes (PlatformIO 6.1.18, 2026-11-09)

| Board | Clock | RAM used | Flash used |
| --- | --- | --- | --- |
| Mega2560 | 16 MHz | 5,497 / 8,192 B | 67,378 / 253,952 B |
| ESP32-C3 | 160 MHz | 84,812 / 327,680 B | 348,758 / 1,310,720 B |
| ESP32-S3 | 240 MHz | 89,412 / 327,680 B | 362,101 / 1,310,720 B |

## Running on the boards (Windows PowerShell, repository root)

No sensors are needed. The command replaces the board's firmware.
Close any serial monitor first. Use a new output folder each time.

```powershell
py -m pip install -r mcu\requirements.txt
py -m serial.tools.list_ports
py mcu\run_boards.py --board mega2560 --port COM7 --out runs\mega_e5
py mcu\run_boards.py --board esp32c3  --port COM8 --out runs\c3_e5
py mcu\run_boards.py --board esp32s3  --port COM9 --out runs\s3_e5
```

Each run uploads once and captures two sessions. Expected ending:
`Validated runs\..._e5`. Send back the whole output folder (zip). It holds raw
logs, parsed JSON, firmware images and source hashes. Keep failed folders too.

## Regenerating the fixture

`python mcu/gen_fixture.py` (needs `scripts/download_data.py` and
`scripts/extract_d2.py`). It rewrites the two generated headers and the manifest.
