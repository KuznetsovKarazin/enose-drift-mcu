# Firmware and board runs (E5, E7): Mega2560, ESP32-C3, ESP32-S3

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

## E7: complete streams on the boards (revision)

The `*_hil` environments build a second firmware (`src/hil/hil_main.cpp`)
that receives real samples from the host over USB and runs complete streams
with the same kernels: D1 batches 2-10 in the order of E2 seed 0 with 20
references per batch, DS transfer fitted and fused into the head on the board
(pairs 1->3, 1->4, 1->5), and three complete 600 s D2 recordings through the
streaming extractor. The board returns its predictions; `hil.py` compares them
with the host float32 replica, and records kernel time per frame and the peak
stack depth (stack painting). It needs the datasets on the PC:

```powershell
git pull
py -m pip install -r requirements.txt -r mcu\requirements.txt
py scripts\download_data.py
py scripts\extract_d2.py
py mcu\hil.py --board mega2560 --port COM7 --out runs\mega_e7
py mcu\hil.py --board esp32c3  --port COM8 --out runs\c3_e7
py mcu\hil.py --board esp32s3  --port COM9 --out runs\s3_e7
```

The Mega2560 link runs at 1,000,000 baud. The d = 128 streams on the Mega2560
take about 15 min each and run only with `--mega-d128`. The protocol and
`hil.py` were tested against the same firmware source compiled for the host
(`host/hil_host_shim.h`), which reproduced all 13,285 predictions per stream.

`mcu/hil_expected.json.gz` holds the host float32 outputs that `hil.py` compares
with the board (made by `python mcu/make_hil_expected.py`), so the board run
needs no C compiler on the PC. Without that file, `hil.py` recomputes them,
which requires gcc.

## Compiler stack frames

`stack/stack_probe.c` includes the kernels with inlining disabled so that the
compiler reports the stack frame of each function (`-fstack-usage`). It is not
linked into the firmware. For example, with the PlatformIO toolchains:

```sh
cd mcu && avr-gcc -mmcu=atmega2560 -Os -fstack-usage -Iinclude -c stack/stack_probe.c -o stack_probe.o
# stack_probe.su: ek_ridge 67 B + ek_chol_solve 56 B = 123 B, the deepest chain
```
