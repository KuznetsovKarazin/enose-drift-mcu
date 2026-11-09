# enose-drift-mcu

Drift compensation and calibration transfer for metal-oxide gas sensor
arrays under microcontroller memory and time budgets. The study measures
accuracy per byte of persistent state and per microsecond on Arduino
Mega2560, ESP32-C3 and ESP32-S3, using two public UCI datasets
(Gas Sensor Array Drift with concentrations; Twin gas sensor arrays).

Status: data loaders, streaming feature extractor and state accounting are
in place; the experimental protocol is a draft ([PROTOCOL.md](PROTOCOL.md)).

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py          # ~225 MB from archive.ics.uci.edu, SHA-256 verified
python -m pytest -q
```

`scripts/download_data.py --from DIR` uses zips already on disk (they are still verified).

## Layout

- `data/datasets.json`: URLs, digests, licences and citations. Raw data are not committed.
- `src/enose_drift/data.py`: D1/D1b and D2 loaders.
- `src/enose_drift/features.py`: O(1)-state streaming feature extractor and its host equivalent (bit-identical in float32).
- `src/enose_drift/budget.py`: persistent-state bytes of methods M0-M9.

## Relation to other work

The online-update code (NLMS, RLS, block OLS) and the MCU timing harness are
adapted from [lut-kan-native](https://github.com/KuznetsovKarazin/lut-kan-native)
(MIT). No data or results are shared with that repository.

## Data licences

D1/D1b: research use only, per the UCI pages; cite Vergara et al. (2012) and
Rodriguez-Lujan et al. (2014). D2: CC BY 4.0, doi:10.24432/C5MW3K; cite
Fonollosa et al. (2016).
