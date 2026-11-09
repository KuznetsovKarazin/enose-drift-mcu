# Literature notes (working, 2026-11-09)

Scope check for the claim "drift compensation with on-device adaptation,
byte-exact state and measured MCU time". Entries marked *read* were opened;
*listed* ones were seen only in search results and must be read before citing.

| Work | Data | MCU evidence | On-device adaptation | Status |
| --- | --- | --- | --- | --- |
| Vergara et al., Sens. Actuators B, 2012, classifier ensembles | UCI 224 | none | no | dataset paper |
| Rodriguez-Lujan et al., Chemometr. Intell. Lab. Syst. 130, 2014 | UCI 270 | none | no | dataset paper |
| Fonollosa et al., Sens. Actuators B, 2016, direct standardization, doi:10.1016/j.snb.2016.05.089 | UCI 361 | none | no | dataset paper |
| Krayden et al., "TinyML-Based Real-Time Drift Compensation for Gas Sensors Using Spectral-Temporal Neural Networks", Chemosensors 13(7) 223, 2025, doi:10.3390/chemosensors13070223 | own GMOS clean-air recordings, not UCI | MCU named only as BLE module; latency measured on a desktop GPU (~46 ms); model 154-182 kB int8; no RAM/energy | no (offline-trained, inference only) | read |
| Li et al., "A Comprehensive Review of Algorithms for Drift Compensation in MOS Gas Sensor Arrays", Chemosensors 14(6) 143, 2026 | review | none; names performance-efficiency trade-off and deployable models as open gaps, and lack of standard protocols/regression benchmarks | - | read (partly via extraction) |
| "Drift-Robust Lightweight Deep Learning on Open Gas Sensor Benchmarks...", PMC13257611 | open benchmarks | unknown | unknown | listed, page blocked (captcha) |
| "Sensor-Drift Compensation in E-Nose Gas Recognition Using Knowledge Distillation", Informatics 13(1) 15 / arXiv 2507.17071 | UCI 224 (likely) | unknown | unknown | listed |
| "Unsupervised Attention-Based Multi-Source Domain Adaptation ... E-Nose", arXiv 2409.13167 | UCI 224 (likely) | unknown | unknown | listed |
| Cross-domain active learning for e-nose drift, Micromachines 13 1260, 2022 | unknown | unknown | unknown | listed |

Working conclusion: in what was read, no work reports online drift
adaptation running on a microcontroller with measured state bytes and
update time; the 2026 review names this trade-off as a gap. Still to do:
read the listed items, and search for "incremental/online learning e-nose
microcontroller", Artursson 2000 (component correction), Zhang & Zhang 2015 (DAELM).
