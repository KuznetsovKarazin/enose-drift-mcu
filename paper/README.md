# Manuscript (IEEE Sensors Journal, special issue on bio-inspired e-nose, tongue and skin)

Revision 1 of 2026-11-10 (after two internal reviews; see `response_r1.md`).
Built with the official IEEE JSEN LaTeX template: `ieeecolor.cls`, `jsen.sty`
and `IEEEtran.bst` are unmodified copies from `IEEE-JSEN-LaTeX-template-202405.zip`
(SHA-256 of the files: ieeecolor.cls 51d9f42a…, jsen.sty 7d9a30f7…, IEEEtran.bst 314f0ece…).

```sh
python experiments/e6/analyse.py results/e6_20261110   # statistics and E6 summaries
python paper/make_assets.py                             # tables, figures, ram.json (no refitting)
cd paper && latexmk -pdf main.tex
```

- `references.tex`: bibliography, checked entry by entry (authors, volume, pages, DOI).
- `ram.json`: RAM of the firmware configurations (Table VII).
- `response_r1.md`: point-by-point response to both reviews.

## Open items before submission

- **E7 board runs** (`python mcu/hil.py --board ... --out runs/..._e7`): fill the
  sentences in red (`\pending{...}`) in Section VI: agreement on complete streams,
  time per sample, peak stack, arena size.
- Zenodo archive and DOI for the code and board captures (on acceptance).
- Confirm from the full text of Fonollosa et al. 2016 whether repetitions R1-R4
  follow the day order (the UCI page lists the days only).
- Q15 kernels foreseen by PROTOCOL.md are not included (stated in the Discussion).
- IEEE membership grade in the author line, author photo and biography if the
  journal requests them at the final stage.
- The AI-use statement in the Acknowledgment must stay accurate (IEEE policy).
