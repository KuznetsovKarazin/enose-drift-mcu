# Manuscript draft (IEEE Sensors Journal, special issue on bio-inspired e-nose, tongue and skin)

Draft of 2026-11-10. Built with the official IEEE JSEN LaTeX template:
`ieeecolor.cls`, `jsen.sty` and `IEEEtran.bst` are unmodified copies from
`IEEE-JSEN-LaTeX-template-202405.zip` (SHA-256 of the files:
ieeecolor.cls 51d9f42a…, jsen.sty 7d9a30f7…, IEEEtran.bst 314f0ece…).

```sh
python paper/make_assets.py        # tables, figures, stats.json from results/ (no refitting)
cd paper && latexmk -pdf main.tex
```

`stats.json` holds the paired Wilcoxon tests with Holm correction quoted in the text.

## Open items before submission

- References to complete or verify: author lists of [li2026] and [krayden]
  (marked "[Authors to be completed]"); page ranges and volumes of
  Artursson 2000, Padilla 2010, Ziyatdinov 2010, Rodriguez-Lujan 2014,
  Muezzinoglu 2009, Wang 1991, Zhang 2015 were written from memory and must be
  checked against the publisher pages; DOIs can be added once verified.
- Literature still to read (docs/LITERATURE.md): knowledge-distillation and
  multi-source domain adaptation e-nose papers, cross-domain active learning.
- Confirm from Fonollosa et al. 2016 that repetitions R1-R4 are in day order.
- Not in this version although foreseen by PROTOCOL.md: Q15 kernels, peak stack,
  complete streams on the boards. The Discussion states this.
- IEEE membership grade in the author line, author photo and biography if the
  journal requests them at the final stage.
- AI-use statement in the Acknowledgment must stay accurate (IEEE policy).
- Zenodo archive and DOI for the code and board captures.
