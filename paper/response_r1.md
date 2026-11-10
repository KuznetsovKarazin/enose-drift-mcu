# Response to the reviewers (revision 1)

We thank both reviewers for their careful reading. Their comments led to a substantially revised manuscript: the memory accounting now covers the head, scratch memory and the calibration peak; the statistical analysis no longer treats dependent units as independent; complete data streams were re-run with the firmware kernels; and less favourable calibration scenarios were added. All new analyses were made after the protocol had been frozen. They are labelled post hoc in the paper (E6, E7) and logged with their reasons in `PROTOCOL_CHANGES.md`; no frozen hyperparameter, split or seed of E1–E5 was changed, and no E1–E5 result was replaced.

Items marked **[E7]** depend on the complete-stream runs on the three boards, which are being repeated with the new firmware; the corresponding sentences are marked in red in the manuscript and will be completed before resubmission.

---

## Reviewer 1

**3.1 Holm-adjusted p = 0.047 is impossible with eight batches.**
With eight batches, the smallest two-sided exact Wilcoxon p-value is 2/2⁸ = 0.0078. Holm's procedure is step-down: the i-th smallest of m p-values is multiplied by (m − i + 1), not by m, and the adjusted values are made monotone. In the family of nine tests, the two comparisons at p = 0.0078 ranked fourth and fifth, after three E4 tests with smaller p-values; the fourth rank has the multiplier 9 − 4 + 1 = 6, which gives 0.0078 × 6 = 0.047, and monotonicity carries this value to the fifth. The value was therefore attainable and correctly computed. We agree, however, with the more important point behind the comment, which Reviewer 2 also raised: the eight batches share the carried-over state, and the 19 board pairs share boards. The Wilcoxon tests are therefore withdrawn. The paper now reports per-batch differences for E2 (mean, number of batches in favour, range) and a board-level cluster bootstrap for E4 (new Table III, Section IV "Statistics").

**3.2 The persistent-state definition excludes the head held in RAM.**
Agreed. Section III now distinguishes state, persistent RAM (state plus the head whenever the method modifies it), scratch memory and the calibration peak. Table II, Table VI and Fig. 1 use persistent RAM; the new Table VII gives all four quantities for every firmware configuration. For example, NLMS at d = 128 keeps 3,100 B rather than 4 B, and RLS at d = 128 keeps 69.7 kB.

**3.3 RLS instability and hyperparameters on the grid edges.**
We kept the frozen selections but added post hoc sensitivity runs (Section V-C and V-D). In E2, β = 0.95 instead of the selected 0.9 raises RLS at d = 128, k = 20 from 0.923 to 0.940 but lowers it at d = 16 from 0.876 to 0.813; bounding the trace of P lowers d = 16 to 0.613. In E3, the selected β = 0.8 winds up at k = 20 (worst gas RMSE 0.95), whereas β = 0.95 is stable (0.171 and 0.124 for d = 16 and 128, worst gas 0.27 and 0.19). The text states that the RLS figures depend on β by several points.

**3.4 "per Byte and per Microsecond" in the title and abstract overpromises.**
Agreed. The title is now "Memory and Time Costs of Drift Compensation and Calibration Transfer for Electronic Noses on Microcontrollers", and the abstract reports RAM and time as separate quantities.

**Minor 1. The 11-point gain is stated without caveat.**
Conclusion 1 and Section V-A now state that the gain appears in seven of eight batches (six of eight at d = 16), that EWMA loses up to 0.095 in single batches, and that it decreases when gases arrive in blocks (0.614 → 0.541 at d = 128).

**Minor 2. Give the numbers of M2–M4.**
Added to Table II and Section V-A. Adaptive centroids lower the accuracy (0.302 / 0.436); the per-feature affine correction stays near the static head (0.465 / 0.464 at k = 20). Component correction reaches 0.658 at d = 128, k = 20 with 1,540 B, above EWMA; the first draft wrongly described it as giving small or negative changes, and we thank the reviewer for prompting the check.

**Minor 3. Scratch memory and stack are contradictory.**
Scratch memory is now defined and tabulated (Table VII). Compiler stack frames of the kernels are reported (deepest chain 123 B on AVR, 256 B on RISC-V, 160 B on Xtensa); the measured peak stack on the boards is reported from E7 **[E7]**.

**Minor 4. Repetition order assumed to be day order.**
The dataset description lists the measurement days of each board but not the mapping from repetition index to day; Section II-A now says so and states that only the day-to-day analysis of E4 depends on the assumption.

**Minor 5. Random transfer pairs are optimistic.**
Agreed and stated in Section V-E and the limitations; the one-gas scenario on D1 (Table IV) shows how strongly label-driven updates depend on the coverage of the calibration session.

**Minor 6. Incomplete references.**
All references were checked against the publishers' records; author lists, volumes, pages and DOIs were completed (Li et al. 2026, Krayden et al. 2025, Padilla, Ziyatdinov, Rodriguez-Lujan, Muezzinoglu, Wang, Persaud).

**Minor 7. Archive the code with a DOI.**
The repository will be archived with a DOI on acceptance; the Data and Code Availability section says so.

**Minor 8. The abstract should say that the head is a linear ridge classifier.**
Done.

**Minor 9. "2 B" in Fig. 1.**
Fig. 1 now plots persistent RAM with a labelled zero, and the static head is drawn at 0.

**Minor 10. Cost of E1 retraining.**
Section V-A now gives it: retraining needs the labels of a whole batch and either the batch (about 1.9 MB at d = 128) or its normal equations (69.7 kB at d = 128, 1.6 kB at d = 16).

**Questions 1–7.** The answers are contained in the items above: statistics (3.1), head in RAM and peak memory (3.2, Table VII), forgetting factors and wind-up (3.3), the k = 1 collapse of block least squares (Section V-A: the almost unpenalised intercept moves towards the single reference class; penalising it raises 0.243 to 0.313 at d = 16 and 0.481 to 0.590 at d = 128), board-level agreement on complete streams (Section VI-A, E6 and **[E7]**), and the transfer scenario (Minor 5).

---

## Reviewer 2

**1. Report the full RAM including the head and buffers (e.g., block OLS 3,796 B, DS 4,420 B arena) and the measured maximum memory.**
Agreed; see the response to Reviewer 1, 3.2. Table VII gives head, state, persistent RAM, scratch, calibration peak and peak for each configuration: block least squares at d = 16 needs 3,816 B at the refit (20 more than the reviewer's count because of the stored labels), DS with a separate map 4,420 B. The firmware leaves 7,392 B on the Mega2560; configurations that exceed it are marked. The measured peak stack is reported from E7 **[E7]**.

**2. Fuse DS into the head (272 / 1,040 B).**
Done (Eq. (9)). The fused head is computed on the board after the fit; in float32 it agrees with the separate float64 map on all 86,880 predictions at d = 16 and on all but 2 of 86,880 at d = 64. The text, Fig. 1 and Table VI use the fused size, and Table VII gives both variants.

**3. Formulas versus code (intercept penalty 1e-9, RLS initialisation, description of M9).**
Thank you; the descriptions now match the code. Eq. (2) introduces Λ = diag(λ, …, λ, 10⁻⁹), used in every ridge fit; P₀ = (AᵀA + Λ)⁻¹; Eq. (8) uses Λ_γ; M9 is described as block diagonal (one affine map of the q = 2 or 8 features of each sensor), not as a band. A post hoc variant with the intercept penalised like the weights is reported (M7b).

**4. Board checks used short fixtures only; run full float32 streams and selected sequences on the boards; the extractor fixture covers only the first ~102 s.**
Agreed. (i) All E2 streams of the deployable methods (920 streams, 12.3 million predictions) were re-run with the firmware's float32 C kernels; 65 predictions differ from float64 and no accuracy changes by more than 1.3 × 10⁻⁵. (ii) The C extractor was run over all 640 complete recordings (38.0 million samples) and gives bit-identical features. (iii) A new firmware (E7) receives real samples over USB and runs complete E2 streams, DS with fit and fusion on the board, and three complete 600 s recordings through the extractor **[E7]**.

**5. The statistics ignore dependence (pairs share boards; batches share state).**
Agreed; see Reviewer 1, 3.1. The new Table III uses a cluster bootstrap over boards for E4 and per-batch reporting for E2. Two conclusions of the first draft are weakened accordingly: the advantage of DS over EWMA at k = 16 (+0.049, 16 of 19 pairs) has a board-bootstrap interval of [−0.004, 0.123], and the paper now says that it is not established with five boards.

**6. DS (k = 16) was compared with EWMA at k = 0 on different test sets.**
Corrected. EWMA is now evaluated at k = 16 on the same records as DS (Table III, Section V-E).

**7. Board-5 methane recordings and the nine gappy traces.**
E4 was repeated without each group and without both (Section V-E). Without board-5 methane, a head trained on board 5 cannot recognise methane at all, so we compare the 15 pairs whose master is another board: DS 0.954 instead of 0.957, EWMA 0.926 instead of 0.931, static 0.857 in both. Without the nine gappy recordings, no mean changes by more than 0.003. The order of the methods is unchanged. We could not find a statement about damaged board-5 methane recordings in the dataset description and therefore present the exclusion as a sensitivity check.

**8. The scenarios are too optimistic (references from all gases, random order); state in E3 that k is per gas and the gas identity is known.**
Agreed. The new Section V-B and Table IV add gas-block arrival and one-gas calibration sessions. With one-gas sessions every label-driven update falls below the static head (e.g., block least squares 0.136 at d = 16), and the conclusions now state this. E3 now says explicitly that the gas identity is known and k counts references of that gas.

**9. Positioning, per-batch results, macro-F1, intervals, and the claim that only full RLS needs an ESP32.**
The introduction now covers label-efficient and online drift compensation (active learning on dynamic clusters, online compensation on large streams, cross-domain active learning) and states that we found no study of on-device adaptation memory and time for electronic noses. Macro-F1 is added to Table II, per-batch differences and bootstrap intervals to Table III, and the 95% intervals of all configurations are in the repository. The ESP32 claim is corrected: at full feature size, RLS, block least squares (13.4 kB persistent) and DS (16.6 kB map or 43 kB calibration peak) all exceed the Mega2560, while EWMA, NLMS and their combination fit (Section VI-B, Discussion).

---

## Other changes

- The AI-use statement in the Acknowledgment was shortened while keeping it accurate: "Anthropic Claude was used to generate the code and to draft the text. The author reviewed, edited and verified all content and takes full responsibility for it."
- The bibliography was moved to `references.tex` and verified entry by entry; the Holm reference was removed together with the tests.
