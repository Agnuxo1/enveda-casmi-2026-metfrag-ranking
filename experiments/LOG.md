# Experiment log — Enveda CASMI 2026 (cycle c2, from 2026-10-28)

Metric: MRR@25 on InChIKey14 after RDKit 2026.03.3 tautomer canonicalisation (official).
Control: C00 = credited fork of the public 0.472 notebook (alexchilton, fork of romanrozen 0.468).
Decisions by JEV (provenance=jev) stored in D:/PROJECTS/.cognition/enveda-casmi-20261028/jev_answer_*.json.

## E00 — Is the public labelled panel a valid proxy for the board? (2026-10-28)
- Hypothesis: the 120-molecule public panel (sanpier/catalogue-experiment-saner, final-25 lists of the pinned
  0.465 pipeline + neural scores) ranks re-ranking variants in the same order as the public leaderboard.
- Design: recompute V0/V1/V2/C472/V4 leader-safe RRF variants on the panel; paired bootstrap (2000) vs V1.
- Control: V1 (public 0.466 recipe).
- Result: V1 0.9176; C472 0.9128 (−0.0048, CI [−0.0125,+0.0015]); V4 0.9118. Board: V1 0.466 < C472 0.472.
- Conclusion: REJECTED. The panel anti-predicts the board for learned-weight changes (library molecules,
  baseline MRR 0.92 vs 0.47 on the hidden NPs). It is used only as a non-regression / discrimination check.

## C00 — Control submission (2026-10-28)
- Kaggle submission ref 57729674, kernel franciscoangulo/enveda-casmi-c00-control-public-0472 v1.

## E01 — MetFrag fragmentation evidence as an independent voter (pre-registered 2026-10-28)
- Hypothesis H1: MetFrag 2.6.11 FragmenterScore (professional in-silico fragmenter, IPB Halle, LGPL-2.1)
  discriminates the true structure among the final-25 candidates (standalone MRR clearly above the
  random-order expectation) and, used as a third RRF voter on ranks 2–25 only (leader kept), does not
  hurt and preferably improves the panel MRR.
- Design: up to 6 spectra/molecule (timsTOF first, single-M adducts), tree depth 2, 10 ppm / 0.003 Da;
  score per candidate = mean over spectra of per-spectrum max-normalised FragmenterScore.
- Acceptance (JEV rule): fusion gain ≥ +0.005 mean with paired-bootstrap CI > 0 → submit;
  neutral (CI covering 0) with strong standalone discrimination → at most one exploratory board submission.
- Execution: Kaggle CPU kernel franciscoangulo/enveda-casmi-e01-metfrag-panel v2 (v1 crashed on a result-parsing
  bug before scoring; fixed). 532 spectrum jobs, 118/120 molecules covered, 0 errors, 2,171 s, mean 16 s/job.
  Environment: Kaggle image ships OpenJDK 21 (/usr/bin/java); MetFrag jar sha256 8c55322f…d8cfb; RDKit 2026.03.3.
- Result (e01_eval_k6.json):
  - Standalone discrimination (MRR when ordering the final-25 by one signal): MetFrag 0.554, neural 0.441,
    random expectation 0.153 → MetFrag carries strong, independent structural evidence.
  - Deployable rule D (control order as one voter, MetFrag rank as second; rank 1 fixed), paired bootstrap 4000:
    w 0.25 → +0.0017 CI95 [+0.0001, +0.0040] (folds +0.0005/+0.0047/0.0000);
    w 0.50 → +0.0014 CI [−0.0028, +0.0054]; w 0.75 → +0.0038 CI [−0.0018, +0.0105]; w 1.0 → +0.0037.
  - Ceiling: only ~15 of 120 panel molecules are not already at rank 1, so the attainable gain is small.
- Conclusion: H1 PARTIALLY ACCEPTED. Strong discrimination; fusion gain positive with CI>0 at w 0.25 but below
  the +0.005 validation threshold → per the pre-registered rule, one exploratory board submission (C01).
  JEV (provenance=jev, conf 0.51) chose w = 0.25.

## E01b — Runtime budget: fewer spectra per molecule (2026-10-28)
- Hypothesis: 2–3 spectra per molecule keep most of the E01 gain at a third of the cost.
- Result (e01b_spectra_budget.json): standalone MetFrag MRR 0.556/0.537/0.554/0.554 for 1/2/3/6 spectra, but the
  fusion gain at w 0.25 falls to +0.0010/+0.0003/+0.0010 (vs +0.0017 with 6). Job time grows with peak count
  (11 s for ≤40 peaks, 25 s for >347 peaks).
- Conclusion: REJECTED (fewer spectra lose most of the fusion gain). Keep 6 spectra; protect runtime with a
  90-min budget and per-molecule partial application (JEV confidence 0.16 on the budget question; tie broken
  by evidence: molecules are independent, so partial application is the same rule on a subset).
