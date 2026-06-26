# Venera — roadmap

Remaining and optional work. `REPORT.md` is the system of record (methods, results);
`project_goal.md` is the goal. The core pipeline and both deliverables exist (rotation
period 243.0216 ± 0.0006 d; period-aligned global stack at up to ~1.2 km/px); this lists
what would refine them.

## Standing principles

- **One body frame, all looks.** Project every look (all epochs, both polarizations,
  N and S pointings) into a single frame parameterized by
  `spin = (period, pole_RA, pole_DEC, W0)`. No subsetting — S-pointing looks are
  essential (N-only makes the pole/period ill-posed).
- **Per-look nuisance terms only from geometry/labels** (delay from `GEO:DELAY_OFFSET`,
  geometric `freq_scale`, per-session Doppler-centering model), never free parameters that can
  absorb signal.
- **Validation rule (non-negotiable):** no registration or rotation claim is real
  until the stacked imagery measurably sharpens / cross-look correlation rises. A
  lower fit residual or tighter σ is not sufficient. The two deliverables police each
  other.
- **Arecibo data alone — no external DEM.**

## Open work

1. **Joint coherence fit + centering floor — DONE** (`scripts/period_joint_coherence.py`,
   `reproject_scp_ocpcentered.py`; §6). The period is now a single-parameter slope fit
   over all 15 pairs, and the centering floor was pushed σ_o 0.072°→0.040° by centering
   each look from its OC twin and registering the geometry-robust SC stacks. Remaining
   limit: **1988 is the only pre-2001 epoch**, so the long baselines share it and its
   offset doesn't average down — needs more early epochs or an external early rate anchor,
   not better centering.
2. **Geometry-multiplicity degeneracy mask** in `projection.py` (currently fixed
   angular cutoffs + 7° SRP exclusion).
3. **Imagery error bars.** Per-epoch bootstrap half-stack metrics; calibrate a
   realistic registration-significance threshold for big-feature maps.
4. **Full-resolution global stack** (`scripts/hi_res_global.py`, 16000×32000,
   ~1.2 km/px) — finish/refresh as needed.

## Deferred / out of scope

- External Magellan topography — out of scope by direction. If relief is ever needed
  (e.g. as a cross-geometry residual), solve it from the Arecibo data itself.
- Forward-model joint LSQR inversion over all epochs (true super-resolution + N/S
  ambiguity removal).
