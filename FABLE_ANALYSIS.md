# FABLE_ANALYSIS — review of the venera period result and writeups

*Claude Fable 5, 2026-07-01. A static code/writeup review (this checkout had no
`results/`, no `arecibo_radar/` data, no `.conda`), with all arithmetic re-derived
independently. This file is written for follow-on agents: each finding carries its
evidence and file references, and each work item has an acceptance criterion.
Read `project_goal.md`, `REPORT.md`, and `PLAN.md` first; this file critiques them.*

## Verdict in three sentences

The qualitative conclusions hold: P sits above IAU/Magellan, is consistent with
Campbell 2019, the pole is not constrainable from this data (adopt IAU), and
rotation-change is a null result. The specific headline **P = 243.0206 ± 0.0007 d is
model-dependent at ~1σ**: a defensible alternative noise model gives 243.0213, so the
honest quote today is ~243.021 ± 0.001 until items W1–W3 below are done. One imagery
claim (REPORT §9 "under one pixel") is arithmetically wrong at the advertised
resolutions and the global stacks need a rebuild at the adopted period.

## Useful constants (verified)

- dω/dP = 360/P² = **0.006096 °/day per day of period** at P = 243.02.
- Leverage dP/dm = P²/(360·Δt): **34.6 (13 yr), 18.7 (24 yr), 16.6 (27 yr),
  15.5 (29 yr), 14.0 (32 yr) mday per degree** of longitude misregistration.
- Shared 1988 offset of σ_o = 0.040° → ΔP ≈ **0.62 mday**; in quadrature with the
  formal 0.4 → **0.74** ≈ the quoted ±0.0007. So the report's widening is numerically
  right but derived nowhere — write it down (see W8).
- LOD oscillation (Margot 2021, ~20 min amplitude): accumulated longitude wobble is
  Δω·τ/2π ≈ **0.0016° (τ=117 d) to 0.0079° (τ=584 d)** — far below the 0.040° floor.
  σ_o is instrumental, not geophysical, and the mean-rate P is robust to LOD.
- Δt²-weight share of the 15 pairs: **long (>20 yr, all 1988-anchored) 72%,
  everything else 28%.**

## Findings (ranked)

### F1. The adopted P is pulled ~1σ low by short/mid baselines under a single global σ_o

REPORT §6's own table: long-baseline-only fit **243.0213** (≈ Campbell 243.0212);
adopted joint fit **243.0206**. The difference (0.7 mday) equals the whole quoted
uncertainty. Cause: the intrinsic-scatter model in
`scripts/period_joint_coherence.py:177-183` applies **one σ_o to every pair**, but the
per-session centering floor must scale with session quality — 2001 (22 looks, REPORT:
"its pair-periods scatter to ≈243.019") should have a floor several times 2017's.
Under Δt² weighting the non-1988 pairs still carry ~28% of the weight and drag the
estimate down. A heteroscedastic model would down-weight exactly those pairs.

Also: the "long-baseline only" row is computed with the *naive* weights `var0`, not
`var_adopt` (`period_joint_coherence.py:198`) — inconsistent with the adopted row it
is compared against.

### F2. The 15 pairs are treated as independent; they are not

All pairs derive from 6 session stacks, so Cov(m_ij, m_ik) = σ_o² when a session is
shared (every long baseline shares 1988). The diagonal-variance fit, the χ²/dof = 1
bisection for σ_o, and the **pair bootstrap** (`period_joint_coherence.py:186-189`)
all assume independence — the bootstrap σ_P is not trustworthy. The report compensates
with the ad-hoc 0.0004 → 0.0007 widening, which happens to match the shared-1988
term (see constants above), but a random-effects GLS produces it natively *and*
changes the point estimate via the correct weights.

Nit: σ_o is tuned on pair residuals, and Var(o_i − o_j) = 2σ²_session, so the
**per-session** floor is ~0.028°, not 0.040°; REPORT §6's "per-session centering
floor σ_o = 0.040°" mislabels a pair-level quantity.

### F3. REPORT §9 "under one pixel" is false at the advertised resolutions

Stacks are baked at P = 243.0216 (`scripts/hi_res_global.py:24`,
`scripts/global_stack_aligned.py:17`) vs adopted 243.0206. ΔP = 1.0 mday accumulates
**0.071° across the 32-yr span** (0.036° vs the mean-epoch reference). Per grid:

| product | deg/px | full-span misalign | vs-ref |
|---|---|---|---|
| session stacks 2000×4000 | 0.0900 | 0.8 px | 0.4 px |
| global_hires 8000×16000 | 0.0225 | 3.2 px | 1.6 px |
| global_fullres 16000×32000 | 0.0112 | **6.3 px (~7.5 km)** | 3.2 px |

The claim holds only at quarter-res. The "~1.2 km/px" headline product cannot deliver
1.2 km sharpness as built. The in-code comments "sub-px difference" on those two
lines are likewise wrong.

### F4. Period validation is partially circular; the declared gate is not the gate used

`venera/coherence.py` designates half-split **`frc_hi`** as the non-negotiable
imagery gate, but `period_joint_coherence.py:208-221` validates with
`stack_sharpness` — computed on the same stacks the period was fit to. A sharpness
maximum at the fitted P is close to tautological. No held-out validation exists. The
interior-NCC band is lat +30…+75 (`LATMIN, LATMAX` at `period_joint_coherence.py:44`),
so the period fit never touches the southern hemisphere — an untapped held-out set.

### F5. The headline estimator is untested script code with two accuracy soft spots

`venera/rotation_fit.py` is tested but is **not** what produced 243.0206; the m_ij
extraction, σ_m-from-curvature heuristic, σ_o bisection, and bootstrap live only in
`scripts/period_joint_coherence.py`. Soft spots:

- σ_m: `sigP = sqrt(2·sigN/k)` treats NCC-grid residuals about a quartic as
  independent noise; adjacent grid points are highly correlated.
- `ndshift(..., order=1)` bilinear shifts (`:112-113`) impose shift-phase-dependent
  blur → a ripple in NCC-vs-P at roughly the pixel period. 1 px = 0.09° at 2000×4000
  ↔ 1.3–1.7 mday over the long baselines — same order as the per-pair σ_P. Use
  Fourier shifts.

### F6. Everything rests on 1988, and 1988 has unique, unaudited processing

The jackknife honestly shows collapse to 243.0192 without 1988. But 1988 is also the
session with a filename-keyed Doppler `fliplr` hack (`venera/data.py:66`), a wider
symmetry-roll search (`data.py:126`, ±100 vs ±50), and known-noisy "autofocus" data
(`venera/selfcal.py` docstring). A 1988-specific centering bias propagates to P at
14–16 mday/° with nothing downstream to catch it.

### F7. fo centering quantization sits at the claimed floor

The Doppler-centering grid search runs at 2-column steps
(`venera/projection.py:238`, `fo_range = range(-60, 61, 2)`) with integer sampling
inside `fit_delay_doppler_curve` (`projection.py:74`, `.astype(int)`). 1 column ≈
0.014° of longitude at disk center, so the quantization scale (~0.028°) ≈ the
per-session floor (~0.028°, see F2 nit). The floor may be partly self-inflicted.
Note the OC-twin fits that set the SC stacks' centering
(`scripts/reproject_scp_ocpcentered.py:64`) use this default 2-col grid.

### F8. Writeup hygiene / consistency

- README links the **Feb-2023 Google Doc** as "the writeup"; the repo has since
  retracted its pole detection (272.77/67.15 — `scripts/pole_sensitivity.py:59`
  says "not detections"), its "speeding up" claim, and its ±0.003 d uncertainty.
  The doc needs a banner or an update.
- Stale numbers: `PLAN.md` intro says "243.0216 ± 0.0006"; `scripts/period_errorbars.py:66`
  plots "adopted 243.0203". REPORT is the system of record at 243.0206 ± 0.0007.
- `project_goal.md` promises W₀ with rigorous uncertainties; the result silently
  drops it. State explicitly: W₀ is gauge-fixed to IAU at the projection, and the
  final map's absolute longitude system is IAU-W₀-at-reference-epoch (d_ref ≈ mean
  of session epochs) under the adopted P — otherwise users cannot tie the map to
  Magellan coordinates.
- All substantiating logs/figures are gitignored (`results/`); the repo's numeric
  claims (σ_o values, NCC tables, the §6 table) are not reproducible from the repo.

## What holds up (do not re-litigate)

- Geometry validation is genuinely strong (`test/test_geometry.py`): body frame vs
  `IAU_VENUS` 2e-14, SRP vs `subpnt`, Δlon = −δω·Δt premise, topocentric-station
  regression guard.
- The rank-deficiency diagnosis (free per-session offsets ↔ slope) in
  `period_joint_coherence.py:5-14` is correct; through-origin slope + scatter floor
  is a reasonable (if approximate) response. The *right* fix is a prior on offsets
  (W1), not free offsets.
- Label-based nuisance policy (delay from `GEO:DELAY_OFFSET`, geometric
  `freq_scale`, diurnal-fo for 2012) is sound and well-argued in REPORT §4, §10.
- Pole adoption (REPORT §7) is justified; the OC-centered-SC decoupling (§6) is a
  clever, evidenced move (σ_o 0.072° → 0.040°).
- LOD reasoning in §8 is right, and the constants above strengthen it.

## Work items (prioritized, with acceptance criteria)

**W1. Random-effects joint refit (fixes F1+F2).** Model m_ij = c·Δt_ij + o_i − o_j +
ε_ij with o_k ~ N(0, σ_o,k²), ε_ij ~ N(0, σ_m,ij²). Marginalize the o_k analytically
(6 sessions — the marginal covariance of m is Σ = diag(σ_m²) + B·diag(σ_o,k²)·Bᵀ with
B the ±1 incidence matrix; GLS for c) or MCMC. Let σ_o,k scale per session (fit by
ML, or ∝ 1/√n_looks as a prior). Report P vs minimum-baseline cutoff as a
specification test. *Accept when:* the fit reproduces both REPORT rows (0.0206 and
0.0213) as limiting cases, the shared-1988 widening emerges from the covariance (no
hand widening), and the P-vs-cutoff curve is flat within its bands.

**W2. Rebuild global stacks at the adopted P; re-gate with FRC (fixes F3, F4).**
Change `P_ADOPT` in `hi_res_global.py` / `global_stack_aligned.py` to the W1 value
(or parameterize; better, store per-session sums so re-alignment is a column roll).
Validate with `coherence.half_split` `frc_hi`, not `stack_sharpness`, at all three
resolutions. Fix REPORT §9's parenthetical with the table from F3. *Accept when:*
`frc_hi`(adopted P) > `frc_hi`(243.0216) > `frc_hi`(IAU) at the hires scales, or the
report honestly states no measurable difference.

**W3. Empirical σ_m and per-session floors (feeds W1, fixes F5's σ_m heuristic).**
Bootstrap looks within each session → half-session stacks → recompute m_ij → the
scatter is a measured σ_m including centering noise; between-session spread of
per-half offsets gives σ_o,k per session. Replace bilinear `ndshift` with Fourier
shifts in the period scan. *Accept when:* measured σ_m are within ~2× the curvature
heuristic (else the old error bars were wrong and REPORT §6 needs its table redone).

**W4. Held-out validation (fixes F4).** Fit the period on the OC-centered SC stacks
(as now); evaluate the imagery gain on data the fit never saw: (a) the OC stacks,
(b) a held-out random half of looks, (c) the southern hemisphere (the interior band
is +30…+75 lat — check ≥2 sessions have S-pointing looks, then run the same
NCC-vs-P machinery on a southern band). *Accept when:* at least one held-out set
shows the peak at the adopted P within its error bar.

**W5. 1988 systematics audit (fixes F6).** (a) Split 1988 into SCP-anchored and
OCP-anchored period fits — two quasi-independent anchors; (b) inject a deliberate
±1-column fo bias into all 1988 looks, re-run the joint fit, report dP/dcol
(prediction from constants: ~0.014° × ~15 mday/° ≈ 0.2 mday/col); (c) verify the
`fliplr` and wide symmetry window don't shift the 1988 fo distribution vs other
sessions. *Accept when:* both sub-anchors agree within the per-session floor and the
injected-bias response matches prediction.

**W6. Sub-pixel fo (attacks F7, lowers the floor that sets the error).** Refine
`fit_delay_doppler_curve`'s fo by parabolic interpolation of the score about the
integer optimum (or a 0.25-col grid near the peak), and rerun
`reproject_scp_ocpcentered.py` → `period_joint_coherence.py scpocp`. *Accept when:*
σ_o (pair-level) drops measurably below 0.040°, or it doesn't and the floor is shown
to be genuinely physical/data-limited — either result is publishable in REPORT §6.

**W7. End-to-end synthetic test + move the estimator into the library (fixes F5).**
Forward-model delay-Doppler images from a known Spin and a synthetic feature field
(invert the `projection.py` mapping), run preprocess → project → stack → joint fit,
require recovery of the injected P within quoted σ. Extract the m_ij/σ_o/GLS
machinery from `period_joint_coherence.py` into `venera/` with unit tests (injected
offsets, known slope, coverage of error bars). *Accept when:* `test/run_all.py`
covers the code path that produces the headline number.

**W8. Writeup fixes (fixes F8, cheap, do alongside).** Banner or update the 2023
Google Doc; fix PLAN.md's stale number and `period_errorbars.py`'s label; add the
0.62-mday shared-anchor derivation to REPORT §6; state the W₀/longitude gauge of the
final map; commit the result logs (small text) or a manifest with hashes so REPORT's
tables are traceable.

**W9 (stretch).** Empirical pole×period cross-scan (P grid × pole offsets ±0.1°) to
demonstrate the period is insensitive to the adopted pole — turns §7's geometry
argument into a measurement. Per-epoch freq_scale sensitivity test: a 1e-3 scale
error mislocates limb-ward features by up to ~0.05°, near the error budget;
`scripts/subset_freqscale_test.py` exists but its results are not in REPORT.

## Environment notes for agents

- Run everything with `.conda/bin/python` from the repo root (conda env and
  `spice_kernels/` are symlinks, gitignored; see `.gitignore`).
- Data path expected by scripts:
  `arecibo_radar/pds-geosciences.wustl.edu/venus/arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/`
  (fetch via `wget_venus_data.sh`; ~916 looks × 2 pols × 256 MB — large).
- Pipeline order: `project_all_looks.py` → `reproject_2012_diurnal.py` →
  `stack_sessions.py` → `reproject_scp_ocpcentered.py` →
  `period_joint_coherence.py scpocp`. Caches are idempotent; delete
  `results/look_cache*` after any geometry change. `period_joint_coherence` also
  caches its period-grid curves (`period_joint_cache*.npz`) keyed only on PGRID —
  **delete it after rebuilding stacks** (stale-cache trap).
- Tests: `.conda/bin/python test/run_all.py` (needs kernels but not the radar data).
- The non-negotiable rule from `project_goal.md` binds all of the above: no
  registration/rotation claim is real unless stacked imagery measurably sharpens.
  W2/W4 are the enforcement of that rule, not optional extras.
