# Venera — system of record

This is the working record for **venera**: its methods, conventions, validation, and
results. The pipeline analyzes every Arecibo planetary-radar observation of Venus
(1988–2020) to recover two things from a single registration — the planet's **rotation
elements** (sidereal period *P*, the pole's right ascension and declination, and the
prime-meridian phase *W₀*) and one co-registered global image.

The raw data are 916 coherent **looks**, each a ~5-minute observation recorded as a
*delay–Doppler* image: radar maps the visible disk by round-trip **delay** (distance
from the point facing the radar) against **Doppler** shift (line-of-sight velocity from
the planet's apparent rotation). `project_goal.md` states the goal and `PLAN.md` tracks
what remains; this file records what exists. Run everything with `.conda/bin/python`
from the repo root.

**Result in one line:** sidereal period **P = 243.0206 ± 0.0007 d** (pole adopted from
the IAU), plus a period-aligned global stack of all 916 looks spanning both hemispheres
(latitude −85° to +83°, ~185° of longitude) at up to ~1.2 km/px.

## 1. Architecture

The durable code is a tested library under `venera/`; each stage is validated on its own:

| module | role | key validation |
|---|---|---|
| `spice_setup.py` | cspyce kernel furnishing (DE440s, leapseconds, PCK) | — |
| `geometry.py` | parameterized spin → body frame, sub-radar point, bistatic `(ô,d̂)` Doppler basis, geometric `freq_scale` | body frame vs SPICE `IAU_VENUS` to 2e-14 |
| `data.py` | `.img`/`.lbl` load + preprocessing (ephemeris label-delay centering) | runs on real data |
| `projection.py` | bilinear delay-Doppler → global lon/lat map; per-look curve fit; diurnal-`fo`; strip-processed for high res | real look → recognizable Venus |
| `registration.py` | band-pass + sub-pixel masked cross-correlation + significance | 0.01 px shift recovery |
| `rotation_fit.py` | longitude-drift LSQ → period + propagated σ + bootstrap | injected P recovered to 0.03 mday |
| `coherence.py` | stack coherence / NCC metrics | injected misalignment lowers metric |
| `selfcal.py` | diagnostic limb-Doppler self-cal (not in the default centering path) | injected center recovered to 0.1 px |

Run the suite with `.conda/bin/python test/run_all.py` (geometry, registration,
rotation_fit, selfcal, coherence — all green).

## 2. Conventions

- **Time.** SPICE `et` is TDB seconds past J2000, so days past J2000 are `et/86400`.
  Each look is timestamped at its START/STOP midpoint, and the body orientation is
  evaluated at the emission epoch `et − lighttime`.
- **Frames.** Positions and velocities come from DE440s in J2000. The Venus body-fixed
  frame is built in-house from `Spin(period, pole_ra, pole_dec, w0)` by the IAU 3-1-3
  Euler construction `M = Rz(W)·Rx(90°−δ0)·Rz(90°+α0)` — identical to SPICE's
  `IAU_VENUS` at nominal constants, but with the spin left free to fit.
- **Stations (topocentric, not geocentric).** The radar transmits from **Arecibo** and
  receives at Arecibo (monostatic), except for the **2012 campaign, which received at
  Green Bank (GBT)** — bistatic. Each station is built as Earth-center plus a
  `georec`/`pxform(IAU_EARTH→J2000)` site offset, so a finite difference recovers its
  diurnal velocity; that velocity is part of the apparent rotation that fixes the
  Doppler axis. For a bistatic look the effective sub-radar direction is
  `ô_eff = normalize(ô_tx + ô_rx)`.
- **Spin.** Venus rotates retrograde (`Ẇ < 0`); `period_days` is the sidereal magnitude.
  The IAU nominal constants (pck00011) are RA 272.76°, Dec 67.16°, W0 160.20°,
  Ẇ −1.4813688°/day — i.e. P 243.0185 d, the value baked into the raw projection.
- **Delay–Doppler geometry.** Two body-fixed unit vectors carry it: `ô` (sub-radar
  direction) and `d̂ = normalize(dô_eff/dt)` (Doppler gradient). A surface point `p`
  then has `delay ∝ 1 − p·ô` and `doppler ∝ p·d̂`, with apparent north `n̂ = ô×d̂`
  (sign fixed by `n̂_z > 0`) selecting the N or S hemisphere via `GEO:POINTING`.
- **Maps.** Equirectangular: row = `(lat+π/2)/π·H`, col = `(lon+π)/2π·W`.

## 3. Geometry — validation

The geometry is checked against SPICE and against first principles:

- The in-house body-fixed matrix matches SPICE `IAU_VENUS` to **max |Δ| = 2e-14**.
- The sub-radar point (SRP — the disk point directly facing the radar) matches SPICE's
  own sub-observer routine (`subpnt`) to **0.00 arcsec** across all six campaigns
  (Venus treated as a sphere, R = 6051.8 km).
- Doppler angles run 7–14°, consistent with Campbell's ~10°.
- The rotation premise `Δlon = −δω·Δt` holds to 3e-11. Its leverage over the 1988–2020
  baseline is ~15.6 mday of period error per degree of longitude misregistration.

## 4. Per-look centering & calibration

Every look needs a few nuisance terms fixed — where its zero-Doppler column and
sub-radar delay row sit, and how Doppler maps to longitude. We take these from geometry
and the PDS labels wherever possible, rather than fitting them freely where they could
absorb real signal:

- **Delay (latitude) centering** comes from the label `GEO:DELAY_OFFSET`, the
  ephemeris-derived round-trip-delay reference for the sub-radar point. For bistatic
  2012 it drifts by thousands of rows across the session, and the label tracks it exactly.
- **Doppler (longitude) centering** is the per-look pixel offset `fo` (with `do` for the
  sub-radar row). Monostatic looks fit it from the limb curve
  (`fit_delay_doppler_curve`). The 2012 bistatic looks instead use a robust per-session
  **diurnal model**, `fo(hour) = a + b·(hour − h0)` (`fit_diurnal_fo`), because there the
  residual Doppler centroid is a smooth function of time of day — a small leftover in the
  providers' time-varying tuning — and the per-look limb fit is unreliable on the low-SNR
  bistatic limb. For 2012, `fo ≈ +5.8 − 8.5·(hour − 17.38)` columns
  (`scripts/reproject_2012_diurnal.py`).
- **Doppler scale** `freq_scale` is predicted from geometry — the apparent-rotation
  bandwidth (`predicted_freq_scale`, one calibration constant `K = 1.001`) — and made
  **bistatic-aware** through `rx_station` so the scale matches the actual Doppler axis.
  A monostatic-only scale would stretch 2012's longitudes by ~0.5° at the limb.

## 5. Stacking

Looks are combined with a per-pixel **3σ-clipped robust mean**
(`scripts/stack_sessions.py`). For the global stack, each look is first
longitude-shifted to the adopted period so the epochs co-register
(`scripts/global_stack_aligned.py`, `hi_res_global.py`). The projector works in latitude
strips (`lat_chunk`), so the output grid and mesh density can scale to the data's native
~1–2 km resolution without exhausting memory.

## 6. Rotation period — P = 243.0206 ± 0.0007 d

A change in period is exactly a per-look rotation in longitude, so the period we want is
the one under which every session co-registers. For each pair of sessions we
forward-project at a grid of trial periods and find the **interior overlapping-disk NCC**
peak — NCC being normalized cross-correlation (1 = identical), measured over the interior
that both maps cover with the coverage edges eroded away so they can't bias the score.
That peak gives `m_ij`, the longitude shift that best aligns the pair, and its curvature
gives `σ_m`. The most period-sensitive pairs anchor on **1988**, the only pre-2001 epoch
(114 looks):

| 1988↔ | Δt (yr) | peak P (d) | peak NCC |
|---|---|---|---|
| 2001 | 13 | 243.0240 | 0.902 — 22 looks, downweighted |
| 2012 | 24 | 243.0215 | 0.943 |
| 2015 | 27 | 243.0220 | 0.939 |
| 2017 | 29 | 243.0206 | 0.941 |
| 2020 | 32 | 243.0211 | 0.955 |

**The joint fit** (`scripts/period_joint_coherence.py`) uses all 15 pairs at once. A
wrong period offsets each session's longitude in proportion to its epoch, so
`m_ij = c·Δt_ij`, and the slope `c` gives the period. We fit that single slope through the
origin with **no free per-session offsets**, and the choice matters: giving each session
its own offset is rank-deficient (an offset pattern linear in epoch is indistinguishable
from a period change), while averaging the per-pair periods is biased, because a
per-session centering error `o_k` shifts a pair's apparent period by ~`o_k/Δt` and so
hits the short baselines hardest. The bare slope fits poorly (**χ²/dof ≈ 6**): the shifts
don't lie on one line, which is the signature of a real **non-rotational per-session
centering floor**, `σ_o`, that we carry as an intrinsic-scatter term. Folding it into the
weights makes them scale as Δt², so the long baselines — where the ~0.2° rotation signal
dwarfs the centering noise — dominate.

**Polarization sets that floor** (`scripts/reproject_scp_ocpcentered.py`). The radar
records both senses of each echo: **OC** (opposite-circular — quasi-specular, strong) and
**SC** (same-circular — diffuse, weak). SC features sit on the surface and survive a
change of viewing geometry, whereas OC carries a quasi-specular glint that tracks the
sub-radar point and decorrelates between epochs (measured: SC cross-epoch NCC 0.96 vs OC
0.93 on the most geometry-diverse pair). But SC's weaker per-look signal makes its
centering noisier. So we use each channel for what it does best — center every look from
its **simultaneous OC twin** (strong signal, precise `fo`) and register the
geometry-robust **SC** stacks. This decoupling drops the centering floor from **σ_o =
0.072°** (both senses mixed) to **0.040°**, below either channel alone (OC 0.060°, SC
0.105°), because it removes both error sources at once. The global *image* (§9) still
stacks all looks for SNR; only the period fit uses the OC-centered SC stacks.

| estimator (decoupled-hybrid stacks) | P (d) | |
|---|---|---|
| naive slope (curve-noise weights only) | 243.0201 | χ²/dof≈6 — rejected |
| **joint slope + offset-jitter (adopted)** | **243.0206** | formal ±0.0004 |
| long-baseline pairs only (Δt > 20 yr) | 243.0213 | |
| combined-pol joint fit (cross-check) | 243.0207 | σ_o = 0.072° |

**The honest uncertainty is ±0.0007, not the formal ±0.0004**, because the whole result
rests on 1988. Dropping any of the other five sessions leaves the period at
243.0205–243.0211, but dropping 1988 collapses it to 243.0192 — the post-2001 baselines
are simply too short to measure a period. Since every long baseline shares the 1988
anchor, 1988's own centering offset moves them all together and does not average down, so
we widen the formal ±0.0004 to ≈ ±0.0007.

**2001** (22 looks, the smallest and noisiest session) tests the centering floor rather
than the period: its short baselines (Δt 11–19 yr) divide that offset by a small Δt, so
its pair-periods scatter to ≈243.019 and it carries no weight in the result.

| source | period (days) |
|---|---|
| Magellan / IAU (baked into the projection) | 243.0185 |
| **venera (this work)** | **243.0206 ± 0.0007** |
| Campbell et al. 2019 | 243.0212 ± 0.0006 |
| Margot et al. 2020 | 243.0226 ± 0.0013 |

venera agrees with Campbell to within 1σ and sits above the IAU/Magellan value, but it
lacks the precision to separate the modern radar determinations from one another.

## 7. North pole — adopt IAU

These data barely constrain the pole. On the most SRP-diverse pair
(`scripts/pole_sensitivity.py`), a feature moves 1.27° per degree of declination but only
0.16° per degree of right ascension; against ~0.16° of registration noise, that pins the
venera pole to only ~±0.15° in Dec and ~±1.2° in RA — 10–100× coarser than the
IAU/Magellan pole (±0.01°). So **venera adopts the IAU pole (RA 272.76°, Dec 67.16°).**

## 8. Has Venus' rotation changed? — null result

Earth-based radar clusters at 243.022–243.023 d, whereas Magellan (1990–1994) measured
243.0185 d. Venus' super-rotating atmosphere trades angular momentum with the solid
planet through solar thermal tides, producing length-of-day swings of up to ~20 minutes
about the mean (Margot et al. 2021); a determination over a short baseline can therefore
differ from the long-term mean by minutes, simply through which phase it happens to
sample. A steady drift large enough to explain the Magellan-to-modern offset would be
~0.2 mday/yr — far below venera's per-session precision. venera resolves neither a
secular drift nor the LOD swings: at this precision, a **null result**.

## 9. The global image

The image stacks all 916 looks — six epochs (1988–2020), both polarizations, and N and S
pointings — period-aligned and combined with the σ-clipped robust mean. (The on-disk
stacks were built at P = 243.0216; the 0.0010 d gap to the adopted 243.0206 is under one
pixel over the full 32-year baseline, so the imagery is unaffected.)

- **Latitude — both hemispheres**, −85° to +83°: N-pointing looks map the north and
  S-pointing the south, and a faint equatorial seam marks where the two
  separately-imaged hemispheres meet across the sub-radar gap.
- **Longitude — a ~185° swath** (~−115° to +70°E), not the full globe. Earth-based radar
  only ever sees the sub-Earth disk, and the six inferior-conjunction epochs sample a
  limited range of sub-radar longitudes; the full 360° is not recoverable from this data.
- **Resolution products** (`results/session_stacks/`, `results/figures/`): session and
  combined stacks at 2000×4000 (~9.5 km/px), `global_hires` at 8000×16000 (~2.4 km/px),
  and `global_fullres` at 16000×32000 (~1.2 km/px, the data's finest mode).

## 10. The 2012 bistatic campaign

2012 is the only bistatic campaign (Arecibo transmits, GBT receives; `GEO:MODE="B"`), and
it is handled with the GBT receiver geometry (§2) plus the per-session diurnal-`fo`
centering and bistatic-aware `freq_scale` (§4). With those, 2012 registers and looks
near-monostatic: per-look Maxwell scatter is ±0.09° against 2017's ±0.05°, where the
diurnal-`fo` fix brought it down from ±0.27°. A residual softness against the monostatic
sessions remains, and it is intrinsic to the data — delay-Doppler imaging is
N/S-ambiguous, the mirror hemisphere is normally suppressed by beam pointing, and GBT's
wider beam suppresses it less (per the PDS archive description), leaving a faint
contrast-reducing haze and modestly lower SNR. 2012 is fully usable, and its session-mean
feature positions are sound for the rotation elements.

## 11. References

- B. A. Campbell & D. B. Campbell (2022), *Planet. Sci. J.* 3, 55 — the PDS data set
  (`arecibo_radar/doc/venus_radar.pdf` locally).
- B. A. Campbell et al. (2019) — mean rotation rate of Venus from 29 years of radar.
- J.-L. Margot et al. (2020/2021) — spin state, moment of inertia, and length-of-day
  variations of Venus from radar speckle.
- M. E. Davies et al. (1992) — Magellan SAR (IAU value);
  see `Estimates of the rotation period of Venus.csv`.

## 12. Open items

- **More early epochs.** The period uncertainty is no longer set by the centering floor
  (the all-15-pair joint fit and the OC-centered-SC decoupling are done — §6) but by
  **1988 being the only pre-2001 epoch**: every long baseline shares it, so its offset
  doesn't average down. No Arecibo data can fix this; the only lever is an external early
  rate anchor (e.g. Magellan's ~1992 point) — a different question from the no-external-
  DEM rule, since it is a scalar rate, not topography.
- **Geometry-multiplicity degeneracy mask** in `projection.py` (currently fixed angular
  cutoffs + 7° SRP exclusion).
- **Imagery error bars.** Registration significance is naturally low on big-feature maps;
  add per-epoch bootstrap half-stack error bars to the imagery metrics.
