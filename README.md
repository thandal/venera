# Venera : Venus Planetary Radar Astronomy

Analysis of 32 years of Arecibo planetary-radar observations of Venus (1988–2020) to
measure its **rotation elements** and build a **registered global radar image**.

Writeup: [Amateur Estimates of the Rotational Elements of Venus using 32 years of Arecibo Planetary Radar](https://docs.google.com/document/d/18l8Y1FTjtgzgq0ohklEkboS9BqEBN5sGElGdCluKGRo)

## What this is

[Planetary radar astronomy](https://en.wikipedia.org/wiki/Radar_astronomy) maps
celestial bodies with active radar. From 1988 to 2020 Arecibo intermittently observed
Venus in delay-Doppler mode; the data are
[archived at the PDS Geosciences Node](https://pds-geosciences.wustl.edu/missions/venus_radar/index.htm)
(Bruce A. Campbell & Donald B. Campbell 2022, *Planet. Sci. J.* 3 55,
doi:10.3847/PSJ/ac4f43). This repository projects each coherent look into a Venus
body-fixed frame, finds the spin (period and pole) under which all looks co-register,
and stacks them into a global image — the rotation elements and the crisp image are
two readouts of the same registration (see `project_goal.md`).

## Results

- **Sidereal rotation period: P = 243.0206 ± 0.0007 d**, pole adopted from IAU
  (RA 272.76°, Dec 67.16°) — consistent with Campbell 2019 (243.0212) and Margot 2020
  (243.0226), above the Magellan/IAU 243.0185.
- **Global stack** of all 916 looks, both latitude hemispheres (−85°…+83°), at up to
  ~1.2 km/px (`results/figures/global_*`, `results/session_stacks/global_*.npz`).

Coverage is **both north and south** but only the **Earth-facing ~185° of longitude**
— Earth-based radar sees only the sub-Earth disk, and the six inferior-conjunction
epochs sample a limited longitude range.

See **`REPORT.md`** for methods, conventions, validation, and the full result.

## Code layout

- `venera/` — tested library (cspyce + DE440s geometry, projection, registration,
  rotation fit, stacking). The durable code; run `.conda/bin/python test/run_all.py`.
- `scripts/` — drivers (project all looks → stack → period estimate → global map);
  see `scripts/README.md`.
- `PLAN.md` — remaining/optional refinements.

Example delay-Doppler image:
![venus_ocp_20150813_161747_small.png](/figures/venus_ocp_20150813_161747_small.png)
