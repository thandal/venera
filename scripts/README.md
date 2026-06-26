# scripts/ — drivers for the pipeline

Run with `.conda/bin/python scripts/<name>.py` from the repo root. The library
(`venera/`) + tests (`test/`) are the durable code; these are the drivers. See
`REPORT.md` for methods and results.

## Build the products
- `project_all_looks.py` — project every Arecibo look (all epochs, both pols, N+S)
  into the body frame; cache each as a cropped half-res map (`results/look_cache/`).
  Parallel, idempotent (delete the cache to force a re-run after a geometry change).
- `reproject_2012_diurnal.py` — reproject the bistatic 2012 looks with the
  per-session Doppler-centering model (`fo` vs time-of-day; overwrites their cache
  entries).
- `stack_sessions.py` — per-session and combined σ-clipped stacks + PNGs.
- `global_stack_aligned.py` — period-aligned global stack (both hemispheres) at
  2000×4000 / 8000×16000.
- `hi_res_global.py` — full-resolution period-aligned global stack
  (16000×32000, ~1.2 km/px; strip-processed, checkpointed).

## Rotation elements
- `period_solve_all.py` — **adopted period**: forward-project at trial periods +
  interior overlapping-disk correlation, all session pairs.
- `period_errorbars.py` — per-pair period error bars from the interior-NCC curve
  width (1988-anchored).
- `pole_sensitivity.py` — geometry-only pole leverage → venera cannot beat the IAU
  pole (adopt IAU).
- `rotation_elements.py`, `run_rotation_estimate.py` — `register_maps`
  longitude-drift fit; **sanity check only** (full-disk offsets run high vs the
  interior-correlation method above).

## QA / spot checks
- `assess_registration.py` — interior-NCC registration-quality matrix across sessions.
- `register_on_sphere.py` — pairwise best-rotation registration on the sphere.
- `project_one.py` — single `.img` → global-map PNG.

## Diagnostics (one-off investigation tools, not the routine pipeline)
`characterize_fo.py`, `period_scan.py`, `interior_region.py`, `blur_2012.py`,
`halfstack_check.py`, `intra_session_check.py`, `maxwell_panels.py`,
`maxwell_drift.py`, `maxwell_look_scatter.py`, `single_look_compare.py`,
`flip_figures_2012.py`, `render_session_flip.py`, `reproject_2012_bistatic.py`,
`fo_diurnal_fix.py`, `freq_scale_check.py`, `subset_freqscale_test.py` — used to
develop and validate the centering, 2012-bistatic, and period work.

## archive/
`scripts/archive/` — earlier scaffolding, reference only; not part of the pipeline.
