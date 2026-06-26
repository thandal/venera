# Project goal

## What we are doing

Analyze **all** available Arecibo planetary radar data of Venus to produce two
deliverables that are really one:

1. The best, most rigorously characterized estimate of the **rotation elements** of
   Venus — sidereal period `P`, north-pole direction (`pole_RA`, `pole_Dec`), and
   prime-meridian phase `W₀` — with scientifically rigorous uncertainties.
2. The **crispest, most accurately registered global image** of Venus, by stacking
   **all available looks** together.

## Why these are one problem, not two

The correct rotation elements are exactly the body-frame parameters under which
every look co-registers. That co-registered stack *is* the crisp image. We do not
measure the spin and then separately make a picture — **we find the spin (and pole
and phase) that make the picture sharp, and read both deliverables off the same
optimization.**

## Method (one pipeline)

- One body frame parameterized by `spin = (P, pole_RA, pole_Dec, W₀)`.
- Project **every look** into that frame — all epochs (1988–2020), both
  polarizations (SCP + OCP), and **both N and S pointings**. No subsetting. (The
  S-pointing looks are essential: N-only coverage makes the pole-rotation component
  degenerate with tilt and the period ill-posed.)
- Per-look nuisance terms only where the PDS labels justify them (e.g. Doppler
  centering) — not free parameters that can absorb signal.
- Find the `spin` parameters that maximize **stack coherence** (looks mutually
  register; combined sharpness / cross-look correlation maximized).
- Rotation elements = the optimum (+ bootstrap over looks). Final image = the stack
  at the optimum.

## The non-negotiable validation rule

**No registration or rotation-elements claim is real until the stacked imagery
measurably sharpens** (or cross-look correlation rises). A lower fit residual or a
tighter σ is *not* sufficient — those are trivially faked by adding ill-posed
degrees of freedom. The imagery is the validation of the science; the two
deliverables police each other. Report faithfully; never let an unvalidated number
into the writeup.
