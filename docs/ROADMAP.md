# TurboChook — Roadmap

One milestone = one coherent, runnable step. Do them in order; each has a hard acceptance
gate. See [`DESIGN.md`](DESIGN.md) for the architecture every milestone must obey.

## M0 — Walking skeleton (retire the toolchain risk FIRST)

Goal: prove the build + GPU offload before any physics.

- CMake project, C++23, the three build configs (gpu / multicore / host-serial).
- A `saxpy` and a `std::transform_reduce` over a big array via `par_unseq`.
- Confirm GPU offload with **nsys** (kernels appear in the timeline) — not just "it ran".
- `std::mdspan` availability check (real `std::` or vendored `kokkos/mdspan`).

**Acceptance:** builds in all three configs; the GPU config shows kernels in nsys; the
reduction result matches the serial result. *No solver yet — this is the de-risking step.*

## M1 — 2D scalar field on a Cartesian grid

Goal: the data model + iteration + time loop, end to end, on the simplest PDE.

- `Field2` (`mdspan<layout_left>`) — start with a simple owning `Field` (owns a `vector`);
  Arena comes in M2/M3.
- `for_each_cell`, a linear-advection **or** heat-diffusion stencil, SSP-RK2, a
  double-buffer.
- Output: raw binary or CSV dump (host-side); a tiny Python/matplotlib viewer is fine.
- Host-serial analytical test.

**Acceptance:** converges at the expected order; conserves the scalar to ~machine eps;
runs on GPU and host-serial with matching results.

## M2 — 2D shallow-water (the real target)

Goal: a genuine Rakali analog.

- `EquationSet SWE` (N=3), `RiemannSolver` = **HLL first, then HLLC** (add the contact
  correction; consider the `Vec<N>` upgrade here for the star-state math).
- SSP-RK2 integrator (as an `Integrator` policy — open decision #2).
- Boundaries: **reflective** + **periodic** as `BC` policies with halo-row fill (open
  decision #3).
- Runtime→compile-time dispatch via `variant`/`visit` for the flux choice.
- Test cases (DESIGN §10): **dam-break vs exact Riemann**, **lake-at-rest**
  (well-balanced), **radial symmetry** of a Gaussian drop, **mass conservation**.

**Acceptance:** dam-break matches the exact SWE Riemann solution within scheme tolerance;
lake-at-rest stays flat (well-balanced); Gaussian drop stays radially symmetric; total mass
drift ~ machine eps; GPU offload confirmed in nsys.

## M3 — Consolidate the architecture

Goal: prove the seams.

- Swap the owning `Field` for the **Arena** (open decision #4: a `Workspace` owning
  `U`/`U1`/`K`) — **zero kernel changes** (the proof the boundary holds).
- Add a second `Integrator` (Fwd-Euler or RK3) — flux untouched.
- (Optional) `Vec<N>` upgrade if not already done.

**Acceptance:** Arena swap-in changes no kernel; results bit-stable vs M2; a second
integrator drops in without touching flux/physics.

## Later (not scheduled)

- `EquationSet Euler` (N=4) to exercise N-genericity.
- 3D (`Field<3>`), custom mdspan layouts (AoSoA), more BCs.
- These are explicitly out of near-term scope — see DESIGN §11.
