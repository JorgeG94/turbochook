# TurboChook — Roadmap

One milestone = one coherent, runnable step. Do them in order; each has a hard acceptance
gate. See [`DESIGN.md`](DESIGN.md) for the architecture every milestone must obey.

## M0 — Walking skeleton (retire the toolchain risk FIRST)

Goal: prove the build + GPU offload before any physics.

- CMake project (the `CMakeLists.txt` sketch exists), C++23, the three build configs
  (`TC_STDPAR` = gpu / multicore / off).
- `lib/numerics/parallel.hpp` — the `tc::par` execution-policy seam (`par_unseq`, or `seq`
  under `TC_STDPAR_OFF`) + `for_each_cell`.
- A `saxpy` and a `std::transform_reduce` over a big array via `tc::par`.
- **doctest wired via CTest** (`FetchContent` + `doctest_discover_tests`) with one trivial
  passing test, so the test harness is proven before any physics.
- Confirm GPU offload with **nsys** (kernels appear in the timeline) — not just "it ran".
- `std::mdspan` / `std::print` availability check (real `std::`, else the hand-rolled
  `tc::mdview` / logger fallback — **never Kokkos**) — depends on nvc++'s stdlib, so verify here.

**Acceptance:** builds in all three configs; `ctest` green (host); the GPU config shows
kernels in nsys; the reduction result matches the serial result. *No solver yet — de-risk the
toolchain, the offload, AND the test harness.*

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

## M2 — C-grid barotropic shallow-water (the PoC = the ocean core's foundation)

Goal: the proof-of-concept, and *not* a throwaway — this **is** the ocean core's barotropic
(fast) mode, so everything here is load-bearing for the north star.

- **Arakawa C-grid** staggered `BaroState` (η centres, u/v faces) from the Arena.
- Operators as compile-time policies (DESIGN §5): **`Continuity`** (PPM thickness flux),
  **`Coriolis`** (Sadourny PV-conserving), **`PGF`** (`-g ∇η`). Port the numerics from
  Rakali `src/core/ocean/`; the algorithm, not the source. **Exact file→procedure map:
  [`PORTING_MAP.md`](PORTING_MAP.md)** (items 1–6 + the driver are M2).
- **Split-explicit** time integration (`Integrator` policy: fast barotropic substeps) — start
  with a single explicit forward-backward if that's simpler, build to the split.
- Boundaries: **wall** + **periodic** as `BC` policies, halo-row fill before each stage
  (branch-free interior — decision #3).
- `Vec<N>` not needed yet (barotropic is scalar-per-face); introduce with M3 layers.
- Test cases (DESIGN §10, ocean-appropriate — NOT dam-break): **geostrophic adjustment**,
  **lake-at-rest** (well-balanced — flat η stays flat), a **Kelvin/Rossby wave**, **mass
  conservation**.

**Acceptance:** geostrophic adjustment reaches the correct balanced state; lake-at-rest stays
flat; the wave propagates at the right speed; total mass drift ~ machine eps; GPU offload
confirmed in nsys. This rung proves the C-grid + split-explicit machinery — the hard part.

## M3 — Two layers → baroclinic instability (the "it's a real ocean model" rung)

Goal: make **eddies from physics, not numerics** (the Rakali `ocean_eddy_certification`
lesson). A stacked-shallow-water **2-layer Phillips** model.

- Second layer; per-layer thickness + the co-located tracer path (`SystemView<N>`, S/T at
  centres); `PGF = gprime` (2-layer reduced gravity); barotropic/baroclinic split proper.
- Certify with a 2-layer baroclinic channel that spins up eddies.

**Acceptance:** a baroclinic channel goes unstable and produces eddies at the right scale;
energy/enstrophy behave; GPU-stable over a long run.

## M4 — EOS + FV pressure gradient + a vertical coordinate

- Nonlinear EOS; FV-PGF; a `Vcoord` policy (sigma / z* / ALE remap).

## M5 — Vertical mixing

- `Vmix` policy: PP81 interior → KPP boundary overlay.

## Later (the long arc toward full parity)

- More layers, more vcoords, tides, real forcing, tripolar/unstructured `Mesh`. This is a
  multi-month+ arc — Rakali's ocean core is enormous (see its CLAUDE.md). Climb one rung at a
  time; a stable GPU baroclinic core is already a serious result.
