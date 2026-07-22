# TurboChook — Status & Handoff

Current state of the project, plus the verified toolchain findings a fresh
contributor (human or agent) needs so they don't re-derive them. Read this after
[`DESIGN.md`](DESIGN.md) to pick up work.

---

## Where we are

**The north star is reached: TurboChook reproduces two-layer baroclinic instability
end-to-end on the GPU.** A geostrophic jet on a spherical band (59°N, periodic-x /
wall-y) destabilises, meanders, breaks, and rolls up into spiral eddies (a 70-day,
256² run, ~24 min of V100 wall time at ~20 s/sim-day, mass drift 0.1%). Everything
below it is host-green + GPU-offload-verified:

- **M2** single-layer operators (continuity/Coriolis/PGF) — validated on the plane
  AND the sphere (mass conservation, lake-at-rest, geostrophic-balance residual
  0.027%).
- **M3-stage-1** two-layer core — coupling PGF exact, barotropic + baroclinic wave
  speeds to 0.1%.
- **Periodic-x** via mesh-owned indexing (ADR-7): shift_x/shift_y/bc_at, seam-crossing
  validated.
- **M3-stage-2** the `demo_baroclinic` run itself.

### What's built

```
src/core/     types.hpp      Real/Index + Field<Rank> (mdspan ↔ MdView seam)
src/lib/      log/error/arena/profiler.hpp   plumbing (unchanged from M0)
src/numerics/ parallel.hpp   tc::par seam + for_each_cell/face
              integrator.hpp Integrator concept + SSPRK2 / SSPRK3 / ForwardEuler
                             — advance() is generic over the state type
src/mesh/     cartesian_mesh.hpp   CartesianMesh (uniform beta-plane)
              spherical_mesh.hpp   SphericalMesh (lat/lon, f=2Ω sinφ)
              masked_mesh.hpp      MaskedMesh (land/sea wet() mask)
              iterate.hpp          for_each_x_face / y_face (interior faces)
src/physics/  state/       baro_state (BaroState + Params), layered_state (LayeredState<NL>) [M3]
              continuity/  continuity (ContinuityFlux<Ppm>, telescoping), reconstruction (PCM…WENO)
              momentum/    coriolis (SadournyEnstrophy, PV+KE), pgf (FvPgf, −g∇η),
                           two_layer_pgf (reduced-gravity coupling) [M3], pgf_layered [stub → M4]
              vertical/    vcoord · remap · vmix            [stubs → M4/M5]
              tracer/      tracer · eos                     [stubs → M4/Later]
              lateral/     dissipation · lateral_mix        [stubs → M3.5/Later]
              forcing/     forcing                          [stub → Later]
              core/        ocean_core → BarotropicPoC, multilayer_core (MultilayerCore<NL>, unsplit) [M3],
                           split_multilayer_core (SplitMultilayerCore<NL>), barotropic (FB subcycler)
src/diag/     reduce.hpp + quantity/registry (the Registry) + report (Reporter) + diagnostics
src/bc/       bc.hpp  WallBC / PeriodicBC   (fold / sponge / obc stubs)
examples/programs/  demo_* / m0_walking_skeleton / reco_demo   (thin mains, each links the lib)
tests/        test_*.cpp     56 doctest cases
```

### Verified

**Single layer (M2), GPU-offload-confirmed (~30× over host on a big problem):**
- Gravity wave `c = √(gH)` to **0.003%**; geostrophic balance held to machine-eps.
- Mass conservation to ~1e-12 (telescoping continuity); Coriolis via Sadourny.
- Root-caused a blow-up to **SSP-RK2 instability on the imaginary axis** (gravity
  waves); the one-line policy swap to **SSP-RK3** fixed it — energy then conserved
  <0.2%. (RK2/Heun amplifies every oscillatory mode; RK3's region includes the
  imaginary axis to |ωΔt|≲1.73.)

**Two layer (M3), host-green:**
- Reduced-gravity PGF coupling: linear h₁,h₂ → **exact** −∇p₁, −∇p₂.
- Two-layer lake-at-rest → zero RHS (well-balanced across the coupling).
- Barotropic mode `√(g(H₁+H₂))` to 3%; **baroclinic (internal) mode
  `√(g'H₁H₂/(H₁+H₂))` to 0.1%** — the actually-new stratified physics.

### Key modelling facts established

- **Unsplit, external-CFL limited.** `dt < CFL·dx/√(g(H₁+H₂))` (fast external gravity
  wave). No barotropic subcycling — this matches the reference's `bc_inst` (`M=0`,
  `dt=6s`). Split-explicit is a future ~M× optimization, not required. A layered
  (isopycnal) core has **no vertical-advection CFL**.
- **The instability run must resolve `Rd_bc = c'/f`** (~17 km with H=200/800,
  Δρ=2). Under-resolving it makes the eddies rotation-smeared (the reference uses
  ~5 km). Wave-speed *tests* use f=0 to isolate the gravity wave.

---

## What the run taught us (for the follow-ups)

- **Seed placement is decisive.** A whole-domain cos·cos interface checkerboard
  mostly radiates far-field gravity waves and barely grows; a **front-localised
  sech²(y/L) meander** projects onto the jet-trapped mode and grows (e-fold ~5 days).
  `|v1|max` (base v≡0) is the clean growth diagnostic; `|v2|max` confirms coupling.
- **The scheme under-grows the instability ~10×** vs inviscid theory, because
  `Rd ≈ 21 km ≈ 4 cells` is only marginally resolved at 256² and PPM+Sadourny damp
  it. So the **exact-Julia weak jet (L=100 km, U=0.15) decays** rather than rolls up;
  a narrower/stronger jet (L≈40 km, U≈0.5–0.8) overcomes the damping. This is the
  main open quality issue.

## Next steps

1. **Recover the exact-Julia jet.** Close the ~10× growth gap so `L=100 km, U=0.15`
   goes unstable: (a) 512² to resolve `Rd`; (b) a **scale-selective closure**
   (biharmonic / Leith) instead of uniform Shapiro, which diffuses the broad jet;
   (c) a less-diffusive continuity option at the deformation scale.
2. **Factor dissipation into a tested module** (`src/physics/lateral/dissipation.hpp`): the
   Shapiro filter (currently inline in `demo_baroclinic`) + a Laplacian/Leith
   viscosity, each on its own policy axis, with unit tests.
3. **`RunConfig` struct + presets** (code-first) for the bc_inst / eq-wave cases.
4. **Vorticity / PV rendering** (sharper eddy cores than the interface view).

Deferred (per the user): full tracer chain (PCM→WENO9) + advection-scheme study +
FP32 mixed precision. See [`ROADMAP.md`](ROADMAP.md).

---

## Verified toolchain findings (nvc++ 26.5 / V100 — the durable payoff)

These are established; treat them as constraints, not open questions.

1. **C++ stdpar offloads the full idiom set** — `std::for_each` over
   `std::views::iota`, `std::reduce`/`transform_reduce`, `std::mdspan` (`m[i,j]`),
   managed `std::vector`, and fixed-size `std::array<Real, MAX_NZ>` column locals.
2. **`std::mdspan<layout_left>` is zero-cost** vs manual index math. **Gate the
   fallback on `__has_include(<mdspan>)`, NOT `__cpp_lib_mdspan`** — nvc++ 26.5
   leaves that feature-test macro undefined despite shipping a working header.
3. **Capturing `this` in a member kernel** passes on the host but hard-crashes on
   the GPU with `cudaErrorIllegalAddress`. Hoist member views to locals, capture
   `[=]`. (The `compute()` discipline in every physics module.)
4. **Keep state device-resident.** A per-step host copy forces a host↔device
   migration every step and is **~100×+ slower** on the V100 (a 1.0× no-op on the
   host build — the penalty is purely migration). One managed arena mapped once is
   the whole memory model; a stray per-step diagnostic copy silently reintroduces
   the penalty.
5. **stdpar has no async** — every `for_each` synchronises. Many tiny kernels can't
   be overlapped, so **fuse** them (a fused loop beat 1000 tiny ones by ~2 orders of
   magnitude in a microbench).
6. **Column kernels are occupancy-bound** (fixed-size per-thread locals + a serial
   vertical recurrence): ~10× lower throughput than flat maps even when resident.
   That's where the eventual perf effort goes (remap, vmix), and it is
   GPU-fundamental — don't try to "fix" it away (see
   [`GPU_STDPAR_NOTES.md`](GPU_STDPAR_NOTES.md)).
7. **Device helpers MUST be header-visible.** Under nvc++ `-stdpar`, a device-called
   helper whose body lives in a separate `.cpp` (with no device caller in that TU)
   is **not** given a device version → it fails to LINK (`nvlink: undefined
   reference`), and `-gpu=lto` does not rescue it. So write device helpers as
   `inline` header functions / templates; plain `inline` suffices, no force-inline
   needed.

## Build & environment notes

- **V100 → compute capability 7.0 → `TC_GPU_ARCH=cc70`** (many defaults are cc80).
- **Iterate on the host build** (g++, ~seconds). The `nvc++ -stdpar=gpu` build is
  slow — the cost is its optimizer + device-lowering pipeline (the parser is ~2% of
  it), so no header diet fixes it; reserve it for offload/throughput checks.
- **doctest via `FetchContent`** may need `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` at
  configure (its bundled CMake predates the local CMake's floor — a doctest quirk,
  not our code).
- **Verify GPU offload by speed**, not just "it ran": build `gpu` + `host` on a big
  problem and require `gpu/host ≫ 1`. A build can report a GPU backend yet run at
  host speed if it didn't actually offload.
