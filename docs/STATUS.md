# TurboChook — Status & Handoff

Current state of the project, plus the verified toolchain findings a fresh
contributor (human or agent) needs so they don't re-derive them. Read this after
[`DESIGN.md`](DESIGN.md) to pick up work.

---

## Where we are

**Milestone 0 (walking skeleton) is complete and host-verified.** The toolchain,
the iteration seam, the test harness, and the whole compile-time policy stack build
and run. No real physics yet — the M2 operator bodies are stubs.

### What's built

```
src/core/     types.hpp      Real/Index + Field<Rank> (mdspan ↔ MdView seam)
src/lib/      log.hpp        std::format/print logger, compile-time-checked
              error.hpp      exceptions + source_location, host/device split
              arena.hpp      size-once bump allocator; hands out Field views
              profiler.hpp   RAII nested-region timing (self vs inclusive)
src/numerics/ parallel.hpp   tc::par seam (par_unseq ↔ seq) + for_each_cell/face
              integrator.hpp Integrator concept + SSPRK2 / ForwardEuler (stubs)
src/mesh/     cartesian_mesh.hpp   Mesh concept + CartesianMesh model
src/physics/  reconstruction.hpp   Reconstruction concept + PCM/PLM/PPM/PQM/WENO
              baro_state.hpp       staggered C-grid BaroState + Params
              continuity.hpp       ContinuityModule + ContinuityFlux<R> (PPM)
              coriolis.hpp         CoriolisModule + SadournyEnstrophy
              pgf.hpp              PgfModule + FvPgf
              ocean_core.hpp       OceanCore<Cont,Cor,Pgf,Bc,Integ> composition
src/bc/       bc.hpp         BoundaryCondition + WallBC / PeriodicBC
src/          m0_walking_skeleton.cpp   (thin main; top-level *.cpp = executable)
tests/        test_m0.cpp    (5 doctest cases)
```

### Verified (host build, g++13)

- Builds clean; `m0_walking_skeleton` runs; the parallel reduction equals the serial
  reference to machine precision.
- `ctest` 5/5 green; the profiler tree shows correct region nesting.
- The `tc::MdView` fallback is the live path on g++13 (no `<mdspan>` there), so the
  `__has_include` seam works both ways.

### Not yet done

- **Every `compute()` is a `// TODO(M2)` stub.** The types, concepts, and
  composition are real and type-check; the numerics are empty.
- No GPU-build offload check yet (`nvc++ -stdpar=gpu`); no multicore build check.

---

## Next steps (M2 — fill the operators)

Smallest-first, each with an analytical test written alongside (DESIGN §10):

1. **`FvPgf::compute`** — `-g ∂η/∂x` onto u-faces, `-g ∂η/∂y` onto v-faces. No
   workspace, no reconstruction — the gentlest first real kernel. Test:
   **lake-at-rest** (flat η stays flat — well-balancedness).
2. **`ContinuityFlux<Ppm>::compute`** — the PPM swept thickness flux;
   `∂η/∂t = -∇·(H u)`. Test: **mass conservation** (total-mass drift ~ machine eps).
3. **`SadournyEnstrophy::compute`** — PV `q = (f+ζ)/h` at corners, PV-flux into u/v.
4. **The `Integrator` + BC halo fill**, then the ocean test cases: **geostrophic
   adjustment**, a **Kelvin/Rossby wave** at the right speed.
5. **GPU offload check** on a real M2 run (verify-by-speed).

See [`ROADMAP.md`](ROADMAP.md) for M3+ (layers → baroclinic instability → EOS/vcoord
→ vertical mixing).

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
