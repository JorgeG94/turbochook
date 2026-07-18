# TurboChook — GPU-Native C++23 Ocean Dynamical Core

A from-scratch ocean dyn-core on **ISO C++ standard parallelism**
(`std::execution::par_unseq` + `nvc++ -stdpar=gpu`). Architectural port of the Rakali
Fortran **ocean core** (`../rakali_dc`, `sim_type='ocean'`) into idiomatic modern C++.
Learning-focused; correctness and clean architecture over feature count.

**North star:** Arakawa **C-grid**, continuity-PPM, PV-conserving-Coriolis, split-explicit
hydrostatic ocean. **PoC on-ramp:** a 2D C-grid *barotropic* shallow-water solver (the core's
fast mode — the PoC *is* the foundation). This targets the C-grid/split-explicit **ocean**
regime, **not** the coastal HLL/HLLC Godunov regime — do not build a Riemann-solver SWE.

> **Read [`docs/DESIGN.md`](docs/DESIGN.md) before writing any code** — it is the settled
> architecture (the prime directive, the three-layer split, the ADRs, the code skeletons).
> [`docs/FOUNDATIONS.md`](docs/FOUNDATIONS.md) is the `lib/core/` layer (directory layout +
> logger + error handling, with reference implementations).
> [`docs/ROADMAP.md`](docs/ROADMAP.md) is the milestone ladder; do them in order.
> [`docs/PORTING_MAP.md`](docs/PORTING_MAP.md) maps each ocean algorithm to the exact Rakali
> `src/core/ocean/` file→module→procedure to port it from (port the algorithm, not the source).
> [`docs/LESSONS_FROM_RAKALI.md`](docs/LESSONS_FROM_RAKALI.md) — which Rakali design was
> Fortran-tax (C++ erases it) vs GPU-fundamental (carries over unchanged — do NOT "fix" it).
> `../stdpar_patterns_cpp` — standalone nvc++ stdpar reproducers (the MRE habit). The core
> patterns turbochook needs (for_each/iota, reduce, mdspan-`m[i,j]`, managed memory, cross-TU
> inline device helpers) are **verified to offload zero-cost on nvc++ 26.5 / V100** there.
> nsys is down on the DGX box → verify offload by SPEED (`gpu_check.sh`), not the timeline.

## Directory layout (where things go matters)

`lib/` is the library, header-first. `lib/core/` is the physics-free foundation (turbochook's
`pic` equivalent: `types.hpp`, `log.hpp`, `error.hpp`, `arena.hpp`, `timer.hpp`) — depends on
nothing but the stdlib; everything depends on it. Above it: `lib/mesh`, `lib/physics`,
`lib/numerics`, `lib/bc`, `lib/io`. Thin `main`s in `app/`, host-serial tests in `tests/`.
Single flat `tc` namespace; `lib/` on the include path (`#include "core/log.hpp"`). Full tree
+ the dependency rule in [`docs/FOUNDATIONS.md`](docs/FOUNDATIONS.md).

## Logging & errors

- **Logging** = `tc::logger()` (C++23 `std::format`/`std::print`, level-gated, compile-time-
  checked format strings). **Host-only** — never log from a kernel; device code signals via a
  flag/NaN buffer the host reduces.
- **Errors** = **exceptions on the host**. `tc::Error : std::runtime_error` carries an `Errc`
  + `std::source_location` (its `what()` self-reports code + file:line). Host ops `throw`;
  catch once at `main` (log + exit); RAII unwinds cleanly. Fail-loud, fail-early (bad
  config/scheme dies at setup). **Kernels never throw** (`par_unseq` + escaping exception =
  `std::terminate`; device can't throw). Device failures (NaN/CFL) are detected by a
  post-step host reduction, then thrown host-side. Don't throw per-cell in the hot loop.
- **Profiling** = `tc::profiler()` + `TC_PROFILE("name")` (RAII, `steady_clock`, nested
  regions → self vs inclusive), a port of Rakali's hierarchical profiler. Host-side; wraps
  kernel launches (stdpar syncs per call). NVTX/nsys for the fine GPU timeline.

## The prime directive (non-negotiable)

**stdpar-on-GPU forbids runtime polymorphism inside kernels.** A kernel is a callable
handed to a parallel algorithm; everything it touches must be trivially copyable,
device-accessible, and captured **by value**. Inside a parallel region: NO virtual /
`std::function` / RTTI / exceptions / captured `this` / owning containers / allocation.
Design is value-semantic, data-oriented, compile-time-dispatched, with a hard host/device
split.

## Kernel-authoring hard rules

1. Capture **by value**; only `mdspan` views + POD `Params` cross into a kernel. Never
   `this`, never a `std::vector`/owner.
2. No virtual / exceptions / RTTI / `std::function` / allocation inside a kernel.
3. Grid fields are `std::mdspan<Real, dextents<int,Rank>, std::layout_left>` (column-major
   = Fortran order; the default `layout_right` is wrong here). mdspan is non-owning.
4. Per-cell value types (`Cons`/`Flux`) are `std::array<Real,N>`; read names via structured
   bindings, index `[v]` for generic code.
5. **Arena sized once, never grows** (else dangling mdspans).
6. `layout_left` + the parallel/fast-varying index on axis 0 = coalesced access.
7. **Double-buffer**: read old, write new (no read-neighbour-write-own of the same buffer).
8. **Verify GPU offload with nsys** — a green host/serial run does NOT prove offload.

## Build & test

C++23, CMake ≥ 3.25. One source, three configs:

```bash
# GPU (production)
cmake -B build-gpu   -DCMAKE_CXX_COMPILER=nvc++ -DTC_STDPAR=gpu       && cmake --build build-gpu
# CPU threads
cmake -B build-mc    -DCMAKE_CXX_COMPILER=nvc++ -DTC_STDPAR=multicore && cmake --build build-mc
# Host serial (tests/CI; par_unseq -> seq)
cmake -B build-host  -DCMAKE_CXX_COMPILER=g++                         && cmake --build build-host
ctest --test-dir build-host --output-on-failure
```

(Exact CMake option names are established in M0 — the above is the intended shape.)

`std::mdspan` is C++23; if the stdlib lacks it, hand-roll a minimal `layout_left`
`tc::mdview` (~40 lines, the sliver we use) behind a `__has_include(<mdspan>)` seam.
**Never vendor Kokkos** — see the dependency policy below.

## Dependencies (policy)

- **stdlib-first.** `lib/core/` depends on **nothing but the C++ stdlib** — that is its whole
  purpose. Everything the stdlib now gives us (`std::format`/`print`, `mdspan`, parallel
  algorithms, `chrono`, `source_location`) replaces what Rakali needed `pic` for.
- **doctest** is the *only* dependency, and it is **test-only** (fetched, never shipped in the
  library).
- **No Kokkos. Ever.** Not the framework, not `kokkos/mdspan`. If `std::mdspan` is missing,
  hand-roll the sliver (above).
- **Escape valve for heavy machinery:** if turbochook ever genuinely needs numerics we don't
  want to reimplement, the sanctioned path is a **C-ABI bridge to Rakali's Fortran** (it
  already exposes an `iso_c_binding` FFI) — **not** adopting a C++ framework (Kokkos, Eigen,
  Trilinos, …).

## Conventions

- **Naming:** types & concepts `PascalCase` (`Field2`, `Arena`, `Continuity`, `SSPRK2`);
  functions/variables/methods `snake_case` (`for_each_cell`, `phys_flux_x`); template
  params `PascalCase`; namespace `tc`; physical constants `UPPER_SNAKE`
  (`inline constexpr Real GRAVITY = 9.81`). Files `snake_case.hpp` / `.cpp`.
- **No `using namespace`** (not `std`, not `tc`). Fully-qualify *named* entities: `std::`
  always, and `tc::` from outside the namespace (`tc::Vec<3>`, never a bare `Vec` or the
  stuttering `tc::tcVec`). **Operators are exempt** — they resolve by ADL, so write `a + b`
  for two `tc::Vec`, and `dot(a,b)` / structured-binding `get` likewise; never
  `tc::operator+`.
- **Header-only** for templated device code (definitions must be visible at the call site).
- `#include` what you use; prefer `<algorithm>`/`<numeric>`/`<execution>`/`<mdspan>`/`<span>`
  /`<array>`/`<ranges>` std facilities over hand-rolled loops where they offload.
- Physics kernels are **pure free functions over views** → unit-testable on host with
  `std::execution::seq`. Add an analytical test with (or before) each kernel — they are the
  highest-leverage bug-catchers (DESIGN §10).
- No local/scratch files outside `build*/` and `tmp/` (git-ignored). No package installs
  beyond the toolchain.

## Commit conventions

Conventional-commit style (`feat:`, `fix:`, `docs:`, `test:`, `build:`). One capability per
commit. End commit messages with:

```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

Branch + PR for anything non-trivial; do not force-push shared branches without a note.

## Status

**Spec only — no code yet.** Next: Milestone 0 (walking skeleton) per `docs/ROADMAP.md`.
