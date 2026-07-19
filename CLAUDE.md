# TurboChook — GPU-Native C++23 Ocean Dynamical Core

A from-scratch ocean dynamical core on **ISO C++ standard parallelism**
(`std::execution::par_unseq` + `nvc++ -stdpar=gpu`), in idiomatic modern C++23.
Learning-focused; correctness and clean architecture over feature count.

**North star:** an Arakawa **C-grid**, continuity-PPM, PV-conserving-Coriolis,
split-explicit hydrostatic ocean. **PoC on-ramp:** a 2D C-grid *barotropic*
shallow-water solver — the core's fast mode, so the PoC *is* the foundation. This
targets the C-grid/split-explicit **ocean** regime, **not** a coastal HLL/HLLC
Godunov one — do not build a Riemann-solver SWE.

> **Read [`docs/DESIGN.md`](docs/DESIGN.md) before writing any code** — the settled
> architecture (the prime directive, the three-layer split, the ADRs, the code
> skeletons). [`docs/FOUNDATIONS.md`](docs/FOUNDATIONS.md) is the `src/core/`+`src/lib/` layer
> (directory layout + reference implementations). [`docs/ROADMAP.md`](docs/ROADMAP.md)
> is the milestone ladder; do them in order. [`docs/STATUS.md`](docs/STATUS.md) is
> the current state + verified toolchain findings — **read it to pick up work.**
> [`docs/GPU_STDPAR_NOTES.md`](docs/GPU_STDPAR_NOTES.md) — what modern C++ + stdpar
> makes easy vs. what stays GPU-fundamental (do NOT try to "fix" the latter).
> [`docs/CPP_STYLE.md`](docs/CPP_STYLE.md) — the C++ house style (naming, struct-vs-class,
> concepts/policies, kernel rules); the detailed reference this file summarizes.

## Directory layout (where things go matters)

`src/` holds all the sources, over two physics-free base layers: `src/core/` is
the numeric substrate (`types.hpp` — `Real`, `Index`, `Field`, the mdspan seam),
and `src/lib/` is the physics-agnostic plumbing (`log.hpp`, `error.hpp`,
`profiler.hpp`, `arena.hpp`; later `assert`, MPI and CUDA/HIP wrappers). `core/`
depends on nothing but the stdlib; `lib/` depends on the stdlib and `core/`. Above
them the header-first library: `src/mesh`, `src/physics`, `src/numerics`,
`src/bc`, `src/io`. Thin `main`s are top-level `src/*.cpp` (no package manager, so
no separate `app/`); host-serial tests in `tests/`. Single flat `tc` namespace;
`src/` on the include path (`#include "lib/log.hpp"`, `#include "core/types.hpp"`).
Full tree + the dependency rule in [`docs/FOUNDATIONS.md`](docs/FOUNDATIONS.md).

## Logging & errors

- **Logging** = `tc::logger()` (C++23 `std::format`/`std::print`, level-gated,
  compile-time-checked format strings). **Host-only** — never log from a kernel;
  device code signals via a flag/NaN buffer the host reduces.
- **Errors** = **exceptions on the host**. `tc::Error : std::runtime_error` carries
  an `Errc` + `std::source_location` (its `what()` self-reports code + file:line).
  Host ops `throw`; catch once at `main` (log + exit); RAII unwinds cleanly.
  Fail-loud, fail-early (bad config/scheme dies at setup). **Kernels never throw**
  (`par_unseq` + escaping exception = `std::terminate`; device can't throw). Device
  failures (NaN/CFL) are detected by a post-step host reduction, then thrown
  host-side. Don't throw per-cell in the hot loop.
- **Profiling** = `tc::profiler()` + `TC_PROFILE("name")` (RAII, `steady_clock`,
  nested regions → self vs inclusive). Host-side; wraps kernel launches (stdpar
  syncs per call). NVTX/nsys for the fine GPU timeline.

## The prime directive (non-negotiable)

**stdpar-on-GPU forbids runtime polymorphism inside kernels.** A kernel is a
callable handed to a parallel algorithm; everything it touches must be trivially
copyable, device-accessible, and captured **by value**. Inside a parallel region:
NO virtual / `std::function` / RTTI / exceptions / captured `this` / owning
containers / allocation. Design is value-semantic, data-oriented,
compile-time-dispatched, with a hard host/device split.

## Kernel-authoring hard rules

1. Capture **by value**; only `mdspan` views + POD `Params` cross into a kernel.
   Never `this`, never a `std::vector`/owner. (Capturing `this` passes on host but
   hard-crashes on the GPU with `cudaErrorIllegalAddress` — hoist member views to
   locals first.)
2. No virtual / exceptions / RTTI / `std::function` / allocation inside a kernel.
3. Grid fields are `std::mdspan<Real, dextents<int,Rank>, std::layout_left>`
   (column-major = Fortran order; the default `layout_right` is wrong here). mdspan
   is non-owning; the owner is the Arena.
4. Per-cell value types (`Cons`/`Flux`) are `std::array<Real,N>` / `tc::Vec<N>`;
   read names via structured bindings, index `[v]` for generic code.
5. **Arena sized once, never grows** (else dangling mdspans).
6. `layout_left` + the parallel/fast-varying index on axis 0 = coalesced access.
7. **Double-buffer**: read old, write new (no read-neighbour-write-own of the same
   buffer).
8. **Verify GPU offload by speed** — a green host/serial run does NOT prove offload
   (build `gpu` + `host` on a big problem and require `gpu/host ≫ 1`; nsys when
   available). CPU-green ≠ GPU-correct.

## Build & test

C++23, CMake ≥ 3.25. One source, three configs:

```bash
# GPU (production)
cmake -B build-gpu   -DCMAKE_CXX_COMPILER=nvc++ -DTC_STDPAR=gpu -DTC_GPU_ARCH=cc70 && cmake --build build-gpu
# CPU threads
cmake -B build-mc    -DCMAKE_CXX_COMPILER=nvc++ -DTC_STDPAR=multicore             && cmake --build build-mc
# Host serial (tests/CI; par_unseq -> seq, no TBB) — the FAST dev loop
cmake -B build-host  -DCMAKE_CXX_COMPILER=g++    -DTC_STDPAR=off                   && cmake --build build-host
ctest --test-dir build-host --output-on-failure
```

- **Iterate on the host build** (g++, ~seconds). Reserve the `nvc++ -stdpar=gpu`
  build for offload checks + throughput — its optimizer/device pipeline is slow
  (the parser is ~2% of it; there is no header diet that fixes it).
- If the doctest `FetchContent` step trips a CMake policy floor, add
  `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` at configure (a doctest-CMake quirk, not our
  code).
- `std::mdspan` is C++23; where the stdlib lacks it (e.g. g++13), a hand-rolled
  `layout_left` `tc::MdView` fallback is selected behind a `__has_include(<mdspan>)`
  seam in `core/types.hpp`. Gate on `__has_include`, **not** `__cpp_lib_mdspan`
  (nvc++ 26.5 leaves that macro undefined despite a working header). **Never Kokkos.**

## Dependencies (policy)

- **stdlib-first.** The base layers `src/core/` (numeric types) and `src/lib/`
  (plumbing) depend on **nothing but the C++ stdlib** — `core/` on the stdlib
  alone, `lib/` on the stdlib plus `core/`. That is their whole purpose (`std::format`/`print`, `mdspan`, parallel algorithms, `chrono`,
  `source_location`).
- **doctest** is the *only* dependency, and it is **test-only** (fetched, never
  shipped in the library).
- **No Kokkos. Ever.** Not the framework, not `kokkos/mdspan`. If `std::mdspan` is
  missing, hand-roll the sliver.
- **No heavy C++ frameworks** (Kokkos, Eigen, Trilinos, …). If genuine heavy
  numerics are ever needed, prefer a thin C-ABI bridge over adopting a framework.

## Conventions

Full house style in [`docs/CPP_STYLE.md`](docs/CPP_STYLE.md); the essentials:

- **Naming:** types & concepts `PascalCase` (`Field2`, `Arena`, `Continuity`,
  `SSPRK2`); functions/variables/methods `snake_case` (`for_each_cell`,
  `phys_flux_x`); template params `PascalCase` **and role-meaningful** (`Scheme`,
  `Module`, `Integ` — not `R`/`M`/`T`; abstract dimensionals `Rank`/`Order`/`N` are
  fine); namespace `tc`; physical constants `UPPER_SNAKE` (`inline constexpr Real
  GRAVITY = 9.81`). Files `snake_case`.
- **No `using namespace`** (not `std`, not `tc`). Fully-qualify *named* entities:
  `std::` always, `tc::` from outside the namespace. **Operators are exempt** (ADL):
  write `a + b` / `dot(a,b)`, never `tc::operator+`.
- **Header-only** for templated device code (definitions must be visible at the
  call site — device helpers must be header-visible, or nvc++ stdpar won't link
  them; LTO does not rescue a separate-TU device helper).
- `#include` what you use; prefer `<algorithm>`/`<numeric>`/`<execution>`/`<mdspan>`
  /`<ranges>`/`<array>` facilities that offload over hand-rolled loops.
- Physics kernels are **pure free functions over views** → unit-testable on host
  with `std::execution::seq`. Add an analytical test with (or before) each kernel —
  they are the highest-leverage bug-catchers (DESIGN §10).
- No local/scratch files outside `build*/` and `tmp/` (git-ignored). No package
  installs beyond the toolchain.

## Commit conventions

Conventional-commit style (`feat:`, `fix:`, `docs:`, `test:`, `build:`). One
capability per commit. Branch + PR for anything non-trivial; do not force-push
shared branches without a note.

## Status

**Milestone 0 complete** — foundation (`src/core` + `src/lib`), the iteration seam
(`src/numerics/parallel.hpp`), the doctest harness, and the composing compile-time
policy stack all build and run (host-verified). The M2 operator bodies are the next
work. See [`docs/STATUS.md`](docs/STATUS.md) and [`docs/ROADMAP.md`](docs/ROADMAP.md).
