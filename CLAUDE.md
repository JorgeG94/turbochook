# TurboChook — GPU-Native C++23 Coastal/Ocean Solver

A from-scratch finite-volume solver on **ISO C++ standard parallelism**
(`std::execution::par_unseq` + `nvc++ -stdpar=gpu`). Architectural port of the Rakali
Fortran solver (`../rakali_dc`) into idiomatic modern C++. Learning-focused; correctness
and clean architecture over feature count.

> **Read [`docs/DESIGN.md`](docs/DESIGN.md) before writing any code** — it is the settled
> architecture (the prime directive, the three-layer split, the ADRs, the code skeletons).
> [`docs/ROADMAP.md`](docs/ROADMAP.md) is the milestone ladder; do them in order.

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

`std::mdspan` is C++23; if a compiler lacks it, vendor `kokkos/mdspan` behind a `std::` shim.

## Conventions

- **Naming:** types & concepts `PascalCase` (`Field2`, `Arena`, `EquationSet`, `HLLC`);
  functions/variables/methods `snake_case` (`for_each_cell`, `phys_flux_x`); template
  params `PascalCase`; namespace `tc`; physical constants `UPPER_SNAKE`
  (`inline constexpr Real GRAVITY = 9.81`). Files `snake_case.hpp` / `.cpp`.
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
