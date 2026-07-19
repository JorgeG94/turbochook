# TurboChook — C++ Style Guide

The house style for writing code in this repo: idiomatic **C++23**, value-semantic,
GPU-native. This is the *how-to-write-it* companion to [`DESIGN.md`](DESIGN.md) (the
architecture / prime directive), [`FOUNDATIONS.md`](FOUNDATIONS.md) (the layout), and
[`GPU_STDPAR_NOTES.md`](GPU_STDPAR_NOTES.md) (what stdpar makes easy vs.
GPU-fundamental). `CLAUDE.md` carries the terse summary; **this file is the detailed
reference.** When they disagree, fix one of them — they must not drift. New to the
machinery (lambda captures, `mdspan`, concepts)? Read [`CPP_PRIMER.md`](CPP_PRIMER.md)
first — it explains *why* these rules work.

The rules exist to serve one constraint above all: **code must offload under
`nvc++ -stdpar=gpu` unchanged.** Most of what follows is downstream of that.

---

## 1. Naming

| Kind | Convention | Examples |
|------|-----------|----------|
| Types, concepts | `PascalCase` | `Field2`, `Arena`, `Continuity`, `WallReconstruction`, `SSPRK2` |
| Functions, variables, methods | `snake_case` | `for_each_cell`, `phys_flux_x`, `mass_flux_x_` |
| Template parameters | `PascalCase`, **meaningful** | `Scheme`, `Module`, `Rank`, `Order` — **not** `R`, `M`, `T` |
| Physical / math constants | `UPPER_SNAKE` | `inline constexpr Real GRAVITY = 9.81` |
| Namespace | single flat `tc` | `tc::Arena`, `tc::logger()` |
| Files | `snake_case` | `ocean_core.hpp`, `baro_state.hpp` |

- **Template parameters carry their role, not a letter.** `template <WallReconstruction
  Scheme>` reads; `template <WallReconstruction R>` does not. A constraint already tells
  you the *kind*; the name should tell you the *role* (`Scheme`, `Module`, `Integ`). The
  only letters that survive are the genuinely abstract dimensionals — `Rank`, `Order`, `N`.
- **Trailing underscore for private data members** (`mass_flux_x_`, `arena_`). Nothing
  else takes a trailing underscore.
- **`Real`/`Index`, never `double`/`int`** for numeric/grid quantities — every signature
  reads intent, and precision is one `using` away (`core/types.hpp`). `Index` is *signed*
  on purpose so `i-1` at a boundary is well-defined.

## 2. Namespaces & qualification

- **No `using namespace`** — not `std`, not `tc`. Ever.
- **Fully-qualify named entities**: `std::` always; `tc::` from outside the namespace.
- **Operators are exempt** (ADL requires it): write `a + b`, `dot(a, b)`, never
  `tc::operator+`.
- One flat `tc` namespace. Directories organise files, **not** namespaces — call sites
  stay short (`tc::Continuity`, not `tc::physics::Continuity`).

## 3. `struct` vs `class` — state is the deciding line

`struct` and `class` differ only in default access, so use the choice to *signal intent*:

- **`struct`** — a stateless type with **no invariant**: a policy/tag, a trait, a POD
  value, or an aggregate. Everything is meant to be public. Examples: the reconstruction
  schemes (`struct Ppm`, `struct Weno5` — empty types carrying `static constexpr` traits
  + static functions), `struct Poly<Order>`, `Params`.
- **`class`** — a type with **private state maintained behind an interface**: it owns
  something and has an invariant to protect. Examples: `Arena` (owns the buffer, "sized
  once, never grows"), `ContinuityFlux` (owns arena-backed flux buffers), `OceanCore`
  (owns the prognostic state + module slots).

> Rule of thumb: **no members → `struct`; private members with an invariant → `class`.**
> If a stateless policy ever grows real per-instance state, that is the signal to promote
> it to a `class`.

## 4. Value semantics & the prime directive

**stdpar-on-GPU forbids runtime polymorphism inside kernels.** A kernel is a callable
handed to a parallel algorithm; everything it touches must be trivially copyable,
device-accessible, and captured **by value**. Inside a parallel region there is **no**
virtual, `std::function`, RTTI, exceptions, captured `this`, owning containers, or
allocation. Design is value-semantic, data-oriented, and **compile-time-dispatched**.

Choosing behaviour is choosing a *type*, not flipping a runtime flag: policies constrained
by concepts, composed at instantiation (§6). The runtime config-string → type mapping is a
host-side `std::variant`/`std::visit` bridge at the boundary (ADR-4), never a branch in the
hot loop.

## 5. Kernel-authoring hard rules

These are non-negotiable; a violation compiles on the host and *crashes or fails to link*
on the GPU (all verified on nvc++ 26.5 / V100 — see [`STATUS.md`](STATUS.md)).

1. **Capture by value.** Only `mdspan`/`Field` views + POD `Params` cross into a kernel.
   **Never `this`** (passes on host, `cudaErrorIllegalAddress` on GPU). Hoist member views
   into locals first, then capture `[=]`:
   ```cpp
   void compute(BaroState s, BaroState k, Params p) const {
       Field2 fx = mass_flux_x_;          // hoist member → local
       tc::for_each_cell(..., [=](Index i, Index j) { /* uses fx, never this */ });
   }
   ```
2. **No** virtual / exceptions / RTTI / `std::function` / allocation inside a kernel.
3. **Grid fields are `Field<Rank>`** = `mdspan<Real, dextents<Index,Rank>, layout_left>`
   (column-major / Fortran order — the default `layout_right` is wrong here). Non-owning;
   the `Arena` owns the storage.
4. **Per-cell value types are `std::array<Real, N>`** (`Cons`/`Flux`/`Vec<N>`) — read names
   via structured bindings, index `[v]` in generic code.
5. **`layout_left` + the parallel/fast-varying index on axis 0 = coalesced access.**
6. **Double-buffer**: read old, write new. Never read-neighbour-write-own of one buffer.
7. **Fuse kernels.** stdpar has no async — every `for_each` synchronises — so many tiny
   kernels can't overlap. Prefer one fused loop.
8. **Device helpers are `inline` and header-visible.** A device-called helper whose body
   lives in a separate `.cpp` fails to *link* under `-stdpar` (`-gpu=lto` does not rescue
   it). Templated/device code is header-only.

## 6. Concepts & policy-based design

The architectural idiom: **one generic algorithm, parameterised by concept-constrained
policies the compiler inlines.**

- **Constrain every policy template parameter with a concept** — `template
  <WallReconstruction Scheme>`, not `template <class Scheme>`. A mis-typed slot is a crisp
  error at instantiation, not a puzzle deep in a kernel.
- **`static_assert` your models** so a broken policy fails at its definition:
  `static_assert(WallReconstruction<Ppm>);`.
- **Keep concepts honest — don't over-unify.** When two families have genuinely different
  shapes, use **sibling concepts**, not one fat interface. The reconstruction axis is the
  worked example: `WallReconstruction` (returns a `Poly`) and `FaceReconstruction` (returns
  a biased value) are disjoint (a nested `kind` requirement enforces it), with a
  `Reconstructor = Wall || Face` **disjoint umbrella** used only by code blind to the kind.
  A consumer constrains on the kind it actually consumes.
- **Tag types over runtime enums for dispatch inside generic code** (`ReconKind::Wall`),
  branched with `if constexpr` — zero runtime cost.
- **Uniform module interface**: a physics module is a type with `{ init(arena, mesh);
  compute(s, k, p); }` (see `ContinuityModule`, `PgfModule`). Distinct concept *names*
  document the slot's intent even when the shapes coincide.

## 7. `const` / `constexpr` / `noexcept`

- **`constexpr` on pure per-cell helpers** — they must be header-visible and device-inlinable
  anyway, and `constexpr` implies `inline` and enables host compile-time tests (`Poly::eval`,
  `Poly::mean_over`, `Plm::minmod`).
- **`const`-correct methods**: `compute(...) const`, accessors return `const&` where they
  don't hand out mutable state.
- Don't sprinkle `noexcept` for ceremony; host ops that `throw tc::Error` are *not*
  `noexcept`. (Kernels can't throw regardless — §5.)

## 8. Headers & includes

- **Header-only for templated / device code** (definitions visible at the call site).
  Host-only pieces with global state or heavy includes *may* have a paired `.cpp`.
- **`#include` what you use** — every name a file names, it includes; don't lean on
  transitive includes.
- **Include path is `src/`** → `#include "core/types.hpp"`, `#include "lib/log.hpp"`,
  `#include "physics/coriolis.hpp"`. No `<turbochook/…>` prefix.
- **Prefer stdlib algorithms that offload** — `<algorithm>`, `<numeric>`, `<execution>`,
  `<ranges>`, `<mdspan>`, `<array>` — over hand-rolled loops.
- **Every header opens with a banner comment** stating what it is and *why it exists* (the
  design rationale, the gotcha it encodes) — not just what it contains. Match the density of
  the surrounding files.

## 9. Errors, logging, profiling (host-only)

- **Errors = exceptions, host path only.** `tc::Error : std::runtime_error` carries an
  `Errc` + `std::source_location`; its `what()` self-reports `code + file:line`. Host ops
  `throw`; **catch once at `main`** (log + exit); RAII unwinds. Fail-loud, fail-early — a
  bad config/scheme dies at setup, not mid-run.
- **Kernels never throw.** Device failures (NaN / CFL) are detected by a *post-step host
  reduction* over a flag/NaN buffer, then thrown host-side. Never per-cell in the hot loop.
- **Logging = `tc::logger()`** (`std::format`/`std::print`, compile-time-checked format
  strings, level-gated). **Host-only** — never log from a kernel.
- **Profiling = `TC_PROFILE("name")`** (RAII, nested self-vs-inclusive) around kernel
  launches; NVTX/nsys for the fine GPU timeline.

## 10. Testing

- **An analytical test alongside (or before) each kernel** — pure free functions over views
  run on the host with `std::execution::seq`; they are the highest-leverage bug-catchers
  (DESIGN §10). Lake-at-rest, mass conservation, geostrophic adjustment, wave speeds.
- **doctest** is the only dependency, and it is **test-only**.
- **Verify GPU offload by speed**, not by "it ran": a green host run does *not* prove
  offload. Build `gpu` + `host` on a big problem and require `gpu/host ≫ 1`.

## 11. Formatting

No `.clang-format` is enforced yet; **match the surrounding file.** Observed house style:

- **4-space indent**, no tabs. Braces on the same line (`if (...) {`, `struct Foo {`).
- **Column-aligned** declarations and trailing comments where it aids scanning (see the
  `static constexpr` blocks in `reconstruction.hpp`).
- Soft width ~**95 cols**; comments wrap.
- One capability per commit; conventional-commit style (`feat:`/`fix:`/`docs:`/`test:`/
  `build:`).
- **No local/scratch files** outside `build*/` and `tmp/` (git-ignored).

---

### The through-line

Everything here reduces to: **value-semantic, compile-time-dispatched, host/device-split
code that a parallel algorithm can copy onto a GPU without modification.** When a style
choice is unclear, ask which option keeps that true.
