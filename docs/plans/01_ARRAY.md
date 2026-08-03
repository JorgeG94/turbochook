# 01 — Array / View: the keystone

**Blocks:** 02, 03, 04, 05 — this is the freeze they are all waiting on.
**Depends on:** steps 1–4 depend on **nothing** and are the freeze. Steps 5–6 need
*declarations only* from 02 (`Region`, `TC_KERNEL`, `device::sync`) and 03
(`ScratchScope`, forward-declared). 02 itself depends on nothing and can start in
parallel, so there is no cycle — **but if 02 has not landed, do steps 1–4 and stop.**

Design: [`../design/ARRAY.md`](../design/ARRAY.md) — authoritative.
Contract: [`../CONTRACT_MEMORY.md`](../CONTRACT_MEMORY.md) §0–§1.

> Rewritten 2026-08-03. An adversarial review found the previous plan's central decision
> ("own the type; do not alias `std::mdspan`") could not work as specified: a custom
> layout with Fortran lower bounds is **not a conforming mdspan layout** — the standard
> fixes the index domain as `[0, extent)` and hardened libc++ traps on the ghost-cell
> access. `View` is now a conforming `std::mdspan` with `layout_left`, zero-based.

---

## Scope

| in | out |
|---|---|
| `Space`, `Loc`, `Scalar`, `DType`, `dtype_of` | anything that allocates (→ 03) |
| `View<T,Rank>` = `std::mdspan<..., layout_left>` | anything that launches (→ 02) |
| `Array<T,Rank,Space,Loc>` — owning handle | `Parity` (→ 04, at group registration) |
| `mirror()` / `copy()` / `HostArray` | reductions, `Region`, `TC_KERNEL` (→ 02) |
| `window` / `layer` / `tracer` / `ColumnView` | `fill` / `fill_by` bodies (→ 03) |
| `VectorField` / `Stagger` / `CVector2,3` | `restagger()` (needs `Mesh`; → later) |
| `enum class Init` — **declared here**, because 03's `alloc` names it | `FieldDesc` (→ 05) |
| **migration of existing consumers — explicitly NOT in this workstream** | |

## Files

```
src/core/types.hpp        Real, Index, GlobalIndex, is_scalar_v, Scalar, DType, dtype_of
src/core/space.hpp        Space, Loc, Parity, host_subscriptable, mirror_is_identity
src/core/view.hpp         View alias + the <mdspan> fallback shim
src/core/array.hpp        Array, HostArray, Init, mirror(), copy(), debug_snapshot/peek
src/core/slice.hpp        window, layer, tracer, ColumnView, column()
src/core/vector_field.hpp Stagger, VectorField<A|B|C>, CVector2, CVector3
tests/test_core_array.cpp
```

**Three collisions with the existing tree — resolve them before writing a line:**

- **`src/core/types.hpp` already exists** (125 lines, 36 dependents, 149 use sites) and
  defines `Real`, `Index = int`, `Field<Rank>` and the `MdView` shim. Append to it; add
  `using Field = View<Real,Rank>` as a compatibility alias — verified to compile clean
  against the real physics headers, because `Field` *is already*
  `mdspan<Real, dextents<int,Rank>, layout_left>`. Note this changes `Index` from `int` to
  `std::int32_t` under all 149 sites: a no-op on LP64/arm64, but say it out loud. Delete
  `MdView` rather than leaving a second, divergent shim.
- **`src/mesh/mesh.hpp:28` already defines `tc::Loc`** (266 use sites) plus `Parity`,
  `x_staggered`, `y_staggered`. Defining `tc::Loc` again in `core/space.hpp` is a hard
  redefinition the moment one TU sees both — which is immediately. **Move** all four into
  `core/space.hpp` and have `mesh.hpp` include it.
- **`TC_KERNEL` is claimed by both this plan and 02.** It does not exist under `src/` today
  (only `spikes/common.hpp:36`). **02 owns it** — it expands to `__host__ __device__`,
  which is backend-aware by definition, and `device/` is the only backend-aware code.
  `slice.hpp` includes `device/backend.hpp` for it.

## Work order

1. **Vocabulary.** Fixed-width `Index`/`GlobalIndex`; `is_scalar_type` as a
   **specialisable trait** (not `is_floating_point_v` directly — `__half`/`bf16` are not
   floating-point on these toolchains); `DType` with a fixed underlying type; `dtype_of`
   **defined**, not just declared.
2. **The two `Space` predicates**, which answer different questions:
   `host_subscriptable` is strict always (portability discipline — code that compiles on
   GH200 must not fail on a V100); `mirror_is_identity` tracks the hardware, so `mirror`
   is free on a coherent build.
3. **`View` alias + the shim.** `std::mdspan<T, dextents<Index,Rank>, layout_left>` behind
   `__has_include(<mdspan>)`. The shim is minimal — `operator[]` ranks 2–4, `extent()`,
   `data_handle()` — and has the **same semantics** as the real thing, so no code path
   diverges.
4. **`Array` — the whole §4 surface, not a subset.** Default constructor (it must be a
   member of state aggregates), a public constructor (the Arena builds them),
   `operator[]` **not** `operator()` — mdspan has no call operator — plus `value_type`,
   `rank_v` (**not** `rank`: `mdspan::rank()` is a *function*, and generic code over both
   must not trip on it), `space`, `loc`, `kernel_view()`/`ckernel_view()`, `extent(int)`,
   `bytes()`, `dtype()`. `bytes()` and `ckernel_view()` need **bodies** — the draft
   declared them and they linked only while nobody called them. `static_assert` message
   ASCII-only: nvc++'s EDG frontend renders non-ASCII as `???`.
5. **`mirror` / `copy` / `HostArray`.** `mirror` **always returns `Space::Host` in the
   type**; the identity lives in the body. Returning the source type on a coherent build
   made the canonical read-back fail to compile under `TC_COHERENT_MEMORY` — on exactly
   the machine the optimisation is for. `copy` **synchronises** — `do_concurrent` is
   contractually async, so a read-back that didn't sync would be racy on every discrete
   device. `HostArray` is a distinct **owning** type (it must outlive `Arena::seal()`, and
   `Array` owns nothing) and needs a **body**, not a forward declaration: an incomplete
   return type is an error at every call site.
6. **Slicing.** `window` (**trailing dimensions only**) / `layer` / `tracer` by hand
   against `layout_left`. A sub-box restricted in dim 0 is not expressible as
   `layout_left` — measured 40/48 elements wrong — which is why `subbox` was renamed and
   narrowed. `std::submdspan` exists on **no toolchain in the matrix today** and would
   return `layout_stride` for the strided case anyway, so the eventual one-liner migration
   covers `layer` and `tracer` only.

## Decisions recorded

- **`View` is a conforming `std::mdspan`, zero-based.** The ghost-cell mechanism is
  unaffected: interior `ng .. ng+nx-1`, and `v[i-1,j]` at `i == ng` reaches halo cell
  `ng-1` with no branch, no clamp, no wrap. Only the *label* on the interior changes, and
  `Region` carries it.
- **`Loc` in the type, `Parity` not.** `Loc` changes extents (a face field has one more
  face than cells) — and must therefore be **enforced by `Arena::alloc`**, not merely
  declared. `Parity` changes no layout and no extent, and vector components exchange as a
  **pair** regardless (a 90° rotation swaps them), so it belongs at group registration.
- **Ranks 2–4 from the start.** Five consumers, three of them not sea ice: tracer bundle,
  tendency history, time-bracketed forcing.
- **Bounds checking comes from the standard library.** `_LIBCPP_HARDENING_MODE` /
  `_GLIBCXX_ASSERTIONS` check every subscript, implemented and tested upstream. Debug/CI
  always on; production tier is open decision 1 in the design doc.

## Tests

| test | asserts |
|---|---|
| compiles | [`design/ARRAY.md`](../design/ARRAY.md) §4's `Array`, `Host`/`Device`, ranks 2–4, `float`/`double` |
| aggregate member | `struct S { Array<double,3,Space::Device> a; }; S s{};` compiles |
| trivially copyable | `static_assert` on `View` and every instantiation above |
| halo reach | interior loop `ng..ng+nx-1`, `v[i-1,j]` at `i==ng` hits `ng-1`, in bounds |
| **hardened run** | full suite under `-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG`, no trap |
| **hardening positive control** | a deliberate `v[extent(0)+4, 0]` **aborts** in that same build |
| read-back compiles | §5's canonical sample, **both** with and without `TC_COHERENT_MEMORY` |
| mirror identity | host-accessible → same data handle, zero allocation |
| mirror identity, coherent | under `TC_COHERENT_MEMORY`, a `Space::Device` **source** is also free |
| mirror staging | `Device` → mirror → `copy` round-trips, and `copy` synchronises |
| **negative: host gate** | `TC_NEGATIVE` build FAILS with the named `static_assert` |
| slicing | `tracer(a,t)` is contiguous; `layer`/`window` extents and offsets correct |
| callable surface | `bytes()`, `ckernel_view()`, `HostArray`, `ColumnView`, `VectorField<A\|B\|C>` all **link** |
| precision generic | the same kernel body instantiates for `float` and `double` |

Spike 04 proves the gate fires under nvc++ **and** clang — port it, don't rewrite it.

**The hardened run is two-sided, and that is not pedantry.** `_LIBCPP_HARDENING_MODE_DEBUG`
is a *value*; the switch is `_LIBCPP_HARDENING_MODE`. Measured: `-D_LIBCPP_HARDENING_MODE_DEBUG`
alone gives `exit=0` on a deliberate out-of-bounds read — the flag does nothing, the suite
passes, and the gate certifies nothing. A suite with no OOB access also passes whether
hardening is live or not. So the control is what distinguishes "clean" from "misspelled",
and the flag must reach **every** TU including FetchContent'd doctest, or it is an ODR
violation.

## Build configurations this plan needs and CMake does not have yet

`_LIBCPP_HARDENING_MODE` / `_GLIBCXX_ASSERTIONS`, `TC_NEGATIVE`, `TC_COHERENT_MEMORY`,
`TC_SINGLE_PRECISION` — none exist in `CMakeLists.txt` today (grep: zero hits). The negative
test additionally cannot live in the `file(GLOB tests/test_*.cpp)` single-binary target: it
needs an `EXCLUDE_FROM_ALL` probe target plus `add_test(... WILL_FAIL TRUE)`, ~15 lines.
Budget for it; the spikes did this with a Makefile target that inverts the exit status.

## Acceptance gate

- All tests green on host, including the **hardened** run and its positive control.
- Negative test fails to compile under nvc++ and any host compiler with `<mdspan>` (not
  "g++" — libstdc++ has none until GCC 16). **nvc++ is not on the mac dev box**: that half
  of the gate runs on the GPU box before merge, and it blocks merge, not local progress.
- Signatures published to 02–05 and not changed afterwards without a note.

## Deferred

`layout_left_padded` (C++26), `layout_stride`, rank > 4, non-owning wrappers over foreign
memory, and any arithmetic on arrays — kernels do arithmetic, arrays hold numbers.
