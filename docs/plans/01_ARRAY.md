# 01 — Array / View: the keystone

**Depends on:** nothing. **Blocks:** 03 (arena), 04 (comm), 05 (api).
**Freeze the signatures first**, then the rest run parallel.

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
| `Space`, `Loc`, `Scalar`, `DType` | anything that allocates (→ 03) |
| `View<T,Rank>` = `std::mdspan<..., layout_left>` | anything that launches (→ 02) |
| `Array<T,Rank,Space,Loc>` — owning handle | `Parity` (→ 04, at group registration) |
| `mirror()` / `copy()` / `HostArray` | reductions (→ 02) |
| `subbox` / `layer` / `tracer` / `ColumnView` | `Region` (→ 02) |

## Files

```
src/core/types.hpp        Real, Index, GlobalIndex, is_scalar_v, Scalar, DType, TC_KERNEL
src/core/space.hpp        Space, Loc, host_subscriptable, mirror_is_identity
src/core/view.hpp         View alias + the <mdspan> fallback shim
src/core/array.hpp        Array, HostArray, mirror(), copy(), debug_snapshot/peek
src/core/slice.hpp        subbox, layer, tracer, ColumnView
tests/test_core_array.cpp
```

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
4. **`Array`.** Default constructor (it must be a member of state aggregates), a public
   constructor (the Arena builds them), `operator[]` **not** `operator()` — mdspan has no
   call operator. `static_assert` message ASCII-only: nvc++'s EDG frontend renders
   non-ASCII as `???`.
5. **`mirror` / `copy` / `HostArray`.** `copy` **synchronises** — `do_concurrent` is
   contractually async, so a read-back that didn't sync would be racy on every discrete
   device. `HostArray` is a distinct **owning** type, because it must outlive
   `Arena::seal()` and `Array` owns nothing.
6. **Slicing.** `subbox` / `layer` / `tracer` by hand against `layout_left` now;
   `std::submdspan` (C++26) replaces them one-for-one later — genuinely, because
   `layout_left` is one of the layouts the standard specifies it for.

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
| compiles | §4 instantiated for `Host`/`Device`, ranks 2–4, `float`/`double` |
| aggregate member | `struct S { Array<double,3,Space::Device> a; }; S s{};` compiles |
| trivially copyable | `static_assert` on `View` and every `Array` alias |
| halo reach | interior loop `ng..ng+nx-1`, `v[i-1,j]` at `i==ng` hits `ng-1`, in bounds |
| **hardened run** | full suite under `_LIBCPP_HARDENING_MODE_DEBUG` **without trapping** |
| mirror identity | host-accessible → same data handle, zero allocation |
| mirror identity, coherent | under `TC_COHERENT_MEMORY`, that includes `Space::Device` |
| mirror staging | `Device` → mirror → `copy` round-trips, and `copy` synchronises |
| **negative: host gate** | `TC_NEGATIVE` build FAILS with the named `static_assert` |
| slicing | `tracer(a,t)` is contiguous; `layer`/`subbox` extents and offsets correct |
| precision generic | the same kernel body instantiates for `float` and `double` |

Spike 04 proves the gate fires under nvc++ **and** clang — port it, don't rewrite it.

## Acceptance gate

- All tests green on host, including the **hardened** run.
- Negative test fails to compile under nvc++ and any host compiler with `<mdspan>` (not
  "g++" — libstdc++ has none until GCC 16).
- Signatures published to 02–05 and not changed afterwards without a note.

## Deferred

`layout_left_padded` (C++26), `layout_stride`, rank > 4, non-owning wrappers over foreign
memory, and any arithmetic on arrays — kernels do arithmetic, arrays hold numbers.
