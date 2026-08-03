# 03 — Arena: the memory stack manager

**Depends on:** 01 (Array signature), 02 (`malloc_device` signature only — not the
whole device layer). **Blocks:** 05.

Contract: [`../CONTRACT_MEMORY.md`](../CONTRACT_MEMORY.md) §1.5.

---

## Scope

One pool, allocated once at startup, **sealed** after init, never grown. Hands out
`Array`s that are non-owning views into itself. Two tiers: monotonic for persistent
fields, an RAII stack above the high-water line for per-step scratch.

```
src/core/arena.hpp        Arena, ScratchScope
src/core/mem_report.hpp   per-label accounting
tests/test_arena.cpp
```

## The API

```cpp
class Arena {
public:
    explicit Arena(MemoryQuantity size, Pool p = Pool::Device);   // ONE device_alloc

    // Rank is sizeof...(Ds) - 1: the trailing argument is the halo WIDTH, not a
    // dimension. `Init` is a TEMPLATE parameter because a function parameter pack
    // must come last -- `alloc(const char*, Ds..., Init = ...)` cannot deduce
    // (verified: "no known conversion from int to Init").
    template <class T, Space S = Space::Device, Loc L = Loc::Center,
              Init I = Init::Poison, class... Ds>
    Array<T, sizeof...(Ds) - 1, S, L> alloc(const char* label, Ds... dims_then_ng);

    void           seal();                 // any persistent alloc after this throws
    ScratchScope   scratch();              // RAII; restores the bump pointer
    MemoryQuantity bytes_used()     const;
    MemoryQuantity bytes_capacity() const;
    void           report()         const; // per-label breakdown
};

class Arenas {                             // one bump stack per Pool
public:
    Arena& device();  Arena& host();  Arena& shared();
    void   seal();                         // seals all three; ONE barrier
};
```

> **No `Parity` template parameter.** An earlier draft ended this signature
> `Array<…, S, L, P>`. `Parity` moved to halo-group registration
> (`group.add(u, v, Parity::Vector)`) because it changes no layout and no extent, and
> vector components must exchange as a **pair** regardless — a 90° rotation swaps them.
> `Array` has four template parameters. See [`../design/ARRAY.md`](../design/ARRAY.md) §4.1.

Cell counts plus a halo width; `Loc` and the arena do the rest:

```cpp
// Halo is in the EXTENTS; the interior box comes from Region. `Loc` supplies the
// extra face -- alloc DERIVES it rather than trusting the caller (see work order 2).
auto h = arena.alloc<Real>("h", nx, ny, nz, ng);                             // Loc::Center
auto u = arena.alloc<Real, Space::Device, Loc::XFace>("u", nx, ny, nz, ng);  // +1 in x
```

## Work order

1. **Bump allocator + alignment.** 128-byte alignment per field so the fast axis stays
   coalescing-friendly. Overflow is a **throw naming the label**, never a silent wrap.
2. **`alloc` DERIVES face extents from `Loc`**, rather than trusting the caller. A
   `Loc::XFace` array allocated without the extra face compiles and then makes the
   halo exchange silently wrong -- `Loc`'s whole justification for being in the type
   is that it changes extents, so it must be enforced, not merely declared.
3. **`seal()`.** Turns "no allocation in the time loop" from a convention into an
   invariant — and it is also what makes CUDA graph capture legal, since replay
   requires stable argument pointers.
3. **Labels + `report()`.** One string per allocation. Cheap now, annoying to
   retrofit, and it is the honest answer to "where did the memory go".
4. **`ScratchScope`.** Restores the bump pointer in its destructor. Exception-safe,
   and it finally makes the mark/restore tier usable — it exists in the current code
   with **zero call sites**, which is why the two-tier story is unproven.
5. **Initialization policy — device-side, async, poisoned.** `alloc` enqueues a
   **device kernel** to fill, never a host loop plus a copy: on a managed pool the first
   write decides page placement, so host-side init lands every page on the host and
   migrates all of them on first device access. Default is a signalling **NaN**, not zero
   — zero is a plausible ocean state (`η = 0`, `u = 0`) and would hide a forgotten
   initialisation, whereas NaN propagates into the `any_nonfinite` check already running
   at cadence. The fill is **async**: per-field synchronisation would give N startup
   stalls and would stop host-side setup (config, output files, metrics) overlapping the
   device work. `seal()` carries the one `device::sync()`.
6. **A sizing pass.** Compute the required bytes from the config *before* allocating,
   so an undersized pool dies at second zero with a number, not at hour six.
7. **Mixed precision falls out free** — the arena hands out bytes, so `alloc<float>`
   and `alloc<double>` coexist in one pool. No work required; just don't preclude it.

## Decisions to record

- **The pool allocator is a per-backend policy**, chosen by 02:

  | build | pool |
  |---|---|
  | discrete GPU | `cudaMalloc` (strict) or `cudaMallocManaged` + advise + prefetch |
  | coherent (GH200) | either — measured identical |
  | SYCL | `sycl::malloc_device` / `malloc_shared` |
  | host / multicore | `aligned_alloc` |

  Measured on PVC: `malloc_shared`+prefetch and `malloc_device` are the **same speed**,
  so the choice has no architectural consequence above the arena.
- **Prefer the strict device-only pool** where available: it gives the `Space` gate
  hardware backing as well as type-system backing.
- **No per-field allocation, ever.** One migration surface, one snapshot, one restart
  copy, truthful accounting.

## Tests

| test | asserts |
|---|---|
| alignment | every field's base address is 128-byte aligned |
| overflow | over-allocating throws and the message names the label |
| seal | `alloc` after `seal()` throws |
| scratch RAII | bump pointer restored on scope exit, including on an exception path |
| report | sum of per-label bytes == `bytes_used()` |
| mixed precision | `alloc<float>` and `alloc<double>` coexist and don't overlap |
| poison default | a freshly `alloc`ed field is all-NaN; `any_nonfinite` sees it |
| zero is explicit | `Init::Zero` gives exact zeros; the default does NOT |
| async fills | N allocs enqueue N fills with **one** sync, inside `seal()` |
| device-side init | on a managed pool, pages are resident on the DEVICE after seal (verify by the spike-01 A/B ratio, not by inspection) |
| no-realloc | every `Array` handed out stays valid after later allocations |
| **graph precondition** | pointers handed out are stable across the whole run |

## Acceptance gate

- `report()` accounts for **every byte**, by label.
- A deliberately undersized pool fails at startup with the required size in the message.
- Scratch scope survives an exception without leaking the bump pointer.
- Works over all four pool backends (host is enough for CI; GPU checked with the spikes).

## Deferred

Defragmentation (there is none — monotonic by design), multiple pools, a host-pinned
staging pool (add when async D2H staging arrives), NUMA placement on the host build.
