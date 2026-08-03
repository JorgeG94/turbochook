# TurboChook — the redesign

Written 2026-08-02, after (a) hardware verification of the memory model on three
machines and (b) a structured read of Oceananigans.jl, the closest existing model
to the script-driven philosophy we have adopted.

`main` is being retired. This document is what replaces the parts of
[`DESIGN.md`](DESIGN.md) that the evidence changed. Where it contradicts DESIGN.md,
this wins; where DESIGN.md is silent, it still applies.

> **The organising principle of this rewrite.** Rakali's module map is shaped by
> Fortran constraints C++ does not have; Oceananigans' is shaped by Julia
> capabilities C++ does not have. **Inherit rakali's physics lessons and
> Oceananigans' interface lessons; re-derive both decompositions.**

---

## 1. Verified, not assumed

Everything in this section was measured, not reasoned about. Spikes live in
[`../spikes/`](../spikes/).

| | V100 (discrete) | GH200 (coherent) | PVC Max 1550 (discrete) |
|---|---|---|---|
| coherence query | 0 | 1 | `usm_system_alloc` = 0 |
| host touches 1 element/iter | **4.1x** slower | **1.1x** slower | **15.4x** slower |
| stdpar vs native kernel | 1.09x | 1.19x | stdpar **does not build** — see §1.1 |
| device-only pool | PASS | PASS | PASS (`malloc_device`) |
| shared pool + prefetch | PASS | PASS | PASS, *identical* speed to device-only |
| host compiler independent of vendor | — | PASS (g++ + nvcc, aarch64) | PASS (icpx only compiles the device TU) |

The coherence query is the portable abstraction it needed to be: `pageableMemoryAccess`
on CUDA and `aspect::usm_system_allocations` on SYCL answer the same question, and the
A/B ratio tracks it on all three machines. **Discrete hardware punishes a stray host
touch by 4–15x**; that is the entire justification for the `Space` gate.

On PVC, `malloc_shared`+prefetch and `malloc_device` measured identically, so the pool
strategy is genuinely a per-backend policy with no architectural consequence above it.

### 1.1 `std::views::iota` is NOT a portable stdpar idiom

`parallel.hpp` records the flat-`iota` + `std::for_each(par_unseq, …)` shape as "the
reliably-offloading idiom (verified on nvc++)". **It does not compile under oneDPL.**
`icpx -fsycl-pstl-offload=gpu` routes `std::for_each` into
`oneapi::dpl::__pattern_walk1`, which for a non-contiguous iterator tries to wrap the
range in a `sycl::buffer` and `set_final_data()` — i.e. copy results *back into* the
iterator. An `iota_view` iterator is a read-only proxy, so it fails deep in libstdc++
with `error: expression is not assignable`.

This is not a bug we can work around at the call site; it is a real difference in what
the two PSTL implementations accept. **It is also the strongest possible vindication of
the single-launcher chokepoint**: as one function it is a one-line backend difference;
spread across 165 call sites it would have been a migration. On Intel the chokepoint
simply uses the native SYCL launcher, which passed cleanly.

**Do not treat "stdpar" as one portable target.** It is three implementations
(nvc++/NVIDIA, oneDPL/Intel, hipstdpar/AMD) that accept different iterator vocabularies.

Also verified on Aurora: gcc 13.4's libstdc++ has no `<mdspan>`, so the `tc::MdView`
fallback is **live on production hardware**, not a laptop convenience. It must be
maintained, not treated as a sunset seam.

### 1.2 PVC implicit scaling is a 6.3x TRAP — one rank per tile, always

Spike 05 reported ~392 GB/s on a ~3.2 TB/s part and I assumed we were using one tile.
**Wrong, and in the wrong direction.** Measured (2048², 200 iters, same kernel):

| launch | GB/s |
|---|---|
| whole card, COMPOSITE (implicit scaling over 2 tiles) | **392** |
| ONE tile, explicitly partitioned | **2479** |
| one tile via `ZE_FLAT_DEVICE_HIERARCHY=FLAT` | 2479 (confirms) |
| one tile + explicit workgroup 256 | **2588** |

**A single tile is 6.3x faster than "the whole card."** Level Zero's implicit scaling
splits the kernel across both tiles while the pool's pages are distributed across both
HBM stacks, so a stencil's neighbour reads cross the inter-tile link and it becomes the
bottleneck. This is the classic implicit-scaling failure mode for stencil codes, and
the default device abstraction walks straight into it.

**Rule: one rank per tile. Never submit a stencil kernel to a COMPOSITE root device.**
This is the Aurora analogue of rakali's `CUDA_VISIBLE_DEVICES`-before-`MPI_Init` trap,
and it is worse — that one produced a misleading diagnostic, this one silently costs 6x.
`ZE_FLAT_DEVICE_HIERARCHY=FLAT` makes a device a tile (12 per node, 448 EUs, 64 GiB).

With that corrected, **Intel is performing fine** and the earlier worry evaporates:

| | effective GB/s, same kernel |
|---|---|
| V100 | 1557 |
| PVC, one tile | 2479 (2588 with explicit workgroup) |
| GH200 | 5600 |

### 1.3 Launcher micro-findings (measured, all three machines' idiom validated)

- **`tc::MdView` costs exactly nothing.** MdView 209.58 vs raw pointer 209.59 GB/s
  (COMPOSITE), 1376.20 vs 1385.65 (FLAT). The mdspan seam is free even in the
  hand-rolled fallback — which matters, because Aurora's gcc 13.4 has no `<mdspan>`
  and the fallback is live on production hardware.
- **Flattened `range<1>` beats `range<2>` by ~1.8x** (2479 vs 1376 per tile), despite
  `range<2>` avoiding an integer `%` and `÷` per work-item. The 1-D range gives 256
  consecutive fast-axis indices per group; a 2-D range gets a tiled workgroup shape
  that breaks full-row coalescing. **`parallel.hpp`'s flatten-and-unflatten idiom is
  validated on Intel as well as NVIDIA** — keep it.
- **An explicit workgroup is worth ~4%** over the runtime's default (2588 vs 2479).
  Expose it on the launcher; don't hardcode.

### 1.4 stdpar on Intel: the flag is a 2x tax, and the failure was narrower than it looked

Refining §1.1. `std::for_each(par_unseq, …)` **does** work on Intel given a proper
random-access iterator — `oneapi::dpl::counting_iterator` compiles and runs. So 05b's
failure was specifically `iota_view`'s **read-only proxy iterator**, not the `std::`
path in general.

But the flag that enables it is expensive. Measured on the same binary and grid:

| | without `-fsycl-pstl-offload` | with it |
|---|---|---|
| `dpl::for_each` + counting_iterator | **418.50** | 196.15 |
| native `q.parallel_for` | 391.57 | 229.56 |
| `std::for_each` + counting_iterator | (compiled out) | 194.82 |
| `std::for_each` + USM `int*` | (compiled out) | 193.77 |

**`-fsycl-pstl-offload=gpu` roughly halves everything in the binary, including kernels
that never touch a PSTL algorithm.** It works by replacing the system allocator with
USM-shared program-wide — the same mechanism as nvc++'s managed default — and that is
not something we want, since we own our pool explicitly.

**Decision: on Intel the chokepoint uses `oneapi::dpl::for_each` with a device policy,
or native `q.parallel_for`. Never the pstl-offload flag.** Note `dpl::for_each` was
*faster than* native here (418 vs 392), so the portable-shaped spelling costs nothing.

This kills the idea of one literal source spelling across all backends — and it does
not matter at all, because the chokepoint is one function. That is the entire argument
for building it first.

### 1.5 What the ocean kernels actually cost (`more_benchmarks_dc_ocean`)

A full ocean core, every hot kernel isolated and measured three ways — `do concurrent`,
a faithful CUDA port, and an optimized CUDA rewrite — on a V100 at **production size**
(473×297×30). This is the real answer to questions our Jacobi spikes could only gesture
at.

**The headline: a fully hand-optimized CUDA rewrite of the whole core is ~8% faster
than optimized `do concurrent`, measured end-to-end in one hot-GPU RK2 loop**
(101.5 vs 94.2 ms/stage). Our synthetic spikes said 9–19%; a real model says 8%.
Consistent, and it settles the escape-hatch question: **stdpar-first is correct, and
the ceiling on a full rewrite is single digits.**

#### The profile — and it should reorder our thinking

| kernel | share of a stage | kind |
|---|---:|---|
| **ocean_redi** | **~40%** | column solver |
| **ocean_vmix_kshear** | **~18%** | column solver |
| ocean_continuity | ~7% | flat stencil |
| ocean_ale_remap | ~6% | column solver |
| ocean_hvisc | ~5% | flat stencil |
| ocean_barotropic (btstep) | ~5% | flat, launch-bound |
| ocean_vmix_epbl | ~2% | column solver |
| MEKE | ~1% | flat, launch-bound |

**Redi + kappa-shear are 58% of a stage, and both are column solvers.** Everything
TurboChook has actually built — continuity, Coriolis, PGF, the barotropic subcycle —
totals roughly 12%.

That is not an argument for building Redi next (it needs EOS, isopycnal slopes and
tracers first). It *is* an argument that **the column-kernel path is the main event and
must be designed for now, not deferred to M5**. `CONTRACT_MEMORY.md` §3.1 is therefore
load-bearing architecture, not a convenience — see the occupancy cliff below.

#### Where each language actually wins

| regime | winner | examples |
|---|---|---|
| memory-bound stencils | **DC/stdpar wins or ties** | redi +7% DC, hvisc +4% DC |
| launch-count-bound light kernels | CUDA | MEKE +57%, ale_remap +20%, btstep +7% |
| register/occupancy-bound | CUDA | kappa_shear +16% |

**The escape hatch pays where you are launch-bound or register-bound — not where you
are bandwidth-bound.** That is a far sharper rule than "measure first", and it means
the native path's targets are predictable in advance.

Amdahl then caps the whole thing: the two giants are near parity, so the ~20% of light
kernels being CUDA +27% blends to +8% overall. **Speeding up a cheap kernel cannot move
a total the giants own.**

#### The algorithm is worth ~3x more than the language

Portable optimizations landed **1.18×–1.34×** per kernel over each one's own DC
baseline — all bit-identical, all also faster on CPU. Three case studies, one pattern:

- **redi**: the centre tracer column is rebuilt 4× per cell (once per face). Hoist it
  → 8 column rebuilds become 5. **1.24–1.35× DC.** Bonus: registers 93→64, local
  memory 16.4→14.4 KB, occupancy 34%→50%.
- **ale_remap**: T and S ride the *same* `h_old`/`h_new` column, so the PPM geometry and
  the overlap sweep are identical between them — the faithful path pays twice. Build
  once, use for both. **1.33× DC**, and 10 launches → 5.
- **continuity**: 11 PPM kernels → 3 by fusing reconstruction + boundary + transport so
  the `hfl/hfr` scratch arrays never reach global memory (~280 MB/call eliminated).
  **1.36× DC / 1.50× CUDA.** And note: **fused DC (0.82 ms) beats the original faithful
  CUDA (0.93 ms)** — fusing in the portable language beats not fusing in the fast one.

The single pattern behind all three: **remove redundant recomputation of shared column
geometry, and stop intermediate arrays from reaching DRAM.**

#### "Everything cleverer LOST"

On continuity, the rejected list is as instructive as the win: full recompute-fusion,
k-blocking for ILP, `__launch_bounds__`, div+flux fusion, and **shared-memory tiling**
all lost. The flat one-thread-per-face layout lets the GPU's own L2 and occupancy do
the reuse better than any hand-rolled scheme; the divergence kernel was already at ~84%
of DRAM peak. **Only removing genuine waste paid.**

Do not build a fusion framework or a tiling abstraction. Fuse to delete intermediate
global arrays and redundant recomputation, then stop.

#### The measurement methodology, worth adopting wholesale

- **Correctness is bit-identity** (`max rel < 1e-12`, FMA-contraction level) — **and
  verify the verifier**: perturb one term by 1 ulp and confirm the check trips.
- **Normalize error by the field's GLOBAL max, not per element.** A per-element
  relative diff reported a spurious 9e-9 on budgets (differences of two large numbers)
  and 1.6e-10 on `v` (which crosses zero) while absolute differences were all ~1e-16.
- **One binary, one device allocation, for any A/B.** Comparing a separately-built
  stdpar binary against a separately-built CUDA one measures allocators and warmup, not
  languages. They needed an `!$acc host_data use_device` bridge for this; **spike 03
  already proved one pool feeds both our launchers**, so we get it for free.
- **Measure at production size.** The gap is a strong function of cell count — 4096²
  makes everything look clean while nothing runs there.
- 32-bit indexing is worth ~1.06× on address-bound flat kernels and **neutral** on
  register-bound column kernels. Keep `Index = int`; don't expect it to save a column
  solver.

Also established:

- `nvc++ -cuda` **does** accept `__global__` and `<<<>>>`; it warns
  "unsupported, nvcc recommended". It works; don't build on it. Put native
  kernels in their own TU.
- CUDA 13 changed `cudaMemAdvise`/`cudaMemPrefetchAsync` to take a
  `cudaMemLocation` struct. Guard on `CUDART_VERSION >= 13000`. *This is the
  device layer earning its keep on day one.*
- nvc++'s EDG frontend renders non-ASCII in `static_assert` messages as `???`.
  **Compiler-diagnostic strings stay ASCII.** Comments may use anything.
- The compile-time `Space` gate (host subscript of `Space::Device` → hard error)
  behaves identically under nvc++ and g++.

Aurora/PVC results pending (spike 05).

---

## 2. The core is the boundary contract, not the Arena

> A kernel receives **only trivially-copyable views + POD, by value**.

The spikes are the proof this line is in the right place: the pool was swapped
three ways and the launcher two ways, across two memory regimes and three
compilers, and no kernel body changed. Everything below the line is
implementation; everything above it is composition.

```
┌─ device/    the ONLY backend-aware code
│    backend tag · context (queue/stream) · alloc/free/memcpy/prefetch/sync
│    do_concurrent ->  stdpar | <<<>>> | sycl        (THE chokepoint)
├─ core/      Space · Tensor<T,Rank,Space> · Field<Rank> · mirror · Arena
├─ ══════════ THE BOUNDARY ═══════════════════════════
├─ mesh/      Mesh concept + region-parameterised iteration
└─ physics/ numerics/ diag/ api/
```

### 2.1 `device/` — and why SYCL is in from the start

CUDA and HIP are one backend wearing two hats (hipify is essentially textual);
supporting both proves nothing. **SYCL is the first genuinely different backend**,
and it forces the one thing CUDA lets you cheat on: an explicit **context/queue**.
`cudaMalloc` needs no context; `sycl::malloc_device` and `q.parallel_for` both do.
Designing CUDA-first and threading a queue through later touches every call site.

Aurora is additionally reachable *through stdpar itself* via oneDPL, so native
SYCL is a selective escape hatch exactly as native CUDA is — not a second
implementation.

| | host | CUDA/HIP | SYCL |
|---|---|---|---|
| device-only | `aligned_alloc` | `cudaMalloc` | `sycl::malloc_device` |
| shared/managed | `aligned_alloc` | `cudaMallocManaged` | `sycl::malloc_shared` |
| coherence query | — | `pageableMemoryAccess` | `aspect::usm_system_allocations` |

`TC_KERNEL` (→ `__host__ __device__`) is a CUDA/HIP-only concern; SYCL lambdas
need no annotation.

### 2.2 `core/` — two verbs, not one

Oceananigans has **`on_architecture`** (host *relocation*: copies, full fidelity,
round-trippable) and **`adapt_structure`** (device *projection*: copies nothing,
deliberately lossy, strips everything non-POD). Their `Field` adapts to *just its
array* — they arrived independently at our owner-never-crosses rule, across ~100
explicit declarations.

We have the first (`mirror()`). We need the second, systematically:

```cpp
template <class T> struct DeviceView;                  // primary undefined -> hard error
template <> struct DeviceView<BaroState> {
    using type = BaroStateView;
    static type of(const BaroState& s) { return {s.eta.view(), s.u.view(), s.v.view()}; }
};
```

**C++ is strictly better than Julia here.** Their failure mode is an opaque GPU
compiler error at first call; ours is `static_assert(is_trivially_copyable_v<...>)`
at the launch site, with a name attached. That improvement costs one line.

Keep from the mirror design: **`mirror()` is the identity where data is already
host-accessible.** One code path serves the host build, coherent machines, discrete
machines — *and the numpy boundary*. Shape `Tensor`/`mirror` with the numpy handoff
(extents, strides, dtype, lifetime) in mind so the C ABI doesn't invent a second
staging concept.

Arena: one pool, sized once, `seal()` after `init()`, RAII scratch scope above the
high-water line, a label per allocation.

---

## 3. The menu explosion — the constraint that reshapes everything

This is the most consequential finding, and it modifies our earlier instinct that
every axis is a compile-time policy.

Count the compile-time menus and *multiply* before committing:

```
dimensionality (3) × advection family (4) × order (5) × EOS (3) × vcoord (6) × immersed (2)
    ≈ 2000 instantiations
```

Oceananigans pays this as JIT latency — per user, per configuration — and their own
test suite carries the comment *"These are slow to compile..."*. We would pay it once,
as build time and binary size, for everyone. On nvc++, whose optimizer pipeline is
already the slow part, that is not a rounding error.

**Rule: a thing is compile-time only if its polymorphism boundary is per-cell.
If the boundary can sit at kernel-launch granularity, make it runtime.** One
virtual call per closure per step is unmeasurable; one per cell is fatal.

Two large wins fall straight out of that rule:

- **Coriolis collapses to a runtime 2D field.** f-plane / β-plane / spherical are
  three *types* in Oceananigans; a precomputed `f(i,j)` array subsumes the entire
  taxonomy at zero cost. Delete the axis.
- **Tier-A closures become a runtime list.** See §5.

---

## 4. Compose by inlining, not by launching

**Oceananigans' entire momentum RHS is one kernel.** `compute_..._Gu!` is a
two-line kernel whose body is a single sum of eight `@inline` pointwise functions:
advection + metric + barotropic PGF + Coriolis + hydrostatic PGF + closure
divergence + immersed BC + forcing. They never launch per term — which is why they
never needed a fusion framework.

Ours currently composes at the **launch** level (`cont_.compute()`, then
`pgf_.compute()`, then `cor_.compute()`, each launching several passes). Theirs
composes at the **inlined pointwise-function** level and launches once.

Adopt theirs. A policy contributes

```cpp
static Real at(FaceView f, StateView s, Params p);   // pure, inlined, no launch
```

and the core sums them inside one `for_each`. Not everything fuses — Coriolis
genuinely needs ζ at corners and KE at centres computed first — so the shape is
*a small number of prerequisite passes, then one combining kernel*, not one kernel
per operator per pass.

This is also the answer to the ROADMAP's barotropic-subcycle fusion item, and it
generalises instead of being a one-off.

**Corollary, and it is a warning.** Their launch-bound hot spot was the same one as
ours — the barotropic substep loop, ~100 tiny launches — and their fix was
**hoisting host-side argument marshalling out of the loop**, not writing a kernel.
In 198 kernels across five backends they wrote **zero** hand-written vendor kernels.
Their escape hatches are: vendor libraries for things kernels can't express (FFT),
*one* arithmetic intrinsic behind a compile-time policy type (an approximate
reciprocal for WENO's FP64 divides), streams/events the portability layer doesn't
model, and host-side launch-path surgery — which silently corrupted results on one
backend and had to be reverted.

**So: measure the host side first. If you must drop out, drop out at the
granularity of one arithmetic operation behind a policy type, not at the
granularity of a kernel.**

---

## 5. Closures: two tiers, split at the granularity seam

We concluded earlier that closures are additive ensembles rather than exclusive
slots. Oceananigans confirms it (a tuple of closures, summed) — and shows *where*
the polymorphism boundary actually sits, which we had wrong:

- **Tier A — producers.** PP81, KPP, EPBL, convective adjustment, tidal mixing,
  kappa-shear, MEKE, Smagorinsky/Leith-computing-a-viscosity. These write into
  shared `kv`/`kt`/`ks`/`νh` fields, **one kernel launch each**.
  → **runtime `std::vector<std::unique_ptr<IClosure>>`** with
  `contribute(state, diffusivities)`. Unlimited arity, no build explosion, one
  virtual call per closure per step. This is precisely what rakali's
  `vmix_assemble` gate already does.
- **Tier B — per-cell flux operators** that must inline into the tendency kernel.
  → **compile-time pack**, realistically 2–4 statically chosen kinds, written as a
  genuine C++23 fold:

```cpp
return (diffusive_flux_x(i, j, k, g, get<Is>(cs), get<Is>(ks)) + ...);
```

Not a hand-unrolled arity ladder. Oceananigans has four such ladders with
*different* limits (5, 4, 4, 3), one of which has no fallback at all — the
canonical cost of hand-unrolling in a language without folds.

Steal three more of their closure ideas verbatim:

- **`static constexpr int halo_width` on each scheme/closure; the framework takes
  the `max` over the ensemble and sizes the halo.** They encode it as a type
  parameter and it is genuinely elegant.
- **Direction-selectivity as a trait that zeroes irrelevant flux components**
  (Horizontal / Vertical / ThreeDimensional), not as branching inside kernels.
  `if constexpr` in C++.
- **Ordering constraints resolved at compile time.** Their `validate_closure`
  *sorts* the user's tuple at runtime so CATKE lands first, with a self-indicting
  comment. We get `constexpr index_of_first<...>()` plus a `static_assert` on
  duplicates — correctness *and* the duplicate check, for free.

---

## 6. The type/value line

Compile-time = template parameter / `if constexpr`, chosen at setup by a dispatch
table. Runtime = plain value or a virtual interface called at launch granularity.

| thing | ours | why |
|---|---|---|
| precision, backend | **compile-time** (build config) | pervades every kernel. **Never a mutable global** — Oceananigans' `defaults.FloatType` makes precision depend on construction *order* |
| grid dimensionality, `Flat` | compile-time | collapses loop nests; menu of 3 |
| grid geometry family | compile-time; metrics stay runtime 2D arrays | access pattern differs per cell |
| grid size / spacing / extents | runtime | |
| per-boundary topology | **runtime** | only affects a boundary-plane kernel |
| advection family + order | compile-time, fixed menu | per-cell stencil. This is exactly what we lose vs Julia — see §7 |
| halo width | compile-time `static constexpr`, `max` at setup | |
| closure ensemble membership | **split — Tier A runtime, Tier B compile-time** | §5; the single most important decision here |
| closure tuning parameters | runtime | |
| **Coriolis form** | **runtime 2D field** | collapses the taxonomy at zero cost |
| EOS, buoyancy formulation, vcoord | compile-time | per-cell |
| free-surface solver, substeps | runtime | once per step |
| BC classification + condition data | runtime (tagged union) | boundary-plane kernels |
| forcing presence/kind | runtime list per field; empty ⇒ no launch | |
| tracer set | runtime registry | as rakali already has |
| diagnostics | compiled catalog, runtime selection by name | §8 |

---

## 7. Scripting: construct → validate → materialize

No input files. A Python script is the configuration. The load-bearing pattern,
and it survives the compile-time/runtime split intact:

1. Python builds **inert, type-erased spec objects** (dataclasses).
2. One `materialize(spec, grid, arch)` factory pass validates and returns the
   concrete monomorphic engine.

Oceananigans has a named `materialize_*` per concern and it is their most portable
idea. Adopt it, with these rules:

- **The grid is the single source of truth for device and precision.** No
  `device=` on the model, no global precision switch.
- **`None` means off, and off is the default for everything optional** —
  bit-identity by construction, the property rakali's namelist defaults have.
- **Coordinate arguments accept `tuple | callable | array`.** One argument, three
  representations. Perfect fit for Python.
- **Bathymetry is a grid decorator**, not a model knob; takes function-or-array,
  materialised into a field at wrap time.
- **Bottom drag is a boundary condition**, not a physics group.
- **`repr(model)` prints the fully resolved post-materialization config.** With no
  input file, this **is** the provenance record — make it round-trippable to a
  reproducible script, and emit it plus a git SHA at startup.
- **Never silently re-type what the user asked for.** Oceananigans downgrades a
  `WENO(order=9)` to `Centered(order=8)` on a short axis and announces it with a
  log line, so `model.advection` no longer equals what was passed. Throw, and say
  what is required — their own hydrostatic model does exactly that while the
  nonhydrostatic one warns. Pick throw.
- **Errors name the value, the constraint, and the fix**, and list what *is*
  available. `"WENO order 11 not built; available: 3, 5, 7, 9"`.

### 7.1 What Python functions cannot do — the largest divergence

Julia inlines a user's boundary-condition function straight into the GPU kernel.
**A Python function cannot run in a device kernel, full stop.** This should drive
the whole forcing/BC design. Three tiers, named in the API so the performance model
is legible:

1. **`ArrayForcing` — the default, not the escape hatch.** Python fills a numpy
   array; we take a zero-copy view. Covers most real usage.
2. **`ScalarForcing(callback, cadence=…)`** — a host-mutable scalar re-read each
   step. Oceananigans' `getbc(::NumberRef) = condition[]` is exactly this, and it
   covers every time-dependent-but-spatially-uniform driver (tides, wind ramps).
3. **`AnalyticForcing(shape=…)`** — a small closed menu of parameterised forms
   (their `Relaxation(rate, mask=GaussianMask, target=LinearTarget)` is the shape).

**Do not offer "pass any Python function and we'll make it fast."** A per-cell host
callback will be orders of magnitude slower and users will not understand why. MOM6
agrees: its escape hatch is a *recompiled Fortran module*, not a runtime callback.

> **Three tiers are not enough — a real global configuration needs a fourth.** The
> largest heat-flux terms in a forced run are **not prescribed**: latent, sensible and
> upward longwave depend on the model's own SST through bulk formulae; SST/SSS
> restoring depends on the model's SST/SSS; frazil depends on `T < T_freeze(S,p)`; and
> penetrated shortwave feeds back into the boundary layer's buoyancy forcing. None of
> those is a numpy array, a host scalar, or a closed-form function of `(x, y, t)`.
>
> **Tier 4 — named, compiled, device-side flux kernels reading the prognostic state**
> plus tier-1 arrays and tier-2 scalars: `tc.SstRestoring(piston=…, target=array)`,
> `tc.BulkFormulae(t_air=…, q_air=…, wind=…)`, `tc.Frazil()`. Architecturally this is
> not a new mechanism — it is an **ensemble** (`CONTRACT_RUNTIME.md` §4.2), each member
> contributing into `Q_heat`/`Q_salt`/mass through one assembly gate with a
> single-writer rule.
>
> Two consequences:
>
> - **Global net-zero adjustments are collectives.** Restoring fluxes must net to zero
>   globally or the ocean drifts — an area-weighted global integral **every step**, then
>   a broadcast scalar. It looks like tier 2, but the scalar comes from a device-side
>   collective. Another reason `Sum::Reproducible` has to be real rather than seamed.
> - **Time interpolation stays on-device.** Forcing data is 3- or 6-hourly against a
>   ~20-minute `dt`, so ~20 steps per record. Pushing a fresh numpy array each step
>   round-trips the host and reinstates the measured 4–15× penalty. Keep an `f0`/`f1`
>   bracket **device-resident** and blend by a host scalar weight — tier 2 driving
>   tier 1.
>
> And since the arena is **sealed after init**, every forcing field that will ever exist
> must be declared at configure time. Adding one mid-run doesn't merely fail — it
> dangles views. `validate()` checks that alongside the tracer count.

### 7.2 Boundary conditions

Steal the **(classification, condition)** orthogonal pair: classification is the
maths (Flux / Value / Gradient / Flather / …), condition is the data
(scalar | host-mutable ref | device array | menu-selected form). It means
`ValueBoundaryCondition(c, scheme=Radiating())` adds a radiating open BC without
inventing a new name.

Note the semantic trap to document: Value/Gradient are implemented by halo
extrapolation, Flux by adding to the tendency. Choosing between them changes the
numerical operator, not just the number.

---

## 8. Driving a run

`run!` in Oceananigans is three lines: `while (running) time_step!(sim)`. Take it.

- **Stop criteria are callbacks that flip `running`.** Time, iteration, wall-clock
  are each a small function on the same footing as anything a user adds.
  Composition is free and `sim.stop_time = 42; sim.run()` resumes a finished run.
- **Time-step alignment driven by schedules.** Every schedule answers
  `next_actuation_time()`; the driver takes the min and *shortens the physics step*
  so output lands exactly on requested times — no interpolation, no drift.
  **Clamp the shortened step against a floor from day one**; they added
  `minimum_relative_step` only after a near-zero aligned step blew up a solve.
- **Schedules must be pure.** Theirs are stateful *and* effectful (`schedule(model)`
  both tests and records), which produced a documented-but-unfixed `AndSchedule`
  desync (because `all` short-circuits, so one child is never queried), a `deepcopy`
  helper, a fire-and-discard call at init, and extra checkpoint surface. Split
  `fires(clock) -> bool` from `record()` and all four disappear.
- **Callsites as a first-class enum**: `AFTER_STEP`, `AFTER_TENDENCIES`,
  `AFTER_UPDATE_STATE`. The tendency callsite is what lets a Python user inject
  forcing without touching C++. **Partition the callback list by callsite once at
  init** — they rebuild a tuple from a type-unstable generator every single step.
- **Schedule combinators** (`Or`, `And`, `ConsecutiveIterations`, `SpecifiedTimes`)
  plus "any predicate on the clock is a schedule".
- **Catch-up semantics decided once and tested**: a clock jump past several
  intervals fires once (their choice), not N times.
- **`Units`** — `1*day`, `20*minutes`. Trivial in Python, large readability win.

### 8.1 Checkpoint / restart

Steal `prognostic_state` / `restore_prognostic_state!` as a **recursive protocol**:
every object — model, timestepper, schedule, callback, writer, accumulator —
reports its own restartable state as a tree, and the checkpointer serialises the
tree. This is why they can restore a half-accumulated time average and a writer's
part number, which most codes lose.

Save the tendencies (that is what makes AB2 restart bit-exact). **Save the RNG** —
they don't, and stochastic forcing silently diverges across a restart.

---

## 9. Diagnostics

Their `Scan{type, scan!, operand, dims}` is *exactly* our integrand × reduction ×
dimension-set. Good validation. What they add is that the **integrand is a
composable algebra** — `Average(w*b, dims=(1,2))` written in script code, fused into
one kernel, with staggering resolved automatically at tree-construction time
(`choose_location` is four lines).

**But the tree has a hard cliff**, admitted in their own docs: past a certain depth
the GPU compiler fails — `u^2` works, a 9-term dissipation-rate expression does not.
Users discover it at runtime as an obscure crash. Tellingly, their own serious
diagnostics are written as `KernelFunctionOperation`s — i.e. our named registry —
not as trees.

For us the constraint is decisive rather than merely risky: **we are AOT-compiled;
a C++ template expression tree cannot be instantiated from a Python string.**

**The synthesis:**

1. Keep the compiled registry of named, staggering-tagged, pure per-cell
   integrands as the primitive set.
2. Keep reduction factored from integrand, over an arbitrary `dims` set.
3. Add a **small closed runtime-interpreted composition layer**: a fixed node set
   (`+ - * /`, unary maths, `∂x/∂y/∂z`, `interp_to(loc)`, constants) evaluated by
   *one* precompiled generic kernel walking a depth-capped flat opcode array.
   Costs ~10–20% versus full fusion on deep trees; buys `w*b`, `∂x(v)-∂y(u)`,
   `sqrt(u²+v²)` from Python with no rebuild and no compile cliff.
4. **Steal `choose_location`** so the interpreter inserts interpolation
   automatically and C-grid diagnostics are correct by construction.

Two of our existing positions are confirmed by their code: diagnostics computed on
device with only reduced results crossing; and a name-indexed registry that multiple
sinks iterate. **Add memoization on clock time** so N writers sharing a diagnostic
compute it once.

Time-averaging belongs **on device in a persistent accumulator driven by the
schedule system** — never "write every step and average offline". Accumulate
`sum += x·Δt` with a single divide at emit, not their repeatedly-rescaling running
form.

Watch the costs they pay and we should not: every write allocates a fresh unpooled
host buffer, and their NetCDF path converts `Float32` back up to `Float64` and down
again — three full-size host allocations per output per write.

---

## 10. Iteration: one chokepoint, region-parameterised

`for_each_cell/face(mesh, …)` becomes the **single** launcher the `device/` layer
dispatches — the two current families (raw-extent and mesh-driven) collapse into
one. Then add two things now, cheaply, that are expensive to retrofit:

- **A region parameter**, default `interior()`. Oceananigans' polymorphic work
  specification (`KernelParameters(size, offsets)`) is one mechanism serving
  comm/compute overlap, halo-inclusive computation, and periphery exclusion. Their
  entire `Distributed` launcher override is **six lines**, because the MPI
  decomposition changes the *workspec*, not the launcher. Add the parameter before
  MPI, not after.
- **A `FaceSource` seam** with `Dense` and `Compressed` models. Their active-cells
  map compresses 3-D iteration into a 1-D gather over a precomputed index list with
  **the operator body unchanged** — a large constant-factor win for a
  bathymetry-fitted ocean, and it composes with the distributed split. Our
  `FaceView` design makes this *easier* than theirs, since we already gather
  connectivity per face.

Keep threads on x–y, never on z: their `(16,16)` blocks with z on the block grid is
right for column-serial ocean physics. Make the vertical the slowest-varying part of
the flat index so columns stay block-local.

---

## 11. What we are deleting

- `Params::{nx, ny, dx, dy, H, H1, H2}` — verified read **nowhere** in `src/`,
  `examples/`, or `tests/`. The mesh ate them, as ADR-7 predicted. `H1/H2/rho1/rho2`
  move onto the PGF policy; `Params` drops to `{dt, g}`, and the NL=2 weld goes with
  it.
- The **BC policy axis** as it stands. `WallBC`/`PeriodicBC` are no-ops; topology
  already lives in the mesh and is consumed at access time via `mesh.edge()` /
  `shift_x` / `bc_at`. Two designs are on the field; the mesh-owned one is the one
  that actually runs periodic-x on a sphere. Pick it, delete the other, and let the
  halo return only with ghosts.
- `lib/config.hpp` / `RunConfig` / presets — the Python script is the config.
- `OceanCore` once `MultilayerCore<1>` covers it.
- ADR-5's `Vec<N>`, which never arrived and is not missed.
- Most of the 19 stub headers. They encode rakali's Fortran decomposition as fact.
  Collapse them into one design note; re-add each file when its first real
  implementation lands.

## 12. Build order

1. **`device/`** — backend tag, context, alloc, `do_concurrent`, `TC_KERNEL`.
   Standalone, independently testable, spikes are its test suite.
2. **`core/`** — `Space`, `Tensor`, `mirror`, `DeviceView`, Arena on the pool.
3. **Migrate consumers** to `.view()` / `mirror()`.
   **Gate: no operator math edited.** If that trips, the boundary is not where we
   think it is — which is worth learning on a mechanical refactor rather than in M4.
4. Collapse the iteration families; add the region parameter.
5. Physics: fuse the tendency; split closures Tier A / Tier B.
6. The Python API and the driver.

## 13. A CI gate worth having from the start

Oceananigans puts a **hard byte budget on host allocations per `time_step!`** and
had to move it when they reduced launch overhead. Our analogue: assert that a step
at fixed problem size issues ≤ N kernel launches, and that host-side per-step time
stays under a threshold. Without a gate, abstraction overhead accretes invisibly.
