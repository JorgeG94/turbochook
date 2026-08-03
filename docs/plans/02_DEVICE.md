# 02 — Device manager

**Depends on:** [`00_VOCAB.md`](00_VOCAB.md) — `MemoryQuantity` (the allocator takes one)
and `TC_ASSERT`. Nothing else; it can otherwise run in parallel with the 01 freeze.
**Its test suite already exists** — the spikes, `alpha` included.

Contracts: [`../CONTRACT_LAYERS.md`](../CONTRACT_LAYERS.md) (this is the only
backend-aware directory in the tree) · [`../CONTRACT_MEMORY.md`](../CONTRACT_MEMORY.md)
§0.0, §2.

---

## Scope

The **only** backend-aware code in the tree. Allocation, context, launch, reduce,
capability query. Nothing else — no physics, no arrays, no Python.

```
src/lib/device/backend.hpp     Backend tag, TC_KERNEL, KernelTag, Regime (annotation)
src/lib/device/context.hpp     Context (stream/queue), initialize/finalize/sync
src/lib/device/alloc.hpp       DeviceAllocation, malloc_device/free/memcpy/prefetch
src/lib/device/launch.hpp      do_concurrent, reduce, Region, LaunchOpts, GraphScope
src/lib/device/native.cu       __global__ trampolines + TC_REGISTER_NATIVE (nvcc)
tests/test_device.cpp
```

## The thesis this workstream implements

`CONTRACT_MEMORY` §0.0, restated because it decides half the design below:

> **The language expresses parallelism. We express memory. Native kernels insert,
> they do not replace.**

**Measured, four for four** (`spikes/alpha`, one `#ifdef`-free `main.cpp` over pointers
from the native allocators): nvc++ `-stdpar=gpu -gpu=mem:separate` on cc70 **and** cc90,
`hipcc --hipstdpar` over `hipMalloc`, `icpx -fsycl` + oneDPL over `sycl::malloc_device`.

Two consequences that shape this plan:

1. **stdpar is *the* path**, not one of two. There is no routine dispatch decision to
   make — see the `Launcher` note under Decisions.
2. **The per-backend dialect surface is ~4 primitives, not N launchers.**
   `do_concurrent`, `reduce`, `alloc`, `memcpy`. Fifty kernels, one file that knows a
   backend exists.

## Backend support tiers

Stating this explicitly, because an unwritten tier quietly becomes untrue:

| backend | stdpar tier | native tier |
|---|---|---|
| CUDA (nvc++) | **yes** | **yes** — the reference native path |
| HIP | **yes** (`--hipstdpar`, measured) | **yes**, via hipify — textual, ~0.1 of a backend |
| SYCL / Intel | **yes** (oneDPL, measured) | **on profile evidence only** |
| host / multicore | **yes** | n/a |

**SYCL gets no native launcher until a specific kernel profiles badly enough there to
justify one.** It is the one genuinely different backend — it forks the *stdpar* path
too, not just the native one, since oneDPL rejects `iota_view` — so a second full
launcher for it is the most expensive thing in the matrix and the least evidenced.

Both PVC findings we already paid for (one tile not the composite root; flat 1-D beats
`range<2>` by 1.8×) live in the **stdpar** launcher, where they benefit every kernel
rather than only the ones someone hand-wrote. `native.sycl.cpp` is therefore **deferred**,
not planned.

## Work order

1. **`alloc.hpp` first** — it unblocks 03. Lift `spikes/common.hpp` almost verbatim,
   including the `CUDART_VERSION >= 13000` guard for the `cudaMemLocation` signature
   change (CUDA 12 vs 13 differ; both boxes in use).

   The owning type, and four things to get right:

   ```cpp
   class DeviceAllocation {          // ONE per pool, NOT per Array
   public:
       DeviceAllocation(MemoryQuantity size, Kind kind, Context& ctx);  // throws
       ~DeviceAllocation();                                             // logs, NEVER throws
       DeviceAllocation(DeviceAllocation&&) noexcept;
       DeviceAllocation& operator=(DeviceAllocation&&) noexcept;
       DeviceAllocation(const DeviceAllocation&)            = delete;
       DeviceAllocation& operator=(const DeviceAllocation&) = delete;
       void* get() const noexcept;
   };
   ```

   - **The destructor must not throw.** Destructors are `noexcept(true)`, so a
     `CUDAAssert` on a failed free calls `std::terminate` with no diagnostic.
     Constructors throw; destructors log and swallow.
   - **Rule of five.** A raw owning pointer with an implicit copy constructor
     double-frees the pool the first time one is passed by value.
   - **`Context&`, non-optional.** SYCL *requires* a queue for both `malloc_device` and
     `free`; CUDA/HIP have an implicit default and ignore theirs. `std::optional<Stream*>`
     would also have two distinct empty states, only one of which gets checked.
   - **The host build must not `Fail()`.** It is where `ctest`, the hardened run and
     workstream 00's whole acceptance gate live — not a degraded fallback.
     `std::aligned_alloc`, as `spikes/common.hpp:247` already does.

   **One plain `cudaMalloc` per pool, not `cudaMallocAsync`.** The async allocator's
   value is stream-ordered reuse across repeated alloc/free, which the sealed arena
   exists to prevent — and a stable base pointer is what makes `GraphScope` legal.
2. **Context.** A queue on SYCL, a stream on CUDA/HIP, nothing on host. Exists
   *because* of SYCL — `sycl::malloc_device` and `q.parallel_for` both need it, CUDA
   lets you cheat, and threading it through later touches every call site.
3. **`initialize(device)` — the two hardware traps, encoded in the API:**
   - **Intel: select ONE TILE, never a COMPOSITE root device.** Measured **6.3×**
     (392 → 2479 GB/s). Query `partition_max_sub_devices`; if the selected device
     partitions, take a sub-device.
   - **CUDA: `CUDA_VISIBLE_DEVICES` must be pinned per rank before `MPI_Init`** —
     that ordering lives in 04, but `initialize()` must document and assert it.
4. **`do_concurrent`.** Contractually **may be async**; `sync()` is required before any
   host read. True today (stdpar blocks) and forever after — documenting it as
   synchronous now would make streams an audit of the whole codebase later.
   Four implementations:
   ```
   Serial/Multicore/nvc++ : std::for_each(tc::par, iota.begin(), iota.end(), f)
   SYCL                   : oneapi::dpl::for_each(device_policy(), counting_iterator(0), +n, f)
   CUDA/HIP               : tc_trampoline<<<blocks, wg>>>(n, f)
   ```
   **`iota_view` is rejected by oneDPL** (read-only proxy iterator) and
   **`-fsycl-pstl-offload` halves the whole binary** — measured. Intel uses
   `counting_iterator` and no offload flag.
5. **`reduce`.** `std::transform_reduce(tc::par, …)` on three backends, oneDPL on
   SYCL. `Acc` explicit and possibly wider than the field type. Support a **struct
   accumulator** so mass + KE + max|u| fuse into one sweep — three separate calls is
   three sweeps and three syncs.
6. **`Region`.** Default `interior()`; an offset box otherwise. Nothing uses the
   offset form until 04, and that is the point.
7. **Native launcher + `TC_REGISTER_NATIVE`.** `has_native_v<F>` defaults false, so
   kernels without a native twin compile to a bare stdpar call with **no branch at
   all**. Register two at most for now — this is a verification tool before it is a
   performance tool.
8. **`GraphScope`** — RAII, captures on first entry and replays after. No-op on
   stdpar. Legal only because the arena never reallocates.
9. **The byte + FLOP recorder.** Per-kernel bytes moved and flops, so achieved bandwidth
   and arithmetic intensity fall out and place each kernel on the roofline
   automatically. Cheap, and it is what turns `Regime` from an assertion into a
   measurement — see below.

## Decisions to record

- **`Launcher::Auto` is deleted, not deferred.** It existed to *choose* between stdpar
  and native. Under the §0.0 thesis there is no routine choice: stdpar is the path, and
  native is an insertion at a named kernel. A dispatch mechanism with one input value is
  machinery pretending to be a decision.

  What survives:
  - **`Launcher::AB`** — runs both on the same pool and asserts bit-agreement. Spike 03
    proved one pool feeds both, so the single-binary harness is free, and it is the
    *only* comparison `more_benchmarks` trusts. This is how an insertion gets justified.
  - **`Regime`** — demoted from a dispatch input to a **human annotation**, and better,
    one the recorder can check. "The author said memory-bound" becomes "this kernel hit
    780 of 900 GB/s at AI=0.3, so it is memory-bound and a native rewrite cannot help
    it" — which is the actual criterion for whether an insertion is worth writing.

  The measurement that motivated the old rule still stands and still argues against
  automatic selection: the stdpar/native gap **shrinks** as the grid grows (launch
  amortisation), so any `nx*ny*nz > threshold` rule picks native exactly where it helps
  least.
- **Named functors, not lambdas, for any kernel with a native path.** A lambda in an
  nvc++ TU cannot be handed to an nvcc-compiled `__global__`. This is the one real
  authoring cost of the native tier, and it is confined to the few kernels that earn it.
- **`do_concurrent` is the portability boundary, and that is worth more than the
  coalescing argument that first motivated it.** The launcher owning flattening was
  justified by `i = n % nx` being a correctness-for-performance invariant. `alpha` showed
  the larger payoff: the same shape absorbed three vendors' dialects with no `#ifdef`
  above it. If a backend ever forces one into a kernel body, the abstraction has failed
  at its only job and that is the finding.

## Tests

Port the spikes; do not rewrite them.

| from | asserts |
|---|---|
| **spike alpha** | **the §0.0 thesis: stdpar over hand-managed device memory, all four toolchains** |
| spike 01 | managed pool + prefetch stays device-resident; host-touch ratio reported |
| spike 02 | device-only pool works under stdpar |
| spike 03 | one pool, two launchers, bit-identical |
| spike 05 | SYCL USM aspects; **tile selection** |
| new | `reduce` matches a serial reference; struct accumulator fuses |
| new | `Region` offset box visits exactly the intended cells |
| new | `initialize()` on Intel selects a sub-device when one exists |
| new | `DeviceAllocation` is move-only; a moved-from one frees nothing |
| new | the recorder's achieved-bandwidth number matches a hand-computed one |

`alpha` is the regression test for the thesis and the first thing to re-run when a
toolchain moves. Port the spikes; do not rewrite them.

## Acceptance gate

- Every spike green on **V100, GH200, PVC, host** through the real `lib/device/` —
  including `alpha`, which must stay `#ifdef`-free above `device.hpp`.
- `capabilities().coherent` correct on all three GPUs (it is `pageableMemoryAccess`
  on CUDA and `aspect::usm_system_allocations` on SYCL — one portable answer).
- Intel path selects a tile: bandwidth matches the 2479 GB/s single-tile figure, not
  the 392 GB/s composite one.
- A kernel with no native registration emits no dispatch branch (check the assembly).
- The host build works with **no GPU toolchain installed at all** — it is where the
  hardened run and 00's acceptance gate live, not a degraded fallback.

## Deferred

`native.sycl.cpp` (see the tier table — on profile evidence only), multi-stream
scheduling policy, graph capture beyond the RAII scope, cooperative/persistent kernels.

**Team-level parallelism is the one deferral with a decision attached.** A flat index
callback cannot express "team of N threads sharing scratch", and the CUDA and SYCL team
models diverge considerably more than their flat models do — so a second primitive would
cost far more than `do_concurrent` did. It is also the *only* remaining argument for
adopting an external framework, every other one having been closed by `alpha`. **D2, the
column-kernel probe, is the measurement that decides it**: column solvers are ~58% of an
ocean stage, and `CONTRACT_MEMORY` §3.1 already records a 0.22× disaster from getting the
per-thread private set wrong. Do not build a team abstraction before that number exists.
