# 02 — Device manager

**Depends on:** nothing. **Can start immediately**, in parallel with the 01 freeze.
**Its test suite already exists** — the five spikes.

Contract: [`../CONTRACT_MEMORY.md`](../CONTRACT_MEMORY.md) §2.

---

## Scope

The **only** backend-aware code in the tree. Allocation, context, launch, reduce,
capability query. Nothing else — no physics, no arrays, no Python.

```
src/lib/device/backend.hpp     Backend tag, TC_KERNEL, Regime/Launcher/KernelTag
src/lib/device/context.hpp     Context (stream/queue), initialize/finalize/sync
src/lib/device/alloc.hpp       malloc_device/malloc_shared/free/memcpy/prefetch
src/lib/device/launch.hpp      do_concurrent, reduce, Region, LaunchOpts, GraphScope
src/lib/device/native.cu       __global__ trampolines + TC_REGISTER_NATIVE (nvcc)
src/lib/device/native.sycl.cpp native SYCL launcher (icpx)
tests/test_device.cpp
```

## Work order

1. **`alloc.hpp` first** — it unblocks 03. Lift `spikes/common.hpp` almost verbatim,
   including the `CUDART_VERSION >= 13000` guard for the `cudaMemLocation` signature
   change (CUDA 12 vs 13 differ; both boxes in use).
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

## Decisions to record

- **Named functors, not lambdas, for any kernel with a native path.** A lambda in an
  nvc++ TU cannot be handed to an nvcc-compiled `__global__`. This is the one real
  authoring cost of runtime launcher choice, and it is confined to the few kernels
  that earn it.
- **`Launcher::Auto` resolves by `Regime`, not by problem size.** Measured: the
  stdpar/native gap *shrinks* as the grid grows (launch amortisation), so a
  `nx*ny*nz > threshold` rule selects native exactly where it helps least.
  MemoryBound → stdpar; LaunchBound / RegisterBound → native.
- **`Launcher::AB`** runs both on the same pool and asserts bit-agreement. Spike 03
  already proved one pool feeds both launchers, so the single-binary harness — the
  only comparison `more_benchmarks` trusts — is free.

## Tests

Port the spikes; do not rewrite them.

| from | asserts |
|---|---|
| spike 01 | managed pool + prefetch stays device-resident; host-touch ratio reported |
| spike 02 | device-only pool works under stdpar |
| spike 03 | one pool, two launchers, bit-identical |
| spike 05 | SYCL native; USM aspects; **tile selection** |
| new | `reduce` matches a serial reference; struct accumulator fuses |
| new | `Region` offset box visits exactly the intended cells |
| new | `initialize()` on Intel selects a sub-device when one exists |

## Acceptance gate

- All five spikes green on **V100, GH200, PVC, host** through the real `device/`.
- `capabilities().coherent` correct on all three GPUs (it is `pageableMemoryAccess`
  on CUDA and `aspect::usm_system_allocations` on SYCL — one portable answer).
- Intel path selects a tile: bandwidth matches the 2479 GB/s single-tile figure, not
  the 392 GB/s composite one.
- A kernel with no native registration emits no dispatch branch (check the assembly).

## Deferred

Multi-stream scheduling policy, graph capture beyond the RAII scope, HIP backend
(shape it, don't build it), cooperative/persistent kernels.
