# alpha — `device::do_concurrent` over hand-managed device memory

Three files, no configure step, one question:

> **Does a parallel algorithm accept pointers WE allocated with the native device
> allocator — no managed memory, no implicit migration?**

That is the thesis the whole design rests on: *the language expresses parallelism,
we express memory, native kernels insert rather than replace.* Two thirds of the
matrix are already proven; one cell is not.

| toolchain | allocator | result |
|---|---|---|
| `nvc++ -stdpar=gpu -gpu=mem:separate`, cc90 | `cudaMalloc` | **PASS** |
| `hipcc --hipstdpar` | `hipMalloc` | **PASS** |
| `icpx -fsycl` + oneDPL | `sycl::malloc_device` | **PASS** |
| cc70 (V100) | `cudaMalloc` | pending — environment, not design |

**Answered: yes, on all three vendors.** The AMD cell was the one in genuine doubt.
`--hipstdpar` is not hipified CUDA — it leans on HMM/XNACK to make *ordinary host
allocations* device-reachable, so its design assumption is that you did **not**
hand-manage memory. Had it required interposing `malloc`, the arena would have
been fighting it rather than composing with it, and AMD would have needed the
native insertion path for everything rather than selectively. It does not.

So *"CUDA is basically HIP"* now extends to the **stdpar** path and not just to
`__global__` kernels, which was not a safe assumption before this ran.

## Run

```
make probe                     # which toolchains exist on this box
make run-host                  # free, no GPU, proves the shape compiles
make run-nvidia ARCH=cc90      # the control
make run-amd    GFX=gfx90a     # the unknown
make run-intel
```

Paste the whole output back — the banner identifies box, backend and which
parallel algorithm was selected, which is most of what makes a result readable
six months later.

If AMD needs help finding rocThrust:

```
make run-amd GFX=gfx90a HIPSTDPAR_EXTRA="--hipstdpar-path=/opt/rocm/include/thrust"
export HSA_XNACK=1        # some configurations need this
```

## Why correctness proves offload, with no timing heuristic

The pointers come from `cudaMalloc` / `hipMalloc` / `sycl::malloc_device` — not
managed, not migratable, **not host-dereferenceable**. So if a `do_concurrent`
writes the right values and a device→host copy reads them back correctly, the loop
body *must* have run on the device: host execution over that pointer would fault
or produce garbage, not the right answer.

A vendor `ALL PASS` therefore means both *"stdpar accepted our hand-managed device
memory"* and *"it actually offloaded"*, with no wall-clock inference. The **host**
target proves neither and says so in its own output — it exists to catch shape
errors for free before you queue on a real machine.

## What each check is for

| check | catches |
|---|---|
| elementwise fill | the basic question: does the algorithm touch device memory at all |
| 2-D flattened stencil | neighbour reads, and an `i`/`j` split that is *defined but wrong* — the Laplacian of a linear field is identically zero, so a transposed split shows up as non-zero rather than as a crash |
| reduction | `transform_reduce`, the other primitive, which diverges more across backends than `for_each` |
| float and double | the same lambda body instantiating at both precisions |

## The one rule for this directory

**`main.cpp` contains no `#ifdef`.** Every per-backend dialect lives in
`device.hpp` — the oneDPL `counting_iterator` (oneDPL rejects `std::views::iota`,
measured in spike 05c), the queue that exists only because SYCL requires one for
both allocation and launch, the allocator switch.

If a backend ever forces an `#ifdef` into `main.cpp`, `do_concurrent` has failed
at its only job, and *that* is the finding worth reporting.
