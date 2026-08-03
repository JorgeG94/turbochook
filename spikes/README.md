# Memory-model spikes

Toolchain questions that must be answered **before** the Arena is rewritten, in the
spirit of M0: retire the risk first, then build on it. Each spike is one question,
one binary, one line of output that settles it.

None of this is `src/` code. It exists to be run once on a real GPU, reported back,
and then deleted or folded into a design note.

## The question behind all of them

We want a pool that is **allocated once at startup, pinned, never grows**, with
`stdpar` as the default execution layer and hand-written CUDA/HIP kernels available
where they pay. Two things stand in the way:

1. `-stdpar=gpu` normally works by intercepting the system allocator so all heap
   data is *managed* — which is why `std::vector` offloads today. Owning the pool
   ourselves means not relying on that.
2. Under `mem:separate` a Fortran `do concurrent` kernel needs OpenACC/OpenMP data
   directives to map named arrays. For a **raw device pointer captured by value**
   there is nothing to map — but whether nvc++ *permits* that configuration is a
   toolchain-policy question, not a semantics one.

## Run order

```bash
cd spikes
make host                   # 01/02/04 with no GPU and no NVHPC — run this anywhere first
make s04_negative           # MUST FAIL to compile
make run ARCH=cc80          # the real thing on the GPU box; set ARCH to your card
```

`make host` is the control. The pool falls back to `aligned_alloc` and staging to
`memcpy`, so the logic runs on a laptop and spike 01 reports a **~1.0x** ratio —
which is exactly what a coherent machine should also report. Seeing the control
first makes a large ratio on the real card meaningful rather than mysterious.
(Verified on macOS/Apple clang: all three PASS, ratio 1.0x.)

`ARCH` is `cc70` (V100), `cc80` (A100), `cc90` (H100/GH200). `CUDA_HOME` must be set
for `s03b`.

## What each one asks, and how to read it

### 04 — compile-time space tags (host only, run first)

Can "host code touched device memory" be a **compile error** rather than a runtime
fault, and does `mirror()` collapse to nothing when the data is already
host-accessible?

- **PASS** = the API shape works, and one diagnostic/IO/test path can serve the host
  build, the coherent build, and the discrete build with no `#ifdef`.
- `make s04_negative` **must fail to compile.** If it links, the gate is decorative.

This is the cheapest and most reusable result — it fixes the `Tensor`/`Field` API
before any hardware is involved.

### 01 — managed pool + advise + prefetch, under stdpar

Can we own the pool (`cudaMallocManaged`, sized once, prefetched to device at init)
rather than lean on allocator interception, and does it stay device-resident?

Prints two variants and their ratio:
- **A** — host never touches the pool during the loop.
- **B** — host reads one element per iteration.

`A/B` is the migration penalty. **≫1 ⇒ discrete** (the data really moves, and the
prefetch is what saves you). **≈1 ⇒ coherent** (Grace-Hopper class;
`pageableMemoryAccess=1` in the header confirms it).

This is the row we expect to carry the design. If it passes, the Arena becomes an
explicit sealed pool that also works under native CUDA/HIP.

### 02 — device-only pool (`cudaMalloc`) under stdpar

The strict version: host cannot dereference the pool at all, so the space tag is
enforced by hardware as well as by the type system.

**A compile failure or a clean runtime error is a result, not a bug.** Paste the
exact message from `build/s02_build.log`. If nvc++ rejects
`-stdpar=gpu -gpu=...,mem:separate`, that closes the question and spike 01 carries
the design — nothing above the Arena changes either way.

### 03 — one pool, two launchers, one binary

The same sealed pool and the same view feeding **both** a stdpar kernel and a
hand-written `__global__`, in one TU compiled by `nvc++ -stdpar=gpu -cuda`.

This is the whole "stdpar by default, native where it pays" strategy in miniature.
If both launchers agree bitwise, native kernels are a **selective** optimisation for
the few launch-bound spots (the barotropic subcycle first) — five hand-written
kernels, not fifty. If they cannot share a pool, discrete GPUs would need a complete
second implementation of every operator, which is a much more expensive commitment
and worth knowing before making it.

The reported `stdpar/cuda` ratio on this large kernel should be near 1 (both are
bandwidth-bound). The gap that matters shows up on *tiny* kernels, which is exactly
the case the native path exists for.

### 03b — g++ host + nvcc device layer

Does the "ordinary system compiler for the host, nvcc only for the device layer"
split hold, with the host TU never including a CUDA header?

This is the fallback that makes the project independent of NVHPC — relevant because
"stdpar on GPU" is vendor-specific in practice (nvc++ for NVIDIA,
`clang --hipstdpar` for AMD, `icpx`/oneDPL for Intel). `03b_device_layer.cu` is the
prototype of the `device_alloc` seam: swap it for a HIP twin and `03b_main.cpp` does
not change by one character.

## Reporting back

For each spike: the toolchain versions (`nvc++ --version`, `nvcc --version`,
`g++ --version`), the device header block the spikes print, PASS/FAIL, the timings,
and — for anything that failed to build — the verbatim compiler output.

The three numbers that actually steer the design:

| number | from | decides |
|---|---|---|
| `pageableMemoryAccess` | any spike's header | unified vs discrete row of the backend table |
| spike 01 `A/B` ratio | spike 01 | whether prefetch alone gets us device residency |
| spike 02 build outcome | `build/s02_build.log` | whether managed memory can be dropped entirely |
